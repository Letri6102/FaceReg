"""Nghiệp vụ chính, độc lập với framework web: đăng ký, check-in (liveness + so khớp 1:1), duyệt, giám sát."""
from __future__ import annotations

import logging
import re
import secrets
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from .config import Settings
from .db import Database
from .face_engine import (
    FaceEngine,
    ImageDecodeError,
    cosine,
    crop_face,
    decode_data_url,
    decode_image_bytes,
    detect_for_enroll,
    encode_jpeg,
    face_geometry,
    fused_score,
    l2_normalize,
    parse_weights,
    sharpness_score,
)
from .liveness import STEP_TEXT, PassiveAntiSpoof, check_pose, generate_steps
from .quality import check_enrollment_quality, pick_main_face

log = logging.getLogger("facereg")

CODE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

STATUS_MESSAGES = {
    "ACCEPT": "Xác thực thành công. Mời bạn vào thi.",
    "REVIEW": "Hệ thống cần giám thị xác nhận thêm. Vui lòng chờ trong giây lát.",
    "REJECT": "Xác thực không thành công. Vui lòng liên hệ giám thị.",
}


class ServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "bad_request"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def normalize_code(code: str) -> str:
    code = (code or "").strip()
    if not CODE_RE.match(code):
        raise ServiceError("Mã thí sinh chỉ gồm chữ, số, dấu chấm, gạch dưới, gạch ngang (tối đa 64 ký tự).",
                           code="invalid_code")
    return code


class FaceRegService:
    def __init__(self, settings: Settings, db: Database, engine: FaceEngine, antispoof: PassiveAntiSpoof | None = None):
        self.s = settings
        self.db = db
        self.engine = engine
        self.antispoof = antispoof or PassiveAntiSpoof(settings)
        self.models: list[str] = list(engine.model_names)
        self.weights = parse_weights(settings.fusion_weights, self.models)
        self.active = list(self.weights)  # các mô hình thực sự dùng để tính điểm
        log.info("Mô hình nhận diện: %s | trọng số: %s | flip TTA: %s", self.models, self.weights, settings.flip_tta)
        if not self.antispoof.enabled:
            log.warning("Liveness thụ động CHƯA bật (ANTISPOOF_MODELS trống). Chỉ dùng thử thách chủ động.")

    # ==================================================================
    # Đăng ký
    # ==================================================================
    def enroll(self, code: str, full_name: str, image_bytes: bytes, replace: bool = False) -> dict:
        code = normalize_code(code)
        full_name = (full_name or "").strip()[:200]
        existing = self.db.get_candidate(code)
        if existing and not replace:
            raise ServiceError(f"Thí sinh {code} đã được đăng ký. Chọn 'Thay ảnh' để cập nhật.", 409, "exists")
        try:
            # Chuẩn hóa về JPEG chất lượng cao rồi xử lý chính bản đã lưu, để sau này tính lại
            # (khi thêm mô hình mới) cho ra đúng cùng một đầu vào.
            photo_jpeg = encode_jpeg(decode_image_bytes(image_bytes), 95)
            img = decode_image_bytes(photo_jpeg)
        except ImageDecodeError as e:
            raise ServiceError(str(e), code="bad_image") from e

        faces, det_img = detect_for_enroll(self.engine, img)
        report, face = check_enrollment_quality(det_img, faces, self.s)
        if not report.ok or face is None:
            return {"ok": False, "code": code, "quality": report.to_dict()}

        embeddings = self.engine.embed(det_img, face)  # lưu đủ mọi mô hình đã nạp, kể cả trọng số 0
        photo_path = self.s.candidates_dir / f"{code}_photo.jpg"
        face_path = self.s.candidates_dir / f"{code}_face.jpg"
        photo_path.write_bytes(photo_jpeg)
        face_path.write_bytes(encode_jpeg(crop_face(det_img, face.bbox, margin=0.25), 92))

        cid = self.db.upsert_candidate(code, full_name, str(photo_path), str(face_path), report.metrics, embeddings)
        log.info("Đăng ký %s (id=%s) thành công", code, cid)
        return {"ok": True, "code": code, "candidate_id": cid, "replaced": bool(existing), "quality": report.to_dict()}

    def delete_candidate(self, code: str) -> dict:
        cand = self.db.delete_candidate(normalize_code(code))
        if not cand:
            raise ServiceError("Không tìm thấy thí sinh.", 404, "not_found")
        for key in ("photo_path", "face_path"):
            try:
                Path(cand[key]).unlink(missing_ok=True)
            except OSError:
                pass
        return {"ok": True}

    # ==================================================================
    # So khớp 1:1
    # ==================================================================
    def _backfill_enroll_set(self, candidate_id: int, set_id: int, missing: list[str]) -> dict[str, np.ndarray]:
        """Tính vector cho mô hình mới thêm từ ảnh 4x6 đã lưu (không cần đăng ký lại)."""
        cand = self.db.get_candidate_by_id(candidate_id)
        if not cand or not Path(cand["photo_path"]).exists():
            return {}
        img = decode_image_bytes(Path(cand["photo_path"]).read_bytes())
        faces, det_img = detect_for_enroll(self.engine, img)
        main, _ = pick_main_face(faces, self.s.multi_face_area_ratio)
        if main is None:
            return {}
        embs = self.engine.embed(det_img, main, missing)
        if embs:
            self.db.add_embeddings_to_set(candidate_id, set_id, "enroll", embs)
            log.info("Bổ sung vector %s cho thí sinh id=%s", list(embs), candidate_id)
        return embs

    def backfill_all(self) -> dict:
        """Bổ sung vector cho mọi thí sinh còn thiếu mô hình (dùng sau khi đổi EXTRA_MODELS)."""
        done = skipped = 0
        for cid in self.db.all_candidate_ids():
            for st in self.db.get_template_sets(cid):
                if st["source"] != "enroll":
                    continue
                missing = [m for m in self.models if m not in st["embs"]]
                if not missing:
                    continue
                if self._backfill_enroll_set(cid, st["set_id"], missing):
                    done += 1
                else:
                    skipped += 1
        return {"updated": done, "failed": skipped}

    def match(self, candidate_id: int, probe: dict[str, np.ndarray]) -> tuple[float, dict[str, float], str]:
        """So khớp 1:1. Trả về (điểm gộp cao nhất qua các bộ mẫu, điểm từng mô hình, nguồn bộ mẫu)."""
        best: tuple[float, dict[str, float], str] | None = None
        for st in self.db.get_template_sets(candidate_id):
            embs = st["embs"]
            missing = [m for m in self.active if m not in embs]
            if missing:
                if st["source"] != "enroll":
                    continue  # mẫu webcam cũ thiếu mô hình mới: bỏ qua
                embs = {**embs, **self._backfill_enroll_set(candidate_id, st["set_id"], missing)}
            res = fused_score(probe, embs, self.weights)
            if res is None:
                continue
            if best is None or res[0] > best[0]:
                best = (res[0], res[1], st["source"])
        if best is None:
            raise ServiceError("Thí sinh chưa có dữ liệu khuôn mặt hợp lệ.", 409, "no_template")
        return best

    # ==================================================================
    # Check-in
    # ==================================================================
    def start_checkin(self, code: str, exam_code: str = "", client_ip: str = "") -> dict:
        code = normalize_code(code)
        cand = self.db.get_candidate(code)
        if not cand:
            raise ServiceError("Không tìm thấy thí sinh với mã này. Vui lòng liên hệ giám thị.", 404, "not_found")
        steps = generate_steps(self.s.challenge_turns)
        sid = secrets.token_urlsafe(24)
        state = {
            "steps": steps,
            "idx": 0,
            "frames": 0,
            "hold": 0,
            "baseline_pitch": None,
            "ref_emb": None,
            "center_embs": [],
            "spoof_scores": [],
            "flags": {"multi_face": 0, "identity_switch": 0},
            "best_q": -1.0,
        }
        self.db.create_session(sid, cand["id"], (exam_code or "").strip()[:64], state,
                               self.s.session_ttl_seconds, client_ip)
        return {
            "session_id": sid,
            "steps": [{"type": st, "instruction": STEP_TEXT[st]} for st in steps],
            "expires_in": self.s.session_ttl_seconds,
            "hold_frames": self.s.challenge_hold_frames,
        }

    def _active_session(self, sid: str) -> dict:
        sess = self.db.get_session(sid)
        if not sess:
            raise ServiceError("Phiên xác thực không tồn tại.", 404, "no_session")
        if sess["status"] == "finished":
            raise ServiceError("Phiên xác thực đã kết thúc.", 409, "finished")
        if sess["status"] != "active" or time.time() > sess["expires_at"]:
            if sess["status"] == "active":
                self.db.update_session(sid, sess["state"], "expired")
            raise ServiceError("Phiên xác thực đã hết hạn. Vui lòng bắt đầu lại.", 410, "expired")
        return sess

    def _frame_response(self, sess: dict, state: dict, feedback: str, passed: bool) -> dict:
        idx = state["idx"]
        steps = state["steps"]
        done = idx >= len(steps)
        cur = steps[min(idx, len(steps) - 1)]
        return {
            "step_index": idx,
            "total_steps": len(steps),
            "step": None if done else cur,
            "instruction": "Hoàn tất" if done else STEP_TEXT[cur],
            "feedback": feedback,
            "step_passed": passed,
            "hold": state["hold"],
            "done": done,
            "expires_in": max(0, int(sess["expires_at"] - time.time())),
        }

    def process_frame(self, sid: str, image_data_url: str) -> dict:
        sess = self._active_session(sid)
        state = sess["state"]
        if state["idx"] >= len(state["steps"]):
            return self._frame_response(sess, state, "Đã hoàn tất các bước", False)

        state["frames"] += 1
        if state["frames"] > self.s.session_max_frames:
            self.db.update_session(sid, state, "expired")
            raise ServiceError("Quá nhiều khung hình. Vui lòng bắt đầu lại.", 429, "too_many_frames")

        def reply(feedback: str, passed: bool = False) -> dict:
            if not passed:
                state["hold"] = 0
            self.db.update_session(sid, state)
            return self._frame_response(sess, state, feedback, passed)

        try:
            img = decode_data_url(image_data_url)
        except ImageDecodeError:
            return reply("Khung hình lỗi, đang thử lại...")

        faces = self.engine.analyze(img)
        main, n_sig = pick_main_face(faces, self.s.multi_face_area_ratio)
        if main is None:
            return reply("Không thấy khuôn mặt. Hãy đưa mặt vào khung hình.")
        if n_sig > 1:
            state["flags"]["multi_face"] += 1
            return reply("Chỉ một người được xuất hiện trước camera.")
        if main.min_side < self.s.checkin_min_face_px:
            return reply("Hãy lại gần camera hơn.")

        emb = l2_normalize(main.embedding)
        if state["ref_emb"] is not None:
            if cosine(emb, np.asarray(state["ref_emb"], dtype=np.float32)) < self.s.identity_consistency_min:
                state["flags"]["identity_switch"] += 1
                return reply("Khuôn mặt thay đổi so với lúc bắt đầu. Hãy giữ nguyên vị trí.")

        step = state["steps"][state["idx"]]
        geom = face_geometry(main.kps)
        pc = check_pose(step, geom, state["baseline_pitch"], self.s)
        if not pc.passed:
            return reply(pc.feedback)

        state["hold"] += 1
        if state["hold"] < self.s.challenge_hold_frames:
            self.db.update_session(sid, state)
            return self._frame_response(sess, state, "Giữ nguyên...", False)

        # ---- Bước đạt ----
        state["hold"] = 0
        if step == "center":
            if state["ref_emb"] is None:
                state["ref_emb"] = emb.tolist()
                state["baseline_pitch"] = geom.pitch_ratio
            state["center_embs"].append({m: v.tolist() for m, v in self.engine.embed(img, main, self.active).items()})
            spoof = self.antispoof.real_probability(img, main.bbox)
            if spoof is not None:
                state["spoof_scores"].append(spoof)
            q = main.det_score * min(sharpness_score(crop_face(img, main.bbox)) / 100.0, 1.0)
            if q > state["best_q"]:
                state["best_q"] = q
                (self.s.checkins_dir / f"{sid}.jpg").write_bytes(encode_jpeg(img, 88))
        state["idx"] += 1
        self.db.update_session(sid, state)
        return self._frame_response(sess, state, "Tốt!", True)

    def finish_checkin(self, sid: str) -> dict:
        existing = self.db.get_checkin_by_session(sid)
        if existing:
            return self._public_result(existing)
        sess = self._active_session(sid)
        state = sess["state"]
        if state["idx"] < len(state["steps"]) or not state["center_embs"]:
            raise ServiceError("Chưa hoàn thành các bước xác thực.", 409, "incomplete")

        centers = state["center_embs"]
        probe = {m: l2_normalize(np.mean([np.asarray(c[m], dtype=np.float32) for c in centers], axis=0))
                 for m in self.active}
        score, per_model, source = self.match(sess["candidate_id"], probe)

        reasons: list[str] = []
        spoof = float(np.mean(state["spoof_scores"])) if state["spoof_scores"] else None
        spoof_flag = spoof is not None and spoof < self.s.antispoof_threshold
        if spoof_flag:
            reasons.append(f"Nghi ngờ giả mạo (điểm liveness {spoof:.2f} < {self.s.antispoof_threshold:.2f}).")
        if state["flags"]["multi_face"] > 0:
            reasons.append(f"Phát hiện nhiều khuôn mặt trong {state['flags']['multi_face']} khung hình.")
        if state["flags"]["identity_switch"] > 0:
            reasons.append(f"Khuôn mặt thay đổi giữa các bước ({state['flags']['identity_switch']} lần).")
        k = self.active[0]
        if len(centers) > 1 and cosine(np.asarray(centers[0][k]),
                                       np.asarray(centers[-1][k])) < self.s.identity_consistency_min:
            reasons.append("Hai lần nhìn thẳng không khớp nhau.")

        if score >= self.s.threshold_accept:
            auto = "ACCEPT"
        elif score < self.s.threshold_reject:
            auto = "REJECT"
            reasons.append("Điểm tương đồng với ảnh đăng ký thấp.")
        else:
            auto = "REVIEW"
            reasons.append("Điểm tương đồng nằm trong vùng cần giám thị xác nhận.")
        if auto == "ACCEPT" and reasons:
            auto = "REVIEW"

        token = secrets.token_urlsafe(24) if auto == "ACCEPT" else None
        captured = self.s.checkins_dir / f"{sid}.jpg"
        checkin_id = self.db.create_checkin(
            session_id=sid,
            candidate_id=sess["candidate_id"],
            exam_code=sess["exam_code"],
            auto_status=auto,
            final_status=auto,
            score=round(score, 4),
            scores={m: round(v, 4) for m, v in per_model.items()},
            spoof_score=None if spoof is None else round(spoof, 4),
            reasons=reasons,
            captured_path=str(captured) if captured.exists() else "",
            monitor_token=token,
            client_ip=sess["client_ip"],
        )
        state["probe"] = {m: v.tolist() for m, v in probe.items()}
        self.db.update_session(sid, state, "finished")

        if (auto == "ACCEPT" and self.s.progressive_enroll
                and score >= self.s.auto_template_min_score and not reasons):
            self.db.add_live_template(sess["candidate_id"], probe, self.s.max_live_templates)

        log.info("Check-in %s: %s score=%.3f %s (mẫu %s) spoof=%s", checkin_id, auto, score,
                 {m: round(v, 3) for m, v in per_model.items()}, source, spoof)
        return self._public_result(self.db.get_checkin(checkin_id))

    def _public_result(self, checkin: dict) -> dict:
        """Kết quả trả về cho thí sinh: KHÔNG kèm điểm số để tránh dò ngưỡng."""
        status = checkin["final_status"]
        out = {
            "checkin_id": checkin["id"],
            "status": status,
            "message": STATUS_MESSAGES[status],
        }
        if status == "ACCEPT":
            out["candidate_code"] = checkin.get("candidate_code")
            out["full_name"] = checkin.get("full_name")
            out["monitor_token"] = checkin.get("monitor_token")
            if self.s.checkin_success_url:
                out["redirect_url"] = self.s.checkin_success_url.format(
                    code=quote(str(checkin.get("candidate_code", ""))),
                    exam=quote(str(checkin.get("exam_code", ""))),
                    token=quote(str(checkin.get("monitor_token") or "")),
                    checkin_id=checkin["id"],
                )
        return out

    def checkin_status(self, sid: str) -> dict:
        checkin = self.db.get_checkin_by_session(sid)
        if not checkin:
            raise ServiceError("Chưa có kết quả cho phiên này.", 404, "not_found")
        return self._public_result(checkin)

    # ==================================================================
    # Giám thị duyệt
    # ==================================================================
    def review(self, checkin_id: int, decision: str, note: str = "") -> dict:
        checkin = self.db.get_checkin(checkin_id)
        if not checkin:
            raise ServiceError("Không tìm thấy lượt check-in.", 404, "not_found")
        decision = decision.upper()
        if decision not in ("APPROVE", "REJECT"):
            raise ServiceError("decision phải là APPROVE hoặc REJECT.")
        final = "ACCEPT" if decision == "APPROVE" else "REJECT"
        token = secrets.token_urlsafe(24) if final == "ACCEPT" else None
        self.db.review_checkin(checkin_id, final, (note or "")[:500], token)

        spoof_ok = checkin["spoof_score"] is None or checkin["spoof_score"] >= self.s.antispoof_threshold
        if final == "ACCEPT" and self.s.progressive_enroll and spoof_ok:
            sess = self.db.get_session(checkin["session_id"])
            probe = (sess or {}).get("state", {}).get("probe")
            if isinstance(probe, dict) and all(m in probe for m in self.active):
                self.db.add_live_template(checkin["candidate_id"],
                                          {m: np.asarray(probe[m], dtype=np.float32) for m in self.active},
                                          self.s.max_live_templates)
        return self.admin_checkin(checkin_id)

    def admin_checkin(self, checkin_id: int) -> dict:
        c = self.db.get_checkin(checkin_id)
        if not c:
            raise ServiceError("Không tìm thấy lượt check-in.", 404, "not_found")
        c.pop("monitor_token", None)
        c["has_capture"] = bool(c.get("captured_path"))
        c.pop("captured_path", None)
        return c

    # ==================================================================
    # Giám sát định kỳ trong giờ thi (không có liveness, chỉ để phát hiện đổi người / vắng mặt)
    # ==================================================================
    def monitor(self, token: str, image_data_url: str) -> dict:
        checkin = self.db.get_checkin_by_token(token or "")
        if not checkin or checkin["final_status"] != "ACCEPT":
            raise ServiceError("Token giám sát không hợp lệ.", 403, "bad_token")
        last = self.db.last_monitor_event_time(checkin["id"])
        if last is not None and time.time() - last < self.s.monitor_min_interval_seconds:
            raise ServiceError("Gửi quá nhanh.", 429, "rate_limited")
        try:
            img = decode_data_url(image_data_url)
        except ImageDecodeError as e:
            raise ServiceError(str(e), code="bad_image") from e

        faces = self.engine.analyze(img)
        main, n_sig = pick_main_face(faces, self.s.multi_face_area_ratio)
        score = None
        if main is None:
            status = "NO_FACE"
        elif n_sig > 1:
            status = "MULTI_FACE"
        else:
            score, _, _ = self.match(checkin["candidate_id"], self.engine.embed(img, main, self.active))
            status = "OK" if score >= self.s.threshold_reject else "MISMATCH"
        self.db.add_monitor_event(checkin["id"], status, None if score is None else round(score, 4))
        return {"status": status}


def load_image_file(path: str) -> bytes:
    p = Path(path)
    if not p.exists():
        raise ServiceError("Không có ảnh.", 404, "not_found")
    return p.read_bytes()


def thumbnail(jpeg_bytes: bytes, max_side: int = 320) -> bytes:
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        sc = max_side / max(h, w)
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    return encode_jpeg(img, 85)
