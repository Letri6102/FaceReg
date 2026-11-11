"""Kiểm thử luồng nghiệp vụ bằng engine giả lập (không cần InsightFace).

Chạy:  python tests/test_services.py      (hoặc: python -m pytest tests)
Ảnh giả mã hóa "danh tính", tư thế và số khuôn mặt vào màu của vài ô pixel ở góc ảnh.
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402
from app.db import Database  # noqa: E402
from app.face_engine import FaceInfo, face_geometry, l2_normalize  # noqa: E402
from app.services import FaceRegService, ServiceError  # noqa: E402

_rng = np.random.default_rng(0)
# Mỗi "mô hình" giả có không gian đặc trưng riêng
BASES = {m: {i: l2_normalize(_rng.normal(size=512)) for i in range(1, 10)} for m in ("m1", "m2", "m3")}
MIX_ID = 20  # danh tính "lưng chừng": cos ≈ 0.37 với người 1


def person_embedding(pid: int, model: str = "m1") -> np.ndarray:
    b = BASES[model]
    if pid == MIX_ID:
        b2 = b[2] - np.dot(b[2], b[1]) * b[1]
        return l2_normalize(0.37 * b[1] + np.sqrt(1 - 0.37 ** 2) * l2_normalize(b2))
    return b[pid]


def make_image(pid: int, yaw: float = 0.0, pitch: float = 0.5, faces: int = 1) -> np.ndarray:
    img = _rng.integers(60, 200, size=(480, 640, 3), dtype=np.uint8)
    img[0:24, 0:24] = (pid * 10, int(128 + yaw * 200), int(pitch * 255))
    img[0:24, 24:48] = (faces * 50, 0, 0)
    return img


def to_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def to_bytes(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


class StubEngine:
    def __init__(self, models=("m1", "m2")):
        self.model_names = list(models)
        self.embed_calls = 0

    def embed(self, img, face, models=None):
        self.embed_calls += 1
        return {m: l2_normalize(person_embedding(face._pid, m) + _rng.normal(scale=0.01, size=512))
                for m in self.model_names if models is None or m in models}

    def analyze(self, img: np.ndarray) -> list[FaceInfo]:
        # Bỏ qua viền trắng (khi service thêm padding cho ảnh 4x6)
        nz = np.argwhere((img < 250).any(axis=2))
        y0, x0 = (nz.min(axis=0) if len(nz) else (0, 0))
        img = img[y0:, x0:]
        b, g, r = img[4:20, 4:20].reshape(-1, 3).mean(axis=0)
        n = int(round(img[4:20, 28:44, 0].mean() / 50))
        pid = int(round(b / 10))
        yaw = (g - 128) / 200
        pitch = r / 255
        out = []
        for k in range(n):
            cx, cy = 320 + k * 5 + x0, 200 + y0
            kps = np.array([[cx - 50, cy], [cx + 50, cy], [cx + yaw * 100, cy + pitch * 100],
                            [cx - 35, cy + 100], [cx + 35, cy + 100]], dtype=np.float32)
            emb = l2_normalize(person_embedding(pid) + _rng.normal(scale=0.01, size=512))
            f = FaceInfo(bbox=np.array([cx - 100, cy - 100, cx + 100, cy + 160], np.float32), kps=kps,
                         det_score=0.9, embedding=emb, pose_deg=(0.0, yaw * 40, 0.0))
            f._pid = pid
            out.append(f)
        return out


POSES = {"center": (0.0, 0.5), "left": (0.35, 0.5), "right": (-0.35, 0.5), "up": (0.0, 0.3)}


def make_service(models=("m1", "m2"), data_dir: str | None = None, weights: str = "") -> FaceRegService:
    tmp = data_dir or tempfile.mkdtemp(prefix="facereg_test_")
    os.environ["DATA_DIR"] = tmp
    os.environ["MODEL_ROOT"] = os.path.join(tmp, "models")
    os.environ["FUSION_WEIGHTS"] = weights
    s = load_settings()
    return FaceRegService(s, Database(s.db_path), StubEngine(models))


def run_challenge(svc: FaceRegService, sid: str, steps: list[str], pid: int, hold: int) -> None:
    for st in steps:
        yaw, pitch = POSES[st]
        for _ in range(hold):
            r = svc.process_frame(sid, to_data_url(make_image(pid, yaw, pitch)))
        assert r["step_passed"], (st, r)


def test_geometry_directions():
    kps = np.array([[270, 200], [370, 200], [355, 250], [285, 300], [355, 300]], np.float32)
    g = face_geometry(kps)
    assert abs(g.yaw_ratio - 0.35) < 1e-6 and abs(g.pitch_ratio - 0.5) < 1e-6


def test_enroll():
    svc = make_service()
    r = svc.enroll("SV001", "Nguyễn Văn A", to_bytes(make_image(1)))
    assert r["ok"], r
    bad = svc.enroll("SV002", "B", to_bytes(make_image(2, faces=0)))
    assert not bad["ok"] and "Không phát hiện" in bad["quality"]["reasons"][0]
    two = svc.enroll("SV003", "C", to_bytes(make_image(3, faces=2)))
    assert not two["ok"] and any("2 khuôn mặt" in x for x in two["quality"]["reasons"])
    try:
        svc.enroll("SV001", "A", to_bytes(make_image(1)))
        raise AssertionError("phải báo trùng")
    except ServiceError as e:
        assert e.status_code == 409
    assert svc.enroll("SV001", "A2", to_bytes(make_image(1)), replace=True)["replaced"]
    try:
        svc.enroll("../hack", "x", to_bytes(make_image(1)))
        raise AssertionError("phải chặn mã không hợp lệ")
    except ServiceError:
        pass


def test_checkin_accept_and_monitor():
    svc = make_service()
    svc.enroll("SV001", "Nguyễn Văn A", to_bytes(make_image(1)))
    start = svc.start_checkin("SV001", "EXAM1")
    steps = [x["type"] for x in start["steps"]]
    assert steps[0] == "center" and steps[-1] == "center" and len(steps) == 5
    sid = start["session_id"]

    # Sai hướng -> không qua bước
    run_challenge(svc, sid, ["center"], 1, svc.s.challenge_hold_frames)
    nxt = steps[1]
    wrong = {"left": "right", "right": "left", "up": "left"}[nxt]
    r = svc.process_frame(sid, to_data_url(make_image(1, *POSES[wrong])))
    assert not r["step_passed"] and r["step_index"] == 1
    # Chưa xong mà finish -> lỗi
    try:
        svc.finish_checkin(sid)
        raise AssertionError("phải báo chưa hoàn thành")
    except ServiceError as e:
        assert e.code == "incomplete"
    run_challenge(svc, sid, steps[1:], 1, svc.s.challenge_hold_frames)
    res = svc.finish_checkin(sid)
    assert res["status"] == "ACCEPT", res
    assert res["monitor_token"] and "score" not in res
    assert svc.finish_checkin(sid)["checkin_id"] == res["checkin_id"]  # idempotent
    sets = svc.db.get_template_sets(svc.db.get_candidate("SV001")["id"])
    assert [st["source"] for st in sets] == ["enroll", "live"], sets
    assert set(sets[1]["embs"]) == {"m1", "m2"}
    row = svc.db.list_checkins(candidate_code="SV001")[0]
    assert set(row["scores"]) == {"m1", "m2"} and all(v > 0.9 for v in row["scores"].values()), row["scores"]

    # Giám sát
    assert svc.monitor(res["monitor_token"], to_data_url(make_image(1)))["status"] == "OK"
    try:
        svc.monitor(res["monitor_token"], to_data_url(make_image(1)))
        raise AssertionError("phải giới hạn tần suất")
    except ServiceError as e:
        assert e.status_code == 429
    svc.s.monitor_min_interval_seconds = 0
    assert svc.monitor(res["monitor_token"], to_data_url(make_image(5)))["status"] == "MISMATCH"
    assert svc.monitor(res["monitor_token"], to_data_url(make_image(1, faces=0)))["status"] == "NO_FACE"
    assert svc.monitor(res["monitor_token"], to_data_url(make_image(1, faces=2)))["status"] == "MULTI_FACE"
    alerts = svc.db.list_checkins(candidate_code="SV001")[0]["monitor_alerts"]
    assert alerts == 3, alerts


def test_impostor_rejected():
    svc = make_service()
    svc.enroll("SV001", "A", to_bytes(make_image(1)))
    start = svc.start_checkin("SV001")
    run_challenge(svc, start["session_id"], [x["type"] for x in start["steps"]], 7, svc.s.challenge_hold_frames)
    res = svc.finish_checkin(start["session_id"])
    assert res["status"] == "REJECT", res
    assert "monitor_token" not in res


def test_review_zone_then_approve():
    svc = make_service()
    svc.enroll("SV001", "A", to_bytes(make_image(1)))
    start = svc.start_checkin("SV001")
    sid = start["session_id"]
    run_challenge(svc, sid, [x["type"] for x in start["steps"]], MIX_ID, svc.s.challenge_hold_frames)
    res = svc.finish_checkin(sid)
    assert res["status"] == "REVIEW", res
    assert svc.checkin_status(sid)["status"] == "REVIEW"
    rows = svc.db.list_checkins(status="REVIEW")
    assert len(rows) == 1 and svc.s.threshold_reject <= rows[0]["score"] < svc.s.threshold_accept
    out = svc.review(rows[0]["id"], "APPROVE", "Giám thị đã kiểm tra CCCD")
    assert out["final_status"] == "ACCEPT" and out["auto_status"] == "REVIEW"
    st = svc.checkin_status(sid)
    assert st["status"] == "ACCEPT" and st["monitor_token"]


def test_multi_face_and_identity_switch_force_review():
    svc = make_service()
    svc.enroll("SV001", "A", to_bytes(make_image(1)))
    start = svc.start_checkin("SV001")
    sid = start["session_id"]
    steps = [x["type"] for x in start["steps"]]
    run_challenge(svc, sid, steps[:1], 1, svc.s.challenge_hold_frames)
    r = svc.process_frame(sid, to_data_url(make_image(1, *POSES[steps[1]], faces=2)))
    assert not r["step_passed"] and "một người" in r["feedback"]
    r = svc.process_frame(sid, to_data_url(make_image(6, *POSES[steps[1]])))  # người khác chen vào giữa chừng
    assert not r["step_passed"] and "thay đổi" in r["feedback"]
    run_challenge(svc, sid, steps[1:], 1, svc.s.challenge_hold_frames)
    res = svc.finish_checkin(sid)
    assert res["status"] == "REVIEW", res
    reasons = svc.db.list_checkins()[0]["reasons"]
    assert any("nhiều khuôn mặt" in x for x in reasons) and any("thay đổi" in x for x in reasons), reasons


def test_passive_liveness_flags_spoof():
    svc = make_service()
    svc.enroll("SV001", "A", to_bytes(make_image(1)))

    class FakeSess:
        def run(self, _out, feeds):
            assert next(iter(feeds.values())).shape == (1, 3, 80, 80)
            return [np.array([[3.0, -1.0, 0.0]], np.float32)]  # lớp 0 (giả) thắng

    svc.antispoof.models = [(FakeSess(), "input", 80, 80, 2.7)]
    start = svc.start_checkin("SV001")
    run_challenge(svc, start["session_id"], [x["type"] for x in start["steps"]], 1, svc.s.challenge_hold_frames)
    res = svc.finish_checkin(start["session_id"])
    assert res["status"] == "REVIEW", res
    row = svc.db.list_checkins()[0]
    assert row["spoof_score"] < 0.1 and any("giả mạo" in x for x in row["reasons"])
    # Duyệt cho qua trường hợp nghi giả mạo -> KHÔNG thêm mẫu webcam
    svc.review(row["id"], "APPROVE")
    srcs = [st["source"] for st in svc.db.get_template_sets(row["candidate_id"])]
    assert "live" not in srcs


def test_expired_session():
    svc = make_service()
    svc.enroll("SV001", "A", to_bytes(make_image(1)))
    start = svc.start_checkin("SV001")
    sess = svc.db.get_session(start["session_id"])
    with svc.db.connect() as c:
        c.execute("UPDATE checkin_sessions SET expires_at = 0 WHERE id = ?", (sess["id"],))
    try:
        svc.process_frame(start["session_id"], to_data_url(make_image(1)))
        raise AssertionError("phải hết hạn")
    except ServiceError as e:
        assert e.status_code == 410
    try:
        svc.start_checkin("KHONGCO")
        raise AssertionError("phải báo không tìm thấy")
    except ServiceError as e:
        assert e.status_code == 404


def test_mirrored_camera_setting():
    svc = make_service()
    from app.liveness import check_pose
    g = face_geometry(np.array([[270, 200], [370, 200], [355, 250], [285, 300], [355, 300]], np.float32))
    assert check_pose("left", g, 0.5, svc.s).passed
    svc.s.camera_mirrored = True
    assert check_pose("right", g, 0.5, svc.s).passed and not check_pose("left", g, 0.5, svc.s).passed


def test_add_model_later_backfills_from_photo():
    """Đăng ký khi chỉ có m1, sau đó bật thêm m2: hệ thống tự tính m2 từ ảnh 4x6 đã lưu."""
    svc1 = make_service(models=("m1",))
    data_dir = str(svc1.s.data_dir)
    svc1.enroll("SV001", "A", to_bytes(make_image(1)))
    st = svc1.start_checkin("SV001")
    run_challenge(svc1, st["session_id"], [x["type"] for x in st["steps"]], 1, svc1.s.challenge_hold_frames)
    assert svc1.finish_checkin(st["session_id"])["status"] == "ACCEPT"  # tạo 1 mẫu live chỉ có m1

    svc2 = make_service(models=("m1", "m2"), data_dir=data_dir)
    cid = svc2.db.get_candidate("SV001")["id"]
    st = svc2.start_checkin("SV001")
    run_challenge(svc2, st["session_id"], [x["type"] for x in st["steps"]], 1, svc2.s.challenge_hold_frames)
    assert svc2.finish_checkin(st["session_id"])["status"] == "ACCEPT"
    sets = svc2.db.get_template_sets(cid)
    assert set(sets[0]["embs"]) == {"m1", "m2"}, "bộ enroll phải được bổ sung m2"
    row = svc2.db.list_checkins(candidate_code="SV001")[0]
    assert set(row["scores"]) == {"m1", "m2"}
    # Người lạ vẫn bị từ chối với cấu hình mới
    st = svc2.start_checkin("SV001")
    run_challenge(svc2, st["session_id"], [x["type"] for x in st["steps"]], 4, svc2.s.challenge_hold_frames)
    assert svc2.finish_checkin(st["session_id"])["status"] == "REJECT"


def test_backfill_all_and_zero_weight():
    svc1 = make_service(models=("m1",))
    for i in (1, 2, 3):
        svc1.enroll(f"SV00{i}", "x", to_bytes(make_image(i)))
    svc2 = make_service(models=("m1", "m3"), data_dir=str(svc1.s.data_dir))
    assert svc2.backfill_all() == {"updated": 3, "failed": 0}
    assert svc2.backfill_all() == {"updated": 0, "failed": 0}
    # Trọng số 0 cho m1 -> chỉ dùng m3
    svc3 = make_service(models=("m1", "m3"), data_dir=str(svc1.s.data_dir), weights="m1:0")
    assert svc3.active == ["m3"]
    score, per, _ = svc3.match(svc3.db.get_candidate("SV002")["id"], {"m3": person_embedding(2, "m3")})
    assert set(per) == {"m3"} and score > 0.95
    try:
        make_service(models=("m1",), weights="khongco:1")
        raise AssertionError("phải báo tên mô hình sai")
    except ValueError:
        pass


def test_migrate_old_database():
    import sqlite3
    import time as _t

    tmp = tempfile.mkdtemp(prefix="facereg_old_")
    db_path = Path(tmp) / "facereg.db"
    (Path(tmp) / "candidates").mkdir()
    photo = Path(tmp) / "candidates" / "SV001_photo.jpg"
    photo.write_bytes(cv2.imencode(".jpg", make_image(1), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE candidates (id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL UNIQUE, full_name TEXT NOT NULL DEFAULT '',
            photo_path TEXT NOT NULL, face_path TEXT NOT NULL, quality_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE templates (id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id INTEGER NOT NULL, source TEXT NOT NULL,
            embedding BLOB NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE checkins (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL UNIQUE, candidate_id INTEGER NOT NULL,
            exam_code TEXT NOT NULL DEFAULT '', auto_status TEXT NOT NULL, final_status TEXT NOT NULL, score REAL NOT NULL,
            spoof_score REAL, reasons_json TEXT NOT NULL DEFAULT '[]', captured_path TEXT NOT NULL DEFAULT '', monitor_token TEXT,
            reviewer_note TEXT NOT NULL DEFAULT '', reviewed_at REAL, client_ip TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL);
    """)
    con.execute("INSERT INTO candidates (code, full_name, photo_path, face_path, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                ("SV001", "A", str(photo), str(photo), _t.time(), _t.time()))
    con.execute("INSERT INTO templates (candidate_id, source, embedding, created_at) VALUES (1, 'enroll', ?, ?)",
                (np.zeros(512, np.float32).tobytes(), _t.time()))
    con.commit()
    con.close()

    svc = make_service(models=("m1", "m2"), data_dir=tmp)
    st = svc.start_checkin("SV001")
    run_challenge(svc, st["session_id"], [x["type"] for x in st["steps"]], 1, svc.s.challenge_hold_frames)
    assert svc.finish_checkin(st["session_id"])["status"] == "ACCEPT"
    assert svc.db.list_checkins()[0]["scores"]


def test_alignment_and_fusion_math():
    from app.face_engine import ARCFACE_DST, align_face, fused_score, similarity_transform

    th, sc, t = np.deg2rad(-12), 1.7, np.array([30.0, 5.0])
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    src = (ARCFACE_DST - t) @ np.linalg.inv(sc * r).T
    m = similarity_transform(src, ARCFACE_DST)
    assert np.abs(src @ m[:, :2].T + m[:, 2] - ARCFACE_DST).max() < 1e-9
    assert align_face(make_image(1), src).shape == (112, 112, 3)
    a, b = person_embedding(1, "m1"), person_embedding(1, "m2")
    score, per = fused_score({"m1": a, "m2": b}, {"m1": a, "m2": -b}, {"m1": 3.0, "m2": 1.0})
    assert abs(per["m1"] - 1) < 1e-5 and abs(per["m2"] + 1) < 1e-5 and abs(score - 0.5) < 1e-5
    assert fused_score({"m1": a}, {"m1": a, "m2": b}, {"m1": 1.0, "m2": 1.0}) is None


def test_calibration_metrics():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from calibrate_threshold import eer, threshold_at_fmr

    rng = np.random.default_rng(1)
    gen = rng.normal(0.6, 0.1, 5000)
    imp = rng.normal(0.05, 0.08, 50000)
    t = threshold_at_fmr(imp, 1e-3)
    assert np.mean(imp >= t) <= 1e-3 and np.mean(imp >= t - 0.01) > 0
    e, te = eer(gen, imp)
    assert 0 < e < 0.01 and 0.2 < te < 0.45, (e, te)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} test đều đạt.")
