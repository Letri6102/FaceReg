"""Kiểm tra chất lượng ảnh đăng ký (ảnh 4x6). Ảnh mẫu kém là nguyên nhân số 1 gây từ chối nhầm."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .face_engine import FaceInfo, brightness_score, crop_face, face_geometry, sharpness_score


@dataclass
class QualityReport:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reasons": self.reasons, "metrics": self.metrics}


def pick_main_face(faces: list[FaceInfo], area_ratio: float) -> tuple[FaceInfo | None, int]:
    """Trả về mặt lớn nhất và số mặt 'đáng kể' (diện tích >= area_ratio · mặt lớn nhất)."""
    if not faces:
        return None, 0
    main = max(faces, key=lambda f: f.area)
    significant = sum(1 for f in faces if f.area >= area_ratio * main.area)
    return main, significant


def check_enrollment_quality(img: np.ndarray, faces: list[FaceInfo], s: Settings) -> tuple[QualityReport, FaceInfo | None]:
    main, n_sig = pick_main_face(faces, s.multi_face_area_ratio)
    if main is None:
        return QualityReport(False, ["Không phát hiện được khuôn mặt trong ảnh."]), None

    reasons: list[str] = []
    if n_sig > 1:
        reasons.append(f"Ảnh có {n_sig} khuôn mặt. Ảnh đăng ký chỉ được có một người.")

    geom = face_geometry(main.kps)
    if main.pose_deg is not None:
        pitch, yaw, roll = main.pose_deg
    else:
        pitch, yaw, roll = 0.0, geom.yaw_deg_estimate(), geom.roll_deg

    crop = crop_face(img, main.bbox)
    sharp = sharpness_score(crop)
    bright = brightness_score(crop)

    metrics = {
        "det_score": round(main.det_score, 3),
        "face_px": round(main.min_side, 1),
        "yaw_deg": round(float(yaw), 1),
        "pitch_deg": round(float(pitch), 1),
        "roll_deg": round(float(roll), 1),
        "sharpness": round(sharp, 1),
        "brightness": round(bright, 1),
        "faces_detected": len(faces),
    }

    if main.det_score < s.enroll_min_det_score:
        reasons.append(f"Độ tin cậy phát hiện mặt thấp ({main.det_score:.2f}). Ảnh có thể bị mờ, che hoặc lóa.")
    if main.min_side < s.enroll_min_face_px:
        reasons.append(f"Khuôn mặt quá nhỏ ({main.min_side:.0f}px, cần ≥ {s.enroll_min_face_px}px). Dùng ảnh độ phân giải cao hơn.")
    if abs(yaw) > s.enroll_max_yaw:
        reasons.append(f"Mặt bị nghiêng ngang {abs(yaw):.0f}° (tối đa {s.enroll_max_yaw:.0f}°). Cần ảnh chụp thẳng.")
    if abs(pitch) > s.enroll_max_pitch:
        reasons.append(f"Mặt bị ngẩng/cúi {abs(pitch):.0f}° (tối đa {s.enroll_max_pitch:.0f}°).")
    if sharp < s.enroll_min_sharpness:
        reasons.append(f"Ảnh bị mờ (độ nét {sharp:.0f}, cần ≥ {s.enroll_min_sharpness:.0f}).")
    if bright < s.enroll_min_brightness:
        reasons.append(f"Ảnh quá tối (độ sáng {bright:.0f}).")
    if bright > s.enroll_max_brightness:
        reasons.append(f"Ảnh quá sáng/bị lóa (độ sáng {bright:.0f}).")

    return QualityReport(ok=not reasons, reasons=reasons, metrics=metrics), main
