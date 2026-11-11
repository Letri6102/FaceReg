"""Chống giả mạo (liveness).

1. Chủ động: hệ thống yêu cầu một chuỗi động tác NGẪU NHIÊN (quay trái/phải, ngẩng lên) và kiểm tra
   từng khung hình phía server. Ảnh in hoặc video quay sẵn khó đáp ứng đúng chuỗi ngẫu nhiên.
2. Thụ động (tùy chọn): mô hình ONNX phân loại mặt thật / ảnh in / màn hình
   (ví dụ MiniFASNet của dự án Silent-Face-Anti-Spoofing). Bật bằng biến ANTISPOOF_MODELS.
"""
from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .face_engine import FaceGeometry

log = logging.getLogger("facereg.liveness")

STEP_TEXT = {
    "center": "Nhìn thẳng vào camera",
    "left": "Từ từ quay mặt sang TRÁI",
    "right": "Từ từ quay mặt sang PHẢI",
    "up": "Ngẩng mặt lên một chút",
}
TURNS = ("left", "right", "up")


def generate_steps(n_turns: int) -> list[str]:
    """Chuỗi bước: center → n động tác ngẫu nhiên (không lặp liền nhau) → center."""
    rng = secrets.SystemRandom()
    turns: list[str] = []
    for _ in range(max(n_turns, 1)):
        options = [t for t in TURNS if not turns or t != turns[-1]]
        turns.append(rng.choice(options))
    return ["center", *turns, "center"]


@dataclass
class PoseCheck:
    passed: bool
    feedback: str


def check_pose(step: str, geom: FaceGeometry, baseline_pitch: float | None, s: Settings) -> PoseCheck:
    """Kiểm tra tư thế khuôn mặt có đúng yêu cầu của bước hiện tại không.

    Khung hình gửi lên là ảnh GỐC của camera (không lật gương). Khi người thi quay sang trái của họ,
    mũi lệch sang bên phải ảnh => yaw_ratio > 0. Nếu camera/driver tự lật ảnh, đặt CAMERA_MIRRORED=true.
    """
    subject_yaw = -geom.yaw_ratio if s.camera_mirrored else geom.yaw_ratio  # >0: quay sang trái của người thi
    base = baseline_pitch if baseline_pitch is not None else geom.pitch_ratio

    if step == "center":
        if abs(subject_yaw) > s.yaw_frontal_ratio:
            return PoseCheck(False, "Hãy nhìn thẳng vào camera")
        if baseline_pitch is not None and abs(geom.pitch_ratio - baseline_pitch) > s.pitch_center_tolerance:
            return PoseCheck(False, "Giữ đầu thẳng, nhìn vào camera")
        return PoseCheck(True, "Tốt")
    if step == "left":
        if subject_yaw >= s.yaw_turn_ratio:
            return PoseCheck(True, "Tốt")
        if subject_yaw <= -s.yaw_frontal_ratio:
            return PoseCheck(False, "Ngược hướng rồi, hãy quay sang TRÁI")
        return PoseCheck(False, "Quay thêm sang TRÁI")
    if step == "right":
        if subject_yaw <= -s.yaw_turn_ratio:
            return PoseCheck(True, "Tốt")
        if subject_yaw >= s.yaw_frontal_ratio:
            return PoseCheck(False, "Ngược hướng rồi, hãy quay sang PHẢI")
        return PoseCheck(False, "Quay thêm sang PHẢI")
    if step == "up":
        if abs(subject_yaw) > s.yaw_turn_ratio:
            return PoseCheck(False, "Giữ mặt chính diện và ngẩng lên")
        if base - geom.pitch_ratio >= s.pitch_up_delta:
            return PoseCheck(True, "Tốt")
        return PoseCheck(False, "Ngẩng mặt lên thêm")
    return PoseCheck(False, "Bước không hợp lệ")


# ---------------------------------------------------------------------------
# Liveness thụ động bằng ONNX
# ---------------------------------------------------------------------------
def _parse_model_name(name: str) -> tuple[float, int | None, int | None]:
    """Theo quy ước tên file của Silent-Face-Anti-Spoofing: '2.7_80x80_MiniFASNetV2', '4_0_0_80x80_MiniFASNetV1SE'."""
    scale = 1.0
    h = w = None
    m = re.search(r"(\d+)x(\d+)", name)
    if m:
        h, w = int(m.group(1)), int(m.group(2))
        head = name[: m.start()].strip("_")
        if head:
            try:
                scale = float(head.split("_")[0])
            except ValueError:
                scale = 1.0
    return scale, h, w


def _crop_scaled(img: np.ndarray, bbox: np.ndarray, scale: float, out_w: int, out_h: int) -> np.ndarray:
    src_h, src_w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
    nw, nh = bw * scale, bh * scale
    cx, cy = x1 + bw / 2, y1 + bh / 2
    lx, ly, rx, ry = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if lx < 0:
        rx -= lx
        lx = 0
    if ly < 0:
        ry -= ly
        ly = 0
    if rx > src_w - 1:
        lx -= rx - src_w + 1
        rx = src_w - 1
    if ry > src_h - 1:
        ly -= ry - src_h + 1
        ry = src_h - 1
    crop = img[int(max(ly, 0)): int(ry) + 1, int(max(lx, 0)): int(rx) + 1]
    return cv2.resize(crop, (out_w, out_h))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class PassiveAntiSpoof:
    def __init__(self, s: Settings):
        self.models: list[tuple] = []
        self.input_scale = s.antispoof_input_scale
        self.real_index = s.antispoof_real_index
        paths = [p for p in s.antispoof_models]
        if not paths:
            return
        import onnxruntime as ort

        for p in paths:
            p = Path(p)
            if not p.exists():
                log.warning("Không tìm thấy mô hình anti-spoof: %s (bỏ qua)", p)
                continue
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            scale, h, w = _parse_model_name(p.stem)
            shape = inp.shape
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                h, w = shape[2], shape[3]
            h, w = h or 80, w or 80
            self.models.append((sess, inp.name, h, w, scale))
            log.info("Đã nạp anti-spoof %s (scale=%.1f, input=%dx%d)", p.name, scale, w, h)

    @property
    def enabled(self) -> bool:
        return bool(self.models)

    def real_probability(self, img_bgr: np.ndarray, bbox: np.ndarray) -> float | None:
        if not self.models:
            return None
        total = 0.0
        for sess, name, h, w, scale in self.models:
            crop = _crop_scaled(img_bgr, bbox, scale, w, h).astype(np.float32) * self.input_scale
            tensor = np.transpose(crop, (2, 0, 1))[None, ...]
            out = np.asarray(sess.run(None, {name: tensor})[0]).reshape(-1)
            probs = out if (np.all(out >= 0) and abs(out.sum() - 1.0) < 1e-3) else _softmax(out)
            total += float(probs[self.real_index]) if self.real_index < len(probs) else 0.0
        return total / len(self.models)
