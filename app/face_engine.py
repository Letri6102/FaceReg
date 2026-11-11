"""Phát hiện khuôn mặt, trích vector đặc trưng (nhiều mô hình) và các hàm hình học/chất lượng ảnh.

- Phát hiện + điểm mốc + tư thế: InsightFace (SCRFD, landmark 3D-68) trong gói `buffalo_l`.
- Nhận diện: một hoặc nhiều mô hình chạy trên CÙNG ảnh mặt đã căn chỉnh 112x112:
    * mô hình chính của gói InsightFace (ArcFace R50 trong buffalo_l)
    * mô hình bổ sung khai báo trong EXTRA_MODELS, ví dụ AdaFace IR-101 (ONNX) hoặc glintr100 (antelopev2)
  Điểm cuối cùng là trung bình có trọng số của cosine từng mô hình (score-level fusion).
- Tùy chọn flip TTA: vector = chuẩn hóa(f(ảnh) + f(ảnh lật ngang)).
"""
from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger("facereg.engine")

# Vị trí chuẩn của 5 điểm mốc trên ảnh 112x112 (dùng chung cho ArcFace / AdaFace)
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)


@dataclass
class FaceInfo:
    bbox: np.ndarray            # [x1, y1, x2, y2]
    kps: np.ndarray             # 5 điểm mốc theo thứ tự InsightFace: mắt trái-ảnh, mắt phải-ảnh, mũi, mép miệng trái, phải
    det_score: float
    embedding: np.ndarray       # vector của mô hình CHÍNH (không TTA) - dùng cho kiểm tra nhanh giữa các khung hình
    pose_deg: tuple[float, float, float] | None = None  # (pitch, yaw, roll) từ landmark 3D nếu có

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


class FaceEngine(Protocol):
    model_names: list[str]

    def analyze(self, img_bgr: np.ndarray) -> list[FaceInfo]: ...

    def embed(self, img_bgr: np.ndarray, face: FaceInfo, models: list[str] | None = None) -> dict[str, np.ndarray]: ...


# ---------------------------------------------------------------------------
# Căn chỉnh khuôn mặt
# ---------------------------------------------------------------------------
def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Ước lượng phép biến đổi đồng dạng (Umeyama) src -> dst. Trả về ma trận 2x3."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_d, dst_d = src - src_mean, dst - dst_mean
    a = dst_d.T @ src_d / len(src)
    d = np.ones(2)
    if np.linalg.det(a) < 0:
        d[-1] = -1
    u, s, vt = np.linalg.svd(a)
    r = u @ np.diag(d) @ vt
    var = src_d.var(axis=0).sum()
    scale = float(s @ d) / var if var > 0 else 1.0
    t = dst_mean - scale * (r @ src_mean)
    m = np.zeros((2, 3))
    m[:, :2] = scale * r
    m[:, 2] = t
    return m


def align_face(img_bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    m = similarity_transform(np.asarray(kps)[:5], ARCFACE_DST * (size / 112.0))
    return cv2.warpAffine(img_bgr, m, (size, size), borderValue=0.0)


# ---------------------------------------------------------------------------
# Mô hình nhận diện
# ---------------------------------------------------------------------------
class Recognizer(Protocol):
    name: str

    def embed_batch(self, faces_112_bgr: list[np.ndarray]) -> np.ndarray: ...  # (N, D)


class InsightFaceRecognizer:
    """Bọc mô hình ArcFaceONNX của gói insightface (w600k_r50, glintr100...)."""

    def __init__(self, name: str, model):
        self.name = name
        self._m = model

    def embed_batch(self, faces):
        return np.asarray(self._m.get_feat(list(faces)), dtype=np.float32)


class AdaFaceRecognizer:
    """AdaFace (ONNX xuất từ checkpoint gốc). Đầu vào: BGR 112x112, chuẩn hóa (x/255 - 0.5)/0.5, NCHW."""

    def __init__(self, name: str, path: str, providers: list[str]):
        import onnxruntime as ort

        self.name = name
        self._sess = ort.InferenceSession(path, providers=providers)
        inp = self._sess.get_inputs()[0]
        self._input = inp.name
        self._fixed_batch = isinstance(inp.shape[0], int) and inp.shape[0] == 1

    def embed_batch(self, faces):
        x = np.stack([f.astype(np.float32) for f in faces])
        x = ((x / 255.0) - 0.5) / 0.5
        x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
        if self._fixed_batch:
            outs = [np.asarray(self._sess.run(None, {self._input: x[i:i + 1]})[0]).reshape(1, -1) for i in range(len(x))]
            return np.concatenate(outs, axis=0)
        return np.asarray(self._sess.run(None, {self._input: x})[0]).reshape(len(x), -1)


def parse_extra_models(spec: str) -> list[tuple[str, Path]]:
    """'adaface:models/x.onnx, insightface:models/glintr100.onnx' -> [(loại, đường dẫn)]."""
    out = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item or item.split(":", 1)[0].lower() not in ("adaface", "insightface"):
            raise ValueError(f"EXTRA_MODELS không hợp lệ: '{item}'. Dạng đúng: adaface:<đường_dẫn.onnx> hoặc insightface:<đường_dẫn.onnx>")
        kind, path = item.split(":", 1)
        out.append((kind.lower(), Path(path.strip())))
    return out


def embed_with(recognizers: list[Recognizer], img_bgr: np.ndarray, kps: np.ndarray, flip_tta: bool,
               models: list[str] | None = None) -> dict[str, np.ndarray]:
    crop = align_face(img_bgr, kps)
    batch = [crop, np.ascontiguousarray(crop[:, ::-1])] if flip_tta else [crop]
    out: dict[str, np.ndarray] = {}
    for r in recognizers:
        if models is not None and r.name not in models:
            continue
        feats = r.embed_batch(batch)
        out[r.name] = l2_normalize(feats.sum(axis=0) if flip_tta else feats[0])
    return out


class InsightFaceEngine:
    def __init__(self, model_name: str, model_root: str, det_size: int, det_thresh: float, use_gpu: bool,
                 extra_models: list[tuple[str, Path]] | None = None, flip_tta: bool = True):
        from insightface.app import FaceAnalysis  # import muộn để phần còn lại chạy được khi chưa cài

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self._app = FaceAnalysis(
            name=model_name,
            root=model_root,
            allowed_modules=["detection", "recognition", "landmark_3d_68"],
            providers=providers,
        )
        self._app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size), det_thresh=det_thresh)
        self.flip_tta = flip_tta
        self.recognizers: list[Recognizer] = [InsightFaceRecognizer(model_name, self._app.models["recognition"])]

        for kind, path in extra_models or []:
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy mô hình bổ sung: {path}")
            name = path.stem
            if kind == "adaface":
                self.recognizers.append(AdaFaceRecognizer(name, str(path), providers))
            else:
                from insightface.model_zoo import get_model

                m = get_model(str(path), providers=providers)
                m.prepare(ctx_id=0 if use_gpu else -1)
                self.recognizers.append(InsightFaceRecognizer(name, m))
            log.info("Đã nạp mô hình nhận diện bổ sung: %s (%s)", name, kind)
        self.model_names = [r.name for r in self.recognizers]

    def analyze(self, img_bgr: np.ndarray) -> list[FaceInfo]:
        faces = self._app.get(img_bgr)
        out: list[FaceInfo] = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            pose = getattr(f, "pose", None)
            out.append(
                FaceInfo(
                    bbox=np.asarray(f.bbox, dtype=np.float32),
                    kps=np.asarray(f.kps, dtype=np.float32),
                    det_score=float(f.det_score),
                    embedding=np.asarray(emb, dtype=np.float32),
                    pose_deg=tuple(float(x) for x in pose) if pose is not None else None,
                )
            )
        out.sort(key=lambda x: x.area, reverse=True)
        return out

    def embed(self, img_bgr: np.ndarray, face: FaceInfo, models: list[str] | None = None) -> dict[str, np.ndarray]:
        return embed_with(self.recognizers, img_bgr, face.kps, self.flip_tta, models)


# ---------------------------------------------------------------------------
# Gộp điểm
# ---------------------------------------------------------------------------
def parse_weights(spec: str, models: list[str]) -> dict[str, float]:
    """'buffalo_l:0.4,adaface_ir101_webface12m:0.6' -> dict. Mô hình không ghi trọng số nhận 1.0."""
    w = {m: 1.0 for m in models}
    for item in (spec or "").split(","):
        if ":" in item:
            k, v = item.rsplit(":", 1)
            k = k.strip()
            if k not in w:
                raise ValueError(f"FUSION_WEIGHTS nhắc tới mô hình '{k}' không có. Các mô hình đang nạp: {models}")
            w[k] = float(v)
    w = {m: v for m, v in w.items() if v > 0}  # trọng số 0 = không dùng mô hình đó khi tính điểm
    if not w:
        raise ValueError("FUSION_WEIGHTS phải có ít nhất một mô hình với trọng số > 0")
    return w


def fused_score(probe: dict[str, np.ndarray], template: dict[str, np.ndarray],
                weights: dict[str, float]) -> tuple[float, dict[str, float]] | None:
    """Trung bình có trọng số cosine của từng mô hình. Trả None nếu template thiếu mô hình nào."""
    per: dict[str, float] = {}
    for m in weights:
        if m not in probe or m not in template:
            return None
        per[m] = cosine(probe[m], template[m])
    total = sum(weights.values())
    return sum(weights[m] * per[m] for m in weights) / total, per


# ---------------------------------------------------------------------------
# Giải mã ảnh
# ---------------------------------------------------------------------------
class ImageDecodeError(ValueError):
    pass


def decode_image_bytes(data: bytes, max_side: int = 1600) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("Không đọc được ảnh (định dạng không hỗ trợ hoặc file hỏng).")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def decode_data_url(data_url: str, max_side: int = 1280) -> np.ndarray:
    """Nhận chuỗi 'data:image/jpeg;base64,...' hoặc base64 thuần."""
    if "," in data_url and data_url.strip().startswith("data:"):
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url, validate=False)
    except Exception as e:  # noqa: BLE001
        raise ImageDecodeError("Dữ liệu ảnh base64 không hợp lệ.") from e
    return decode_image_bytes(raw, max_side=max_side)


def pad_image(img: np.ndarray, ratio: float = 0.2) -> tuple[np.ndarray, int, int]:
    """Thêm viền quanh ảnh. Ảnh 4x6 thường có mặt chiếm gần hết khung, viền giúp detector ổn định hơn."""
    h, w = img.shape[:2]
    py, px = int(h * ratio), int(w * ratio)
    padded = cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return padded, px, py


def detect_for_enroll(engine: FaceEngine, img: np.ndarray) -> tuple[list[FaceInfo], np.ndarray]:
    """Phát hiện mặt trên ảnh 4x6; nếu không thấy thì thử lại với ảnh có viền.
    Trả về (danh sách mặt, ảnh tương ứng với tọa độ của các mặt đó)."""
    faces = engine.analyze(img)
    if faces:
        return faces, img
    padded, _, _ = pad_image(img)
    return engine.analyze(padded), padded


# ---------------------------------------------------------------------------
# Hình học khuôn mặt từ 5 điểm mốc
# ---------------------------------------------------------------------------
@dataclass
class FaceGeometry:
    yaw_ratio: float    # (mũi - tâm hai mắt) / khoảng cách hai mắt, theo trục ngang ảnh. >0: mũi lệch sang phải ẢNH
    pitch_ratio: float  # (mũi - mắt) / (miệng - mắt) theo trục dọc. Ngẩng lên -> giảm
    roll_deg: float

    def yaw_deg_estimate(self) -> float:
        # Mô hình xấp xỉ: độ nhô của mũi ~0.6 lần khoảng cách hai mắt => yaw_ratio ≈ 0.6·tan(yaw)
        return math.degrees(math.atan(self.yaw_ratio / 0.6))


def face_geometry(kps: np.ndarray) -> FaceGeometry:
    kps = np.asarray(kps, dtype=np.float64)
    eyes = kps[[0, 1]][np.argsort(kps[[0, 1], 0])]
    mouth = kps[[3, 4]][np.argsort(kps[[3, 4], 0])]
    le, re = eyes
    nose = kps[2]
    ml, mr = mouth

    roll = math.atan2(re[1] - le[1], re[0] - le[0])
    eye_mid = (le + re) / 2.0
    cos_r, sin_r = math.cos(-roll), math.sin(-roll)
    rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

    def r(p):
        return rot @ (p - eye_mid)

    le_r, re_r, nose_r = r(le), r(re), r(nose)
    mouth_mid_r = r((ml + mr) / 2.0)
    eye_dist = max(float(np.linalg.norm(re_r - le_r)), 1e-6)
    vertical = max(float(mouth_mid_r[1]), 1e-6)
    return FaceGeometry(
        yaw_ratio=float(nose_r[0] / eye_dist),
        pitch_ratio=float(nose_r[1] / vertical),
        roll_deg=math.degrees(roll),
    )


# ---------------------------------------------------------------------------
# Chất lượng ảnh & tiện ích
# ---------------------------------------------------------------------------
def crop_face(img: np.ndarray, bbox: np.ndarray, margin: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), w), min(int(y2), h)
    if x2 <= x1 or y2 <= y1:
        return img[0:1, 0:1].copy()
    return img[y1:y2, x1:x2].copy()


def sharpness_score(face_crop: np.ndarray) -> float:
    """Phương sai Laplacian trên ảnh mặt chuẩn hóa 112x112. Càng cao càng nét."""
    g = cv2.cvtColor(cv2.resize(face_crop, (112, 112)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def brightness_score(face_crop: np.ndarray) -> float:
    g = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(g.mean())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Không mã hóa được JPEG")
    return buf.tobytes()
