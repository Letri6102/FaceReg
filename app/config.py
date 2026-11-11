"""Cấu hình hệ thống, đọc từ biến môi trường hoặc file .env ở thư mục gốc."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Đọc file .env đơn giản (KEY=VALUE), không ghi đè biến môi trường đã có."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw.strip())


@dataclass
class Settings:
    # --- Chung ---
    data_dir: Path
    db_path: Path
    admin_token: str
    cors_origins: list[str]
    checkin_success_url: str

    # --- Mô hình nhận diện ---
    model_name: str
    model_root: Path
    det_size: int
    det_thresh: float
    use_gpu: bool
    extra_models: str            # "adaface:models/adaface/adaface_ir101_webface12m.onnx,insightface:..."
    fusion_weights: str          # "buffalo_l:0.5,adaface_ir101_webface12m:0.5" (trống = bằng nhau)
    flip_tta: bool

    # --- Ngưỡng so khớp (cosine similarity) ---
    threshold_accept: float
    threshold_reject: float

    # --- Chất lượng ảnh đăng ký (ảnh 4x6) ---
    enroll_min_face_px: int
    enroll_max_yaw: float
    enroll_max_pitch: float
    enroll_min_sharpness: float
    enroll_min_brightness: float
    enroll_max_brightness: float
    enroll_min_det_score: float

    # --- Check-in / liveness chủ động ---
    checkin_min_face_px: int
    challenge_turns: int
    challenge_hold_frames: int
    yaw_turn_ratio: float
    yaw_frontal_ratio: float
    pitch_up_delta: float
    pitch_center_tolerance: float
    camera_mirrored: bool
    session_ttl_seconds: int
    session_max_frames: int
    identity_consistency_min: float
    multi_face_area_ratio: float

    # --- Liveness thụ động (ONNX anti-spoof, tùy chọn) ---
    antispoof_models: list[Path]
    antispoof_threshold: float
    antispoof_input_scale: float
    antispoof_real_index: int

    # --- Đăng ký dần dần ---
    progressive_enroll: bool
    auto_template_min_score: float
    max_live_templates: int

    # --- Giám sát trong giờ thi ---
    monitor_min_interval_seconds: float

    @property
    def candidates_dir(self) -> Path:
        return self.data_dir / "candidates"

    @property
    def checkins_dir(self) -> Path:
        return self.data_dir / "checkins"


def load_settings() -> Settings:
    _load_dotenv(BASE_DIR / ".env")

    data_dir = Path(_env("DATA_DIR", str(BASE_DIR / "data")))
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir
    model_root = Path(_env("MODEL_ROOT", str(BASE_DIR / "models")))
    if not model_root.is_absolute():
        model_root = BASE_DIR / model_root

    antispoof_raw = _env("ANTISPOOF_MODELS", "")
    antispoof_models: list[Path] = []
    for item in antispoof_raw.split(","):
        item = item.strip()
        if item:
            p = Path(item)
            antispoof_models.append(p if p.is_absolute() else BASE_DIR / p)

    s = Settings(
        data_dir=data_dir,
        db_path=data_dir / "facereg.db",
        admin_token=_env("ADMIN_TOKEN", "change-me"),
        cors_origins=[o.strip() for o in _env("CORS_ORIGINS", "").split(",") if o.strip()],
        checkin_success_url=_env("CHECKIN_SUCCESS_URL", ""),
        model_name=_env("MODEL_NAME", "buffalo_l"),
        model_root=model_root,
        det_size=_env("DET_SIZE", 640, int),
        det_thresh=_env("DET_THRESH", 0.5, float),
        use_gpu=_env("USE_GPU", False, bool),
        extra_models=_env("EXTRA_MODELS", ""),
        fusion_weights=_env("FUSION_WEIGHTS", ""),
        flip_tta=_env("FLIP_TTA", True, bool),
        threshold_accept=_env("THRESHOLD_ACCEPT", 0.45, float),
        threshold_reject=_env("THRESHOLD_REJECT", 0.28, float),
        enroll_min_face_px=_env("ENROLL_MIN_FACE_PX", 112, int),
        enroll_max_yaw=_env("ENROLL_MAX_YAW", 15.0, float),
        enroll_max_pitch=_env("ENROLL_MAX_PITCH", 15.0, float),
        enroll_min_sharpness=_env("ENROLL_MIN_SHARPNESS", 40.0, float),
        enroll_min_brightness=_env("ENROLL_MIN_BRIGHTNESS", 50.0, float),
        enroll_max_brightness=_env("ENROLL_MAX_BRIGHTNESS", 215.0, float),
        enroll_min_det_score=_env("ENROLL_MIN_DET_SCORE", 0.6, float),
        checkin_min_face_px=_env("CHECKIN_MIN_FACE_PX", 90, int),
        challenge_turns=_env("CHALLENGE_TURNS", 3, int),
        challenge_hold_frames=_env("CHALLENGE_HOLD_FRAMES", 2, int),
        yaw_turn_ratio=_env("YAW_TURN_RATIO", 0.25, float),
        yaw_frontal_ratio=_env("YAW_FRONTAL_RATIO", 0.12, float),
        pitch_up_delta=_env("PITCH_UP_DELTA", 0.10, float),
        pitch_center_tolerance=_env("PITCH_CENTER_TOLERANCE", 0.08, float),
        camera_mirrored=_env("CAMERA_MIRRORED", False, bool),
        session_ttl_seconds=_env("SESSION_TTL_SECONDS", 120, int),
        session_max_frames=_env("SESSION_MAX_FRAMES", 300, int),
        identity_consistency_min=_env("IDENTITY_CONSISTENCY_MIN", 0.35, float),
        multi_face_area_ratio=_env("MULTI_FACE_AREA_RATIO", 0.25, float),
        antispoof_models=antispoof_models,
        antispoof_threshold=_env("ANTISPOOF_THRESHOLD", 0.8, float),
        antispoof_input_scale=_env("ANTISPOOF_INPUT_SCALE", 1.0, float),
        antispoof_real_index=_env("ANTISPOOF_REAL_INDEX", 1, int),
        progressive_enroll=_env("PROGRESSIVE_ENROLL", True, bool),
        auto_template_min_score=_env("AUTO_TEMPLATE_MIN_SCORE", 0.55, float),
        max_live_templates=_env("MAX_LIVE_TEMPLATES", 5, int),
        monitor_min_interval_seconds=_env("MONITOR_MIN_INTERVAL_SECONDS", 3.0, float),
    )
    if s.threshold_reject > s.threshold_accept:
        raise ValueError("THRESHOLD_REJECT phải nhỏ hơn hoặc bằng THRESHOLD_ACCEPT")
    s.candidates_dir.mkdir(parents=True, exist_ok=True)
    s.checkins_dir.mkdir(parents=True, exist_ok=True)
    s.model_root.mkdir(parents=True, exist_ok=True)
    return s
