"""Khởi tạo service dùng chung cho web server và các script dòng lệnh."""
from __future__ import annotations

from .config import Settings, load_settings
from .db import Database
from .services import FaceRegService


def build_engine(settings: Settings):
    from .config import BASE_DIR
    from .face_engine import InsightFaceEngine, parse_extra_models

    extras = [(kind, p if p.is_absolute() else BASE_DIR / p) for kind, p in parse_extra_models(settings.extra_models)]
    return InsightFaceEngine(
        model_name=settings.model_name,
        model_root=str(settings.model_root),
        det_size=settings.det_size,
        det_thresh=settings.det_thresh,
        use_gpu=settings.use_gpu,
        extra_models=extras,
        flip_tta=settings.flip_tta,
    )


def build_service(settings: Settings | None = None) -> FaceRegService:
    settings = settings or load_settings()
    return FaceRegService(settings, Database(settings.db_path), build_engine(settings))
