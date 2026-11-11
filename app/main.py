"""Ứng dụng web FastAPI. Chạy: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .bootstrap import build_service
from .config import BASE_DIR, load_settings
from .services import FaceRegService, ServiceError, load_image_file, thumbnail

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("facereg")

STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_FRAME_CHARS = 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class StartCheckinIn(BaseModel):
    candidate_code: str = Field(..., max_length=64)
    exam_code: str = Field("", max_length=64)


class FrameIn(BaseModel):
    image: str = Field(..., max_length=MAX_FRAME_CHARS, description="data URL JPEG")


class ReviewIn(BaseModel):
    decision: str = Field(..., pattern="^(APPROVE|REJECT|approve|reject)$")
    note: str = Field("", max_length=500)


class MonitorIn(BaseModel):
    token: str = Field(..., max_length=128)
    image: str = Field(..., max_length=MAX_FRAME_CHARS)


# ---------------------------------------------------------------------------
def create_app(service: FaceRegService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "service", None) is None:
            app.state.service = build_service(load_settings())
        svc: FaceRegService = app.state.service
        if svc.s.admin_token in ("", "change-me"):
            log.warning("ADMIN_TOKEN đang dùng giá trị mặc định. Hãy đổi trong file .env trước khi triển khai!")
        svc.db.purge_old_sessions()
        yield

    app = FastAPI(title="FaceReg - Xác thực thí sinh", version="1.0.0", lifespan=lifespan)
    app.state.service = service

    settings_for_cors = service.s if service else load_settings()
    if settings_for_cors.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings_for_cors.cors_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "X-Admin-Token"],
        )

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message, "code": exc.code})

    def svc(request: Request) -> FaceRegService:
        return request.app.state.service

    def require_admin(request: Request, x_admin_token: str = Header("", alias="X-Admin-Token")) -> None:
        expected = request.app.state.service.s.admin_token
        if expected in ("", "change-me"):
            raise ServiceError("Chưa cấu hình ADMIN_TOKEN trong file .env (đang để mặc định nên API quản trị bị khóa).",
                               503, "admin_not_configured")
        if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
            raise ServiceError("Sai hoặc thiếu mã quản trị (X-Admin-Token).", 401, "unauthorized")

    def client_ip(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()[:64]
        return request.client.host if request.client else ""

    def jpeg(data: bytes) -> Response:
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ---------------- Trang web ----------------
    @app.get("/", include_in_schema=False)
    def page_index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/checkin", include_in_schema=False)
    def page_checkin():
        return FileResponse(STATIC_DIR / "checkin.html")

    @app.get("/admin", include_in_schema=False)
    def page_admin():
        return FileResponse(STATIC_DIR / "admin.html")

    @app.get("/api/health")
    def health(service: FaceRegService = Depends(svc)):
        return {
            "ok": True,
            "models": service.models,
            "fusion_weights": service.weights,
            "flip_tta": service.s.flip_tta,
            "passive_liveness": service.antispoof.enabled,
            "threshold_accept": service.s.threshold_accept,
            "threshold_reject": service.s.threshold_reject,
        }

    # ---------------- Check-in (thí sinh) ----------------
    @app.post("/api/checkin/start")
    def checkin_start(body: StartCheckinIn, request: Request, service: FaceRegService = Depends(svc)):
        return service.start_checkin(body.candidate_code, body.exam_code, client_ip(request))

    @app.post("/api/checkin/{session_id}/frame")
    def checkin_frame(session_id: str, body: FrameIn, service: FaceRegService = Depends(svc)):
        return service.process_frame(session_id, body.image)

    @app.post("/api/checkin/{session_id}/finish")
    def checkin_finish(session_id: str, service: FaceRegService = Depends(svc)):
        return service.finish_checkin(session_id)

    @app.get("/api/checkin/{session_id}/status")
    def checkin_status(session_id: str, service: FaceRegService = Depends(svc)):
        return service.checkin_status(session_id)

    # ---------------- Giám sát trong giờ thi ----------------
    @app.post("/api/monitor")
    def monitor(body: MonitorIn, service: FaceRegService = Depends(svc)):
        return service.monitor(body.token, body.image)

    # ---------------- Quản trị: thí sinh ----------------
    @app.post("/api/admin/candidates", dependencies=[Depends(require_admin)])
    async def admin_enroll(
        request: Request,
        code: str = Form(...),
        full_name: str = Form(""),
        replace: bool = Form(False),
        photo: UploadFile = File(...),
    ):
        data = await photo.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ServiceError("Ảnh quá lớn (tối đa 10MB).", 413, "too_large")
        service: FaceRegService = request.app.state.service
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(service.enroll, code, full_name, data, replace)

    @app.get("/api/admin/candidates", dependencies=[Depends(require_admin)])
    def admin_list_candidates(q: str = "", limit: int = 200, offset: int = 0, service: FaceRegService = Depends(svc)):
        return service.db.list_candidates(q.strip(), min(max(limit, 1), 1000), max(offset, 0))

    @app.delete("/api/admin/candidates/{code}", dependencies=[Depends(require_admin)])
    def admin_delete_candidate(code: str, service: FaceRegService = Depends(svc)):
        return service.delete_candidate(code)

    @app.get("/api/admin/candidates/{code}/face", dependencies=[Depends(require_admin)])
    def admin_candidate_face(code: str, service: FaceRegService = Depends(svc)):
        cand = service.db.get_candidate(code)
        if not cand:
            raise ServiceError("Không tìm thấy thí sinh.", 404, "not_found")
        return jpeg(thumbnail(load_image_file(cand["face_path"])))

    # ---------------- Quản trị: check-in ----------------
    @app.get("/api/admin/checkins", dependencies=[Depends(require_admin)])
    def admin_list_checkins(status: str = "", exam_code: str = "", candidate_code: str = "", limit: int = 100,
                            service: FaceRegService = Depends(svc)):
        rows = service.db.list_checkins(status.upper(), exam_code, candidate_code, min(max(limit, 1), 1000))
        for r in rows:
            r.pop("monitor_token", None)
            r["has_capture"] = bool(r.pop("captured_path", ""))
        return rows

    @app.get("/api/admin/checkins/{checkin_id}", dependencies=[Depends(require_admin)])
    def admin_get_checkin(checkin_id: int, service: FaceRegService = Depends(svc)):
        c = service.admin_checkin(checkin_id)
        c["monitor_events"] = service.db.list_monitor_events(checkin_id)
        return c

    @app.get("/api/admin/checkins/{checkin_id}/capture", dependencies=[Depends(require_admin)])
    def admin_checkin_capture(checkin_id: int, service: FaceRegService = Depends(svc)):
        c = service.db.get_checkin(checkin_id)
        if not c or not c.get("captured_path"):
            raise ServiceError("Không có ảnh chụp.", 404, "not_found")
        return jpeg(load_image_file(c["captured_path"]))

    @app.post("/api/admin/checkins/{checkin_id}/review", dependencies=[Depends(require_admin)])
    def admin_review(checkin_id: int, body: ReviewIn, service: FaceRegService = Depends(svc)):
        return service.review(checkin_id, body.decision, body.note)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()
