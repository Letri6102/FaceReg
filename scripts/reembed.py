"""Tính bổ sung vector cho mô hình mới cho toàn bộ thí sinh đã đăng ký.

Hệ thống vẫn tự bổ sung khi thí sinh check-in, nhưng chạy trước giúp lượt check-in đầu không bị chậm.
Dùng sau khi thay đổi EXTRA_MODELS:
    python scripts/reembed.py

Nếu đổi FLIP_TTA hoặc thay file mô hình cùng tên, dùng --all để tính lại toàn bộ.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bootstrap import build_service  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Tính lại mọi mô hình cho bộ mẫu từ ảnh 4x6")
    args = ap.parse_args()
    svc = build_service()
    if args.all:
        n = 0
        for cid in svc.db.all_candidate_ids():
            for st in svc.db.get_template_sets(cid):
                if st["source"] == "enroll" and svc._backfill_enroll_set(cid, st["set_id"], svc.models):
                    n += 1
        print(f"Đã tính lại {n} bộ mẫu. (Mẫu webcam cũ được giữ nguyên; nếu đổi FLIP_TTA nên xóa bằng cách đăng ký lại.)")
    else:
        print(svc.backfill_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
