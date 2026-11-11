"""Đăng ký hàng loạt ảnh 4x6.

Cách 1 - chỉ thư mục ảnh, tên file là mã thí sinh:
    python scripts/bulk_enroll.py --dir D:\\anh_4x6
    (2024001.jpg -> thí sinh 2024001)

Cách 2 - kèm file CSV (UTF-8) có cột: code,full_name[,photo]
    python scripts/bulk_enroll.py --dir D:\\anh_4x6 --csv danh_sach.csv
    Cột photo (nếu có) là tên file ảnh trong --dir; nếu trống sẽ tìm <code>.jpg/.jpeg/.png.

Kết quả ghi vào enroll_report.csv (mở bằng Excel được) để biết ảnh nào không đạt và vì sao.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bootstrap import build_service  # noqa: E402
from app.services import ServiceError  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_photo(folder: Path, code: str) -> Path | None:
    for ext in IMG_EXT:
        for cand in (folder / f"{code}{ext}", folder / f"{code}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Đăng ký hàng loạt ảnh thí sinh")
    ap.add_argument("--dir", required=True, help="Thư mục chứa ảnh")
    ap.add_argument("--csv", help="File CSV: code,full_name[,photo]")
    ap.add_argument("--replace", action="store_true", help="Ghi đè thí sinh đã tồn tại")
    ap.add_argument("--report", default="enroll_report.csv", help="File báo cáo kết quả")
    args = ap.parse_args()

    folder = Path(args.dir)
    jobs: list[tuple[str, str, Path | None]] = []
    if args.csv:
        with open(args.csv, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                code = (row.get("code") or "").strip()
                if not code:
                    continue
                photo = (row.get("photo") or "").strip()
                path = folder / photo if photo else find_photo(folder, code)
                jobs.append((code, (row.get("full_name") or "").strip(), path))
    else:
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                jobs.append((p.stem, "", p))

    if not jobs:
        print("Không có ảnh nào để đăng ký.")
        return 1

    print(f"Đang nạp mô hình... ({len(jobs)} thí sinh)")
    service = build_service()
    ok_count = 0
    with open(args.report, "w", encoding="utf-8-sig", newline="") as out:
        w = csv.writer(out)
        w.writerow(["code", "full_name", "photo", "ket_qua", "ly_do", "face_px", "yaw_deg", "pitch_deg", "sharpness", "brightness"])
        for i, (code, name, path) in enumerate(jobs, 1):
            status, reasons, m = "LOI", "", {}
            if path is None or not path.exists():
                reasons = "Không tìm thấy file ảnh"
            else:
                try:
                    r = service.enroll(code, name, path.read_bytes(), replace=args.replace)
                    m = r["quality"].get("metrics", {})
                    if r["ok"]:
                        status = "OK"
                        ok_count += 1
                    else:
                        status = "KHONG_DAT"
                        reasons = " | ".join(r["quality"]["reasons"])
                except ServiceError as e:
                    reasons = e.message
            w.writerow([code, name, path.name if path else "", status, reasons, m.get("face_px", ""), m.get("yaw_deg", ""),
                        m.get("pitch_deg", ""), m.get("sharpness", ""), m.get("brightness", "")])
            print(f"[{i}/{len(jobs)}] {code}: {status} {reasons}")

    print(f"\nHoàn tất: {ok_count}/{len(jobs)} đạt. Báo cáo: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
