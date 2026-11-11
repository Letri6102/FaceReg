"""So sánh mô hình và hiệu chỉnh ngưỡng trên dữ liệu THẬT của bạn (ảnh 4x6 + ảnh webcam).

Chuẩn bị thư mục, mỗi người một thư mục con:
    calib/
      2024001/
        ref.jpg          <- ảnh 4x6 (tên bắt đầu bằng "ref")
        cam1.jpg         <- ảnh chụp webcam, nhìn thẳng (càng nhiều càng tốt, nhiều điều kiện ánh sáng)
        cam2.jpg
      2024002/
        ...

Nên có ít nhất 100 người x 3 ảnh webcam (=> ~30.000 cặp khác người, đủ để đo ở FMR 0,1%).

Chạy:
    python scripts/calibrate_threshold.py --dir calib --target-far 0.001

Script dùng đúng cấu hình trong .env (MODEL_NAME, EXTRA_MODELS, FLIP_TTA) và:
  1. So sánh TỪNG mô hình và điểm GỘP theo FUSION_WEIGHTS: EER, tỷ lệ từ chối nhầm (FNMR) ở các mức
     nhận nhầm (FMR) 1%, 0,1%, 0,01%. Mô hình/cách gộp nào có FNMR thấp hơn ở cùng FMR là tốt hơn.
  2. Đề xuất THRESHOLD_ACCEPT / THRESHOLD_REJECT cho cấu hình gộp đang dùng.
  3. Thử nhanh vài bộ trọng số gộp khác (nếu có >= 2 mô hình) và in bộ tốt nhất.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bootstrap import build_engine  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.face_engine import decode_image_bytes, detect_for_enroll, parse_weights  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FMRS = (1e-2, 1e-3, 1e-4)


def embed_file(engine, path: Path, enroll: bool) -> dict[str, np.ndarray] | None:
    img = decode_image_bytes(path.read_bytes())
    if enroll:
        faces, det_img = detect_for_enroll(engine, img)
    else:
        faces, det_img = engine.analyze(img), img
    if not faces:
        return None
    return engine.embed(det_img, max(faces, key=lambda f: f.area))


def threshold_at_fmr(imp: np.ndarray, fmr: float) -> float:
    """Ngưỡng nhỏ nhất sao cho tỷ lệ cặp khác người có điểm >= ngưỡng không vượt fmr."""
    desc = np.sort(imp)[::-1]
    k = int(np.floor(fmr * len(desc)))
    return float(np.nextafter(desc[k], np.inf)) if k < len(desc) else float(desc[-1])


def eer(gen: np.ndarray, imp: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate và ngưỡng tương ứng (tính vector hóa, nhanh với hàng chục nghìn cặp)."""
    ts = np.unique(np.concatenate([gen, imp]))
    g, i = np.sort(gen), np.sort(imp)
    far = 1.0 - np.searchsorted(i, ts, side="left") / len(i)   # tỷ lệ cặp khác người có điểm >= t
    frr = np.searchsorted(g, ts, side="left") / len(g)          # tỷ lệ cặp cùng người có điểm < t
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(ts[k])


def report_row(name: str, gen: np.ndarray, imp: np.ndarray) -> list[str]:
    e, _ = eer(gen, imp)
    cells = [name, f"{e:.3%}"]
    for fmr in FMRS:
        if len(imp) < 1 / fmr:
            cells.append("thiếu dữ liệu")
        else:
            t = threshold_at_fmr(imp, fmr)
            cells.append(f"{np.mean(gen < t):.2%} (t={t:.3f})")
    return cells


def fuse(scores: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
    tot = sum(w.values())
    return sum(w[m] * scores[m] for m in w) / tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--target-far", type=float, default=0.001, help="Tỷ lệ nhận nhầm tối đa cho vùng ACCEPT")
    ap.add_argument("--reject-frr", type=float, default=0.01,
                    help="Tỷ lệ người thật bị đẩy vào vùng REJECT tối đa (phần còn lại vào REVIEW)")
    ap.add_argument("--save-scores", help="Ghi toàn bộ điểm ra file CSV để phân tích thêm")
    args = ap.parse_args()

    settings = load_settings()
    engine = build_engine(settings)
    models = engine.model_names
    weights = parse_weights(settings.fusion_weights, models)
    print(f"Mô hình: {models} | trọng số gộp hiện tại: {weights} | flip TTA: {settings.flip_tta}\n")

    refs: dict[str, dict] = {}
    probes: dict[str, list[dict]] = {}
    for person in sorted(p for p in Path(args.dir).iterdir() if p.is_dir()):
        files = [f for f in sorted(person.iterdir()) if f.suffix.lower() in IMG_EXT]
        ref_files = [f for f in files if f.stem.lower().startswith("ref")]
        if not ref_files:
            print(f"[bỏ qua] {person.name}: không có ảnh ref")
            continue
        r = embed_file(engine, ref_files[0], enroll=True)
        if r is None:
            print(f"[bỏ qua] {person.name}: không thấy mặt trong ảnh ref")
            continue
        refs[person.name] = r
        probes[person.name] = []
        for f in files:
            if f in ref_files:
                continue
            e = embed_file(engine, f, enroll=False)
            if e is None:
                print(f"  [cảnh báo] {person.name}/{f.name}: không thấy mặt")
            else:
                probes[person.name].append(e)
        print(f"  {person.name}: {len(probes[person.name])} ảnh webcam")

    names = [n for n in refs if probes.get(n)]
    if len(names) < 2:
        print("Cần ít nhất 2 người có ảnh webcam.")
        return 1

    gen = {m: [] for m in models}
    imp = {m: [] for m in models}
    pairs: list[tuple[str, str, int]] = []
    for a in names:
        for b in names:
            for i, e in enumerate(probes[b]):
                pairs.append((a, b, i))
                for m in models:
                    (gen if a == b else imp)[m].append(float(np.dot(refs[a][m], e[m])))
    gen = {m: np.array(v) for m, v in gen.items()}
    imp = {m: np.array(v) for m, v in imp.items()}

    if args.save_scores:
        with open(args.save_scores, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["ref", "probe_person", "probe_idx", "same", *models])
            gi = {m: iter(gen[m]) for m in models}
            ii = {m: iter(imp[m]) for m in models}
            for a, b, i in pairs:
                src = gi if a == b else ii
                w.writerow([a, b, i, int(a == b), *[f"{next(src[m]):.5f}" for m in models]])

    n_gen, n_imp = len(next(iter(gen.values()))), len(next(iter(imp.values())))
    print(f"\nSố người: {len(names)} | cặp cùng người: {n_gen} | cặp khác người: {n_imp}")

    header = ["Cấu hình", "EER", *[f"FNMR @ FMR={f:g}" for f in FMRS]]
    rows = [report_row(m, gen[m], imp[m]) for m in models]
    fused_gen, fused_imp = fuse(gen, weights), fuse(imp, weights)
    if len(models) > 1:
        rows.append(report_row("GỘP (FUSION_WEIGHTS hiện tại)", fused_gen, fused_imp))
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    print("\n" + " | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print("(FNMR càng thấp càng tốt. t = ngưỡng tương ứng.)")

    if len(models) > 1 and n_imp >= 1 / args.target_far:
        best = None
        for a in np.linspace(0, 1, 11):
            if len(models) == 2:
                w = {models[0]: 1 - a, models[1]: a}
            else:
                w = {models[0]: 1 - a, **{m: a / (len(models) - 1) for m in models[1:]}}
            w = {k: v for k, v in w.items() if v > 0}
            t = threshold_at_fmr(fuse(imp, w), args.target_far)
            fnmr = float(np.mean(fuse(gen, w) < t))
            if best is None or fnmr < best[0]:
                best = (fnmr, w)
        spec = ",".join(f"{k}:{v:.2f}" for k, v in best[1].items())
        print(f"\nTrọng số tốt nhất trên dữ liệu này (FMR={args.target_far:g}): FUSION_WEIGHTS={spec}"
              f"  -> FNMR={best[0]:.2%}")
        print("  Lưu ý: chọn trọng số trên chính tập này dễ lạc quan; nên kiểm lại trên một tập người khác.")

    g, i = (fused_gen, fused_imp) if len(models) > 1 else (gen[models[0]], imp[models[0]])
    accept = threshold_at_fmr(i, args.target_far)
    reject = float(np.quantile(g, args.reject_frr))
    reject = min(reject, accept)
    print("\nĐề xuất cho cấu hình hiện tại (ghi vào .env):")
    print(f"THRESHOLD_ACCEPT={accept:.3f}   # FMR={np.mean(i >= accept):.4%}, FNMR={np.mean(g < accept):.2%}")
    print(f"THRESHOLD_REJECT={reject:.3f}   # {np.mean(g < reject):.2%} người thật rơi vào REJECT")
    print(f"=> {np.mean((g >= reject) & (g < accept)):.2%} người thật sẽ cần giám thị duyệt (REVIEW).")
    if n_imp < 1.0 / args.target_far:
        print(f"\n[!] Chỉ có {n_imp} cặp khác người: chưa đủ để ước lượng FMR={args.target_far:g} tin cậy "
              f"(cần ≥ {int(1 / args.target_far)}). Hãy bổ sung dữ liệu.")
    print("\nNhớ: mỗi khi đổi EXTRA_MODELS / FUSION_WEIGHTS / FLIP_TTA phải chạy lại script này và cập nhật ngưỡng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
