"""Xuất mô hình AdaFace (checkpoint .ckpt gốc) sang ONNX để FaceReg dùng.

Làm một lần, trên máy có Python + PyTorch:
    git clone https://github.com/mk-minchul/AdaFace
    cd AdaFace
    pip install torch
    # Tải checkpoint IR-101 WebFace12M (link Google Drive trong README của repo AdaFace),
    # đặt vào thư mục pretrained/, ví dụ pretrained/adaface_ir101_webface12m.ckpt
    python <đường_dẫn>/FaceReg/scripts/export_adaface_onnx.py \
        --arch ir_101 --ckpt pretrained/adaface_ir101_webface12m.ckpt \
        --out <đường_dẫn>/FaceReg/models/adaface/adaface_ir101_webface12m.onnx

Sau đó trong .env của FaceReg:
    EXTRA_MODELS=adaface:models/adaface/adaface_ir101_webface12m.onnx

Script tự kiểm tra: so sánh đầu ra PyTorch và ONNX Runtime trên cùng đầu vào (nếu có onnxruntime).
Lưu ý giấy phép: mã AdaFace theo MIT, nhưng trọng số huấn luyện trên WebFace (chỉ cho nghiên cứu).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ir_101", help="ir_18 | ir_50 | ir_101")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch

    import net  # module của repo AdaFace

    model = net.build_model(args.arch)
    state = torch.load(args.ckpt, map_location="cpu")
    state = state.get("state_dict", state)
    model_state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(model_state)
    model.eval()

    class Wrapper(torch.nn.Module):
        """AdaFace trả về (feature đã chuẩn hóa, norm). Chỉ xuất feature."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            feat, _norm = self.m(x)
            return feat

    wrapped = Wrapper(model).eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dummy = torch.randn(2, 3, 112, 112)
    torch.onnx.export(
        wrapped, dummy, args.out,
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=13,
    )
    print("Đã xuất:", args.out)

    try:
        import onnxruntime as ort

        with torch.no_grad():
            ref = wrapped(dummy).numpy()
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": dummy.numpy()})[0]
        diff = float(np.abs(ref - out).max())
        print(f"Kiểm tra PyTorch vs ONNX: sai khác lớn nhất = {diff:.2e}", "(OK)" if diff < 1e-3 else "(BẤT THƯỜNG!)")
    except ImportError:
        print("Chưa cài onnxruntime nên bỏ qua bước kiểm tra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
