"""Chuyển mô hình MiniFASNet (.pth) của dự án Silent-Face-Anti-Spoofing sang ONNX để FaceReg dùng
cho liveness thụ động.

Các bước (làm một lần, trên máy có Python + PyTorch):
    git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
    cd Silent-Face-Anti-Spoofing
    pip install torch
    python <đường_dẫn>/FaceReg/scripts/export_antispoof_onnx.py --out <đường_dẫn>/FaceReg/models/anti_spoof

Sau đó trong .env của FaceReg:
    ANTISPOOF_MODELS=models/anti_spoof/2.7_80x80_MiniFASNetV2.onnx,models/anti_spoof/4_0_0_80x80_MiniFASNetV1SE.onnx

Hãy kiểm thử bằng mặt thật / ảnh in / ảnh trên điện thoại để chọn ANTISPOOF_THRESHOLD phù hợp.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.getcwd())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="resources/anti_spoof_models")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV1SE, MiniFASNetV2, MiniFASNetV2SE
    from src.utility import get_kernel, parse_model_name

    mapping = {
        "MiniFASNetV1": MiniFASNetV1,
        "MiniFASNetV2": MiniFASNetV2,
        "MiniFASNetV1SE": MiniFASNetV1SE,
        "MiniFASNetV2SE": MiniFASNetV2SE,
    }
    os.makedirs(args.out, exist_ok=True)
    for name in sorted(os.listdir(args.models_dir)):
        if not name.endswith(".pth"):
            continue
        h, w, model_type, _scale = parse_model_name(name)
        model = mapping[model_type](conv6_kernel=get_kernel(h, w))
        state = torch.load(os.path.join(args.models_dir, name), map_location="cpu")
        if next(iter(state)).startswith("module."):
            state = OrderedDict((k[7:], v) for k, v in state.items())
        model.load_state_dict(state)
        model.eval()
        out_path = os.path.join(args.out, name.replace(".pth", ".onnx"))
        torch.onnx.export(model, torch.randn(1, 3, h, w), out_path,
                          input_names=["input"], output_names=["logits"], opset_version=11)
        print("Đã xuất:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
