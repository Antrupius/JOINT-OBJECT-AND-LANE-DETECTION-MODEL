
"""RP1 deployment: ONNX export and INT8 quantization."""

import torch


def export_onnx(model, path: str, cfg):
    model.eval()
    dummy = torch.randn(1, 3, cfg.image_h, cfg.image_w)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["image"],
        output_names=[
            "obj_logit",
            "cls_logits",
            "lane_logits",
            "p_exist",
        ],
        dynamic_axes={
            "image": {0: "batch"},
            "lane_logits": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"exported ONNX -> {path}")



def quantize_int8(onnx_path: str, out_path: str):
    try:
        from onnxruntime.quantization import (
            quantize_dynamic,
            QuantType,
        )

        quantize_dynamic(
            onnx_path,
            out_path,
            weight_type=QuantType.QInt8,
        )

        print(f"INT8 quantized -> {out_path}")

    except ImportError as e:
        print("ONNX Runtime quantization unavailable:", e)


def quantize_pt(model, cfg):
    """PyTorch dynamic quantization fallback for CPU."""
    m = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8,
    )
    return m
