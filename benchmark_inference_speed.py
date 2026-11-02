
import argparse
import os
import time
import statistics

import numpy as np
import torch

from config import ModelConfig, DataConfig
from datasets import make_dataset
from models.detector import JointDetector


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(checkpoint, device):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    model = JointDetector(ModelConfig()).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    if isinstance(state, dict):
        state = state.get(
            "model_state_dict",
            state.get("state_dict", state.get("model", state))
        )

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def benchmark(fn, warmup, iterations, device):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync(device)

        times = []
        for _ in range(iterations):
            sync(device)
            start = time.perf_counter()
            fn()
            sync(device)
            times.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(times)
    return {
        "mean": mean_ms,
        "median": statistics.median(times),
        "p95": float(np.percentile(times, 95)),
        "min": min(times),
        "max": max(times),
        "fps": 1000 / mean_ms,
    }


def show_results(title, result):
    print(f"\n{title}")
    print(f"  Mean latency:   {result['mean']:.2f} ms")
    print(f"  Median latency: {result['median']:.2f} ms")
    print(f"  P95 latency:    {result['p95']:.2f} ms")
    print(f"  Min / max:      {result['min']:.2f} / "
          f"{result['max']:.2f} ms")
    print(f"  Approx. FPS:    {result['fps']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=r"experiments\latest_run\latest_model.pt"
    )
    parser.add_argument(
        "--data_dir",
        default=r"D:\MinorProjectData\BDD100K"
    )
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--score_thr", type=float, default=0.25)
    parser.add_argument("--nms_iou", type=float, default=0.6)
    parser.add_argument("--max_dets", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    cfg = ModelConfig()
    dcfg = DataConfig()
    dcfg.dataset_name = "bdd100k"
    dcfg.bdd_dir = args.data_dir

    # Get a real, normalized BDD100K image.
    dataset = make_dataset(
        "bdd100k", cfg, dcfg, split="validation"
    )
    image, _ = dataset[0]
    image = image.unsqueeze(0).contiguous()

    print("Input shape:", tuple(image.shape))
    print("Checkpoint:", args.ckpt)

    model = load_model(args.ckpt, device)
    image_gpu = image.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # FP32 model forward only.
    def forward_fp32():
        return model(image_gpu.float())

    result = benchmark(
        forward_fp32, args.warmup, args.iterations, device
    )
    show_results("FP32: model forward only", result)

    # FP32 forward plus detection decoding and NMS.
    # This excludes image loading and preprocessing.
    def forward_and_decode():
        outputs = model(image_gpu.float())
        return model.decode(
            outputs, args.score_thr, args.nms_iou, args.max_dets
        )

    result_full = benchmark(
        forward_and_decode, args.warmup, args.iterations, device
    )
    show_results("FP32: forward + decode/NMS", result_full)

    # Mixed precision on CUDA, if supported.
    if device.type == "cuda":
        def forward_fp16():
            with torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                return model(image_gpu)

        try:
            result_fp16 = benchmark(
                forward_fp16, args.warmup, args.iterations, device
            )
            show_results("FP16 autocast: model forward only", result_fp16)
        except Exception as exc:
            print("\nFP16 benchmark failed:",
                  type(exc).__name__, str(exc))

        allocated = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
        reserved = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )
        print(f"\nPeak GPU allocated memory: {allocated:.1f} MiB")
        print(f"Peak GPU reserved memory:  {reserved:.1f} MiB")

    print("\nReference: 30 FPS = 33.33 ms/frame.")
    print("Timing excludes camera capture, image loading, and display.")
    print("Validate accuracy before adopting FP16.")


if __name__ == "__main__":
    main()
