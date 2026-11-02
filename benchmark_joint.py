"""
Benchmark joint object + lane inference speed.

Run from your project root in the MinorProject conda environment:
    python benchmark_joint.py --image "D:\\path\\to\\road_image.jpg"

Optional:
    --warmup 30
    --iterations 200
    --batch-size 1
    --lane-threshold 0.5
    --obj-threshold 0.25
    --nms-iou 0.45
    --max-dets 50
    --lane-ckpt "runs\\openlane_full\\latest_resume.pt"

Reports:
- end-to-end model inference FPS (preprocess + forward + decode + lane extraction)
- model forward-only FPS
- mean / median / p95 latency
- GPU peak memory (CUDA)
- output CSV with per-iteration timings

This benchmarks one image repeatedly. FPS is inference throughput, not camera/video pipeline FPS.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import ModelConfig
from models.detector import JointDetector


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark joint object and lane inference.")
    p.add_argument("--image", required=True, help="Path to a representative road image.")
    p.add_argument(
        "--lane-ckpt",
        default=r"runs\openlane_full\latest_resume.pt",
        help="Full joint checkpoint from OpenLane lane training (default: latest_resume.pt).",
    )
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--obj-threshold", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--lane-threshold", type=float, default=0.5)
    p.add_argument("--max-dets", type=int, default=50)
    p.add_argument("--output-dir", default=r"runs\speed_benchmark")
    return p.parse_args()


def extract_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_model(ckpt_path, device):
    mc = ModelConfig()
    model = JointDetector(mc).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = extract_state(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Be strict about both heads being present, to avoid benchmarking a partial model.
    missing_text = " ".join(missing)
    if any(token in missing_text for token in ("lane_head", "det_head", "obj_head")):
        raise RuntimeError(
            "Checkpoint appears to be missing one or more prediction heads. "
            f"Missing keys: {missing}"
        )

    if unexpected:
        print(f"[Warning] Unexpected checkpoint keys ({len(unexpected)}); first few: {unexpected[:8]}")

    model.eval()
    return model


def preprocess_image(path, device):
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    resized = cv2.resize(image_bgr, (640, 384), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).float().div_(255.0)
    tensor = tensor.permute(2, 0, 1).contiguous()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return image_bgr, tensor.unsqueeze(0).to(device)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.batch_size < 1:
        raise ValueError("iterations and batch-size must be >=1; warmup must be >=0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        torch.backends.cudnn.benchmark = True

    image_path = Path(args.image)
    _, one_tensor = preprocess_image(image_path, device)
    batch = one_tensor.repeat(args.batch_size, 1, 1, 1)

    model = load_model(args.lane_ckpt, device)
    print(f"Checkpoint: {args.lane_ckpt}")
    print(f"Input tensor: {tuple(batch.shape)}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Measured iterations: {args.iterations}")

    # Warm-up to initialize CUDA kernels / cuDNN algorithms.
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(batch)
    synchronize(device)

    # Forward-only timing.
    forward_ms = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(args.iterations):
            synchronize(device)
            t0 = time.perf_counter()
            _ = model(batch)
            synchronize(device)
            forward_ms.append((time.perf_counter() - t0) * 1000.0)

    # End-to-end model pipeline timing: preprocess already prepared in memory;
    # includes forward + decoder + lane probability/argmax processing.
    end_to_end_ms = []
    rows = []
    with torch.inference_mode():
        for i in range(args.iterations):
            synchronize(device)
            t0 = time.perf_counter()

            output = model(batch)
            detections = model.decode(
                output,
                score_thr=args.obj_threshold,
                iou_thr=args.nms_iou,
                max_dets=args.max_dets,
            )

            lane_logits = output["lane_logits"]
            lane_probs = torch.softmax(lane_logits, dim=-1)
            lane_conf, lane_bins = lane_probs[..., 1:].max(dim=-1)
            lane_bins = lane_bins + 1
            lane_bins = torch.where(
                lane_conf >= args.lane_threshold,
                lane_bins,
                torch.zeros_like(lane_bins),
            )

            synchronize(device)
            elapsed = (time.perf_counter() - t0) * 1000.0
            end_to_end_ms.append(elapsed)
            rows.append({
                "iteration": i + 1,
                "forward_ms": forward_ms[i],
                "pipeline_ms": elapsed,
                "batch_size": args.batch_size,
            })

    def summarize(label, values, batch_size=1):
        arr = np.asarray(values, dtype=np.float64)
        mean_ms = float(arr.mean())
        median_ms = float(np.median(arr))
        p95_ms = float(np.percentile(arr, 95))
        fps = (1000.0 / mean_ms) * batch_size
        print(f"\n{label}")
        print(f"  Mean latency:   {mean_ms:.3f} ms")
        print(f"  Median latency: {median_ms:.3f} ms")
        print(f"  P95 latency:    {p95_ms:.3f} ms")
        print(f"  Throughput:     {fps:.2f} images/sec (FPS at batch size {batch_size})")

    summarize("Forward-only (both heads)", forward_ms, args.batch_size)
    summarize("Forward + decode + lane postprocess", end_to_end_ms, args.batch_size)

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"\nCUDA peak allocated memory: {peak_mb:.1f} MB")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "timings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["iteration", "forward_ms", "pipeline_ms", "batch_size"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer-iteration timing CSV saved to: {csv_path.resolve()}")
    print("\nNote: this is single-image/repeated-image inference speed. "
          "It excludes disk loading and image visualization/writing.")


if __name__ == "__main__":
    main()
