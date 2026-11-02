"""
Joint object + lane inference for the Minor Project.

Run from project root:
  python predict_joint.py --image "path\\to\\image.jpg"

Checkpoint defaults:
  object weights: experiments\\class_aware_conservative\\bdd_epoch10_backup.pt
  lane weights:   runs\\openlane_full\\latest_resume.pt

Outputs annotated image to runs\\joint_predictions\\<image_name>_joint.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from config import ModelConfig
from models.detector import JointDetector


CLASS_NAMES = [
    "car", "truck", "bus", "pedestrian",
    "cyclist", "motorcycle", "traffic_light", "traffic_sign"
]


def parse_args():
    p = argparse.ArgumentParser(description="Joint object and lane prediction")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--obj-ckpt",
                   default=r"experiments\class_aware_conservative\bdd_epoch10_backup.pt",
                   help="Object detection checkpoint")
    p.add_argument("--lane-ckpt",
                   default=r"runs\openlane_full\latest_resume.pt",
                   help="OpenLane-trained checkpoint containing lane-head weights")
    p.add_argument("--out-dir", default="runs/joint_predictions")
    p.add_argument("--obj-threshold", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--lane-threshold", type=float, default=0.50)
    p.add_argument("--max-dets", type=int, default=50)
    return p.parse_args()


def extract_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    # Training script checkpoint format observed in this project:
    # {'model', 'optimizer', 'scaler', 'epoch', 'val_loss', 'best_val_loss', 'args'}
    for key in ("model", "model_state", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    # Raw state_dict fallback
    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint
    raise ValueError("Cannot find model state in checkpoint.")


def load_weights(model, checkpoint_path, device, require_lane=False):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = extract_state(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)

    lane_missing = [k for k in missing if k.startswith("lane_head.")]
    obj_missing = [k for k in missing if k.startswith("obj_head.")]
    if require_lane and lane_missing:
        raise RuntimeError(
            f"Lane checkpoint did not load lane_head weights ({len(lane_missing)} missing). "
            "Check the checkpoint path/format."
        )
    if not require_lane and obj_missing:
        print(f"[WARNING] Object checkpoint has {len(obj_missing)} missing obj_head keys.")
    print(f"Loaded {checkpoint_path}")
    print(f"  missing tensors: {len(missing)} | unexpected tensors: {len(unexpected)}")
    if lane_missing:
        print(f"  lane_head missing: {len(lane_missing)}")
    if obj_missing:
        print(f"  obj_head missing: {len(obj_missing)}")
    return set(missing)


def preprocess(image_bgr, cfg):
    # BGR -> RGB, resize to model's 640x384 input, ImageNet normalization
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (cfg.image_w, cfg.image_h), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, (image_bgr.shape[1], image_bgr.shape[0])


def draw_lanes(canvas, lane_logits, cfg, threshold):
    """
    lane_logits shape (1, num_lanes, grid_h, n_offsets+1).
    Grid class 0 is background; classes 1..n_offsets encode x bins.
    Coordinates are mapped to the resized model-input canvas.
    """
    probs = torch.softmax(lane_logits.float(), dim=-1)[0]  # L,H,O+1
    fg_prob, fg_idx = probs[..., 1:].max(dim=-1)
    argmax = probs.argmax(dim=-1)
    active = (argmax != 0) & (fg_prob >= threshold)
    bins = (fg_idx + 1).cpu().numpy()
    active = active.cpu().numpy()

    H, W = canvas.shape[:2]
    colors = [
        (0, 255, 0),       # green
        (0, 220, 255),     # yellow/orange
        (255, 80, 0),      # blue-ish in BGR
        (255, 0, 255),     # magenta
    ]
    for lane in range(active.shape[0]):
        points = []
        for row in range(active.shape[1]):
            if not active[lane, row]:
                continue
            xbin = int(bins[lane, row])
            # 80 bins represent normalized horizontal locations across input width.
            x = int((xbin - 1) / max(cfg.n_offsets - 1, 1) * (W - 1))
            y = int(row / max(active.shape[1] - 1, 1) * (H - 1))
            points.append((x, y))
        if len(points) >= 2:
            pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, colors[lane % len(colors)], 3, cv2.LINE_AA)
            for x, y in points[::max(1, len(points)//8)]:
                cv2.circle(canvas, (x, y), 3, colors[lane % len(colors)], -1, cv2.LINE_AA)


def draw_detections(canvas, detections, original_size, input_size):
    """
    Detection rows from this project's YOLOv5StyleDetectionHead.decode:
    [x1,y1,x2,y2,class_id,score,0,0] in 640x384 input pixels.
    """
    orig_w, orig_h = original_size
    in_w, in_h = input_size
    sx, sy = orig_w / in_w, orig_h / in_h
    colors = [
        (50, 220, 50), (0, 180, 255), (255, 120, 0), (0, 255, 255),
        (255, 0, 255), (180, 80,255), (255,255,0), (80,200,255)
    ]
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det[:4]]
        cls_id = int(det[4])
        score = float(det[5])
        x1 = int(np.clip(x1 * sx, 0, orig_w - 1))
        x2 = int(np.clip(x2 * sx, 0, orig_w - 1))
        y1 = int(np.clip(y1 * sy, 0, orig_h - 1))
        y2 = int(np.clip(y2 * sy, 0, orig_h - 1))
        color = colors[cls_id % len(colors)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        label = f"{name} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.75, orig_w / 1200.0))
        (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(canvas, (x1, top), (min(orig_w-1, x1+tw+6), y1), color, -1)
        cv2.putText(canvas, label, (x1+3, max(th, y1-4)), font, scale, (0,0,0), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    model = JointDetector(cfg).to(device)

    # Load the full checkpoint produced by lane training. During lane training,
    # obj_head was frozen, so its object weights remain from the initialization
    # checkpoint, while the shared backbone and lane_head were adapted.
    # Loading the full checkpoint keeps backbone/head tensors consistent.
    load_weights(model, args.lane_ckpt, device, require_lane=True)
    model.eval()

    tensor, original_size = preprocess(image, cfg)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
        detections = model.decode(
            output,
            score_thr=args.obj_threshold,
            iou_thr=args.nms_iou,
            max_dets=args.max_dets
        )[0]
        lane_logits = output["lane_logits"]

    # Draw on original-resolution image. Lane points are first mapped from model
    # input coordinates to original image coordinates.
    orig_w, orig_h = original_size
    lane_canvas = cv2.resize(image, (cfg.image_w, cfg.image_h), interpolation=cv2.INTER_LINEAR)
    draw_lanes(lane_canvas, lane_logits, cfg, args.lane_threshold)
    lane_canvas = cv2.resize(lane_canvas, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    result = image.copy()
    # detections coordinates already scaled to original in draw_detections
    draw_detections(result, detections, original_size, (cfg.image_w, cfg.image_h))

    # Composite lane overlay onto result. Lane canvas contains resized original
    # image with lane lines, so extract lane overlay via color mask from lines.
    overlay = lane_canvas
    # Blend the lane image softly, while retaining crisp object boxes/text.
    result = cv2.addWeighted(result, 0.65, overlay, 0.35, 0.0)
    # Re-draw object boxes on top of the blended lane image for readability.
    draw_detections(result, detections, original_size, (cfg.image_w, cfg.image_h))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_joint.jpg"
    cv2.imwrite(str(out_path), result)

    print(f"Objects detected: {len(detections)}")
    print(f"Saved combined visualization: {out_path.resolve()}")


if __name__ == "__main__":
    main()
