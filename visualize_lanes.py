"""
Visualize OpenLane lane predictions from joint_quick_test.pt.
Run from the project folder:
    python visualize_lanes.py
Saves overlays to ./lane_visualizations/
Green = ground-truth lane anchors; red = model predictions.
"""
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from config import ModelConfig, DataConfig
from datasets import make_dataset, collate_fn
from models.detector import JointDetector

CHECKPOINT = r"runs/weighted_ufld_openlane_w4\best_lane_checkpoint.pt"
OUTPUT_DIR = "lane_visualizations"
NUM_IMAGES = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    image = (tensor * std + mean).clamp(0, 1)
    image = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(image)

def draw_lane_points(draw, targets, cfg, color, radius=3):
    # targets: (num_lanes, grid_h), where 0 means no lane and 1..grid_w are columns
    for lane_idx in range(targets.shape[0]):
        points = []
        for row in range(targets.shape[1]):
            cls = int(targets[lane_idx, row])
            if cls <= 0:
                continue
            x = int(round((cls - 1 + 0.5) * cfg.image_w / cfg.grid_w))
            y = int(round((row + 0.5) * cfg.image_h / cfg.grid_h))
            points.append((x, y))
        # Connect nearby valid row anchors; gaps are left unconnected.
        for a, b in zip(points, points[1:]):
            if abs(b[1] - a[1]) <= max(1, int(2 * cfg.image_h / cfg.grid_h)):
                draw.line([a, b], fill=color, width=3)
        for x, y in points:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

def main():
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {os.path.abspath(CHECKPOINT)}\n"
            "Run this script from the folder containing joint_quick_test.pt."
        )

    mc = ModelConfig()
    dc = DataConfig()
    dataset = make_dataset("openlane", mc, dc, split="validation")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = JointDetector(mc, teacher=False).to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)

    if isinstance(state, dict):
       if "model" in state:
          state = state["model"]
       elif "state_dict" in state:
          state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = 0
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if saved >= NUM_IMAGES:
                break
            images, targets = batch
            images = images.to(DEVICE)
            out = model(images)
            logits = out["lane_logits"][0]  # (num_lanes, grid_h, grid_w+1)
            pred = logits.argmax(dim=-1).cpu()  # 0=background; 1..grid_w=columns
            gt = targets["lane_targets"][0].cpu()

            image = denormalize(images[0])
            draw = ImageDraw.Draw(image)
            # Ground truth first, then predictions on top.
            draw_lane_points(draw, gt, mc, color=(40, 220, 70), radius=2)
            draw_lane_points(draw, pred, mc, color=(255, 55, 55), radius=3)

            # Add a small legend.
            draw.rectangle((8, 8, 255, 42), fill=(0, 0, 0))
            draw.text((14, 12), "Green: ground truth | Red: prediction", fill=(255, 255, 255))
            path = os.path.join(OUTPUT_DIR, f"lane_{saved:02d}_datasetidx_{idx}.jpg")
            image.save(path, quality=95)
            saved += 1
            print(f"Saved {path}")

    print(f"\nDone. Saved {saved} images to: {os.path.abspath(OUTPUT_DIR)}")
    print("Inspect whether red predicted lanes align with green ground-truth anchors.")

if __name__ == "__main__":
    main()