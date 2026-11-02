
import argparse
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.utils.data import Subset
from config import ModelConfig, DataConfig
from datasets import make_dataset, collate_fn, DRIVING_CLASSES, MEAN, STD
from models.detector import JointDetector

CHECKPOINT = os.path.join("checkpoints", "bdd100k_7000_1000_final_audit_v1.pt")
BDD_DIR = r"D:\MinorProjectData\BDD100K"

NUM_IMAGES = 6
SCORE_THRESHOLD = 0.25
NMS_IOU = 0.6
MAX_DETS = 100


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BDD100K ground truth and model detections."
    )
    parser.add_argument("--ckpt", default=CHECKPOINT, help="Checkpoint path.")
    parser.add_argument("--data_dir", default=BDD_DIR, help="BDD100K dataset root.")
    parser.add_argument("--images", type=int, default=NUM_IMAGES)
    parser.add_argument("--score_thr", type=float, default=SCORE_THRESHOLD)
    args = parser.parse_args()
    if args.images < 1:
        parser.error("--images must be positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ModelConfig()
    dcfg = DataConfig()
    dcfg.dataset_name = "bdd100k"
    dcfg.bdd_dir = args.data_dir

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    model = JointDetector(cfg).to(device)
    state = torch.load(
        args.ckpt, map_location=device, weights_only=True
    )
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state.get("model", state)))

    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = make_dataset(
        "bdd100k", cfg, dcfg, split="validation"
    )

    # Same deterministic validation subset as the evaluator
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(
        len(dataset), generator=generator
    )[:1000].tolist()

    # Choose six images from that subset
    random.seed(7)
    chosen = random.sample(indices, min(args.images, len(indices)))

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Score threshold: {args.score_thr}")
    print("Green = ground truth | Red = predictions")

    for image_idx in chosen:
        image, target = dataset[image_idx]

        with torch.inference_mode():
            outputs = model(image.unsqueeze(0).to(device))
            detections = model.decode(
                outputs, args.score_thr, NMS_IOU, MAX_DETS
            )[0].cpu().numpy()

        # Convert image tensor (C,H,W) to display format (H,W,C)
        img = image.detach().cpu().permute(1, 2, 0).numpy()

        # Reverse the dataset's normalization
        mean = MEAN.cpu().numpy().reshape(1, 1, 3)
        std = STD.cpu().numpy().reshape(1, 1, 3)

        img = img * std + mean
        img = np.clip(img, 0, 1)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(img)

        # Ground-truth boxes
        gt_boxes = target["boxes"].numpy()
        gt_labels = target["labels"].numpy()

        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box[:4]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="lime",
                facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, max(0, y1 - 3),
                f"GT: {DRIVING_CLASSES[int(label)]}",
                color="black", fontsize=8,
                bbox=dict(facecolor="lime", alpha=0.8, pad=1)
            )

        # Predicted boxes: expected columns
        # [x1, y1, x2, y2, class_id, confidence]
        for det in detections:
            x1, y1, x2, y2, cls_id, score = det[:6]
            cls_id = int(cls_id)

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, min(img.shape[0] - 3, y2 + 2),
                f"P: {DRIVING_CLASSES[cls_id]} {score:.2f}",
                color="white", fontsize=8,
                bbox=dict(facecolor="red", alpha=0.8, pad=1)
            )

        ax.set_title(
            f"Validation index {image_idx} | "
            f"GT: {len(gt_boxes)} | Predictions: {len(detections)}"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
