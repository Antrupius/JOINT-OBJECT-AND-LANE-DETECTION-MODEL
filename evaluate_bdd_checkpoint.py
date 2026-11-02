
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import ModelConfig, DataConfig
from datasets import make_dataset, collate_fn, DRIVING_CLASSES
from models.detector import JointDetector
from evaluate import iou, _average_precision

CHECKPOINT = r"checkpoints\bdd100k_test_resume.pt"
BDD_DIR = r"D:\MinorProjectData\BDD100K"
NUM_IMAGES = 10000
SCORE_THRESHOLD = 0.20  # operating-point threshold for precision/recall/F1
MAP_SCORE_THRESHOLD = 0.0001  # low threshold for confidence-swept AP
NMS_IOU = 0.6
MAX_DETS = 300
MATCH_IOU = 0.5



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate object detection on a fixed BDD100K validation subset."
    )
    parser.add_argument("--ckpt", default=CHECKPOINT, help="Checkpoint path.")
    parser.add_argument("--data_dir", default=BDD_DIR, help="BDD100K dataset root.")
    parser.add_argument("--images", type=int, default=NUM_IMAGES)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    if args.images < 1 or args.batch_size < 1:
        parser.error("--images and --batch_size must be positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig()
    dcfg = DataConfig()
    dcfg.dataset_name = "bdd100k"
    dcfg.bdd_dir = args.data_dir

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    model = JointDetector(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)

    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state.get("model", state)))

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {args.ckpt}")
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = make_dataset("bdd100k", cfg, dcfg, split="validation")
    n_images = min(args.images, len(dataset))
    if n_images < 1:
        raise RuntimeError("BDD100K validation split is empty.")
    if n_images < args.images:
        print(f"Warning: requested {args.images} images; only {n_images} are available.")

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(
        len(dataset), generator=generator
    )[:n_images].tolist()

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"Checkpoint: {args.ckpt}")
    print(f"Validation images: {len(subset)} of {len(dataset)}")
    all_predictions, all_predictions_operating, all_gt_boxes, all_gt_labels = [], [], [], []

    with torch.inference_mode():
        for images, batch in loader:
            images = images.to(device)
            outputs = model(images)
            detections_map = model.decode(
                outputs, MAP_SCORE_THRESHOLD, NMS_IOU, MAX_DETS
            )
            detections_operating = model.decode(
                outputs, SCORE_THRESHOLD, NMS_IOU, MAX_DETS
            )

            for i, (det_map, det_operating) in enumerate(zip(detections_map, detections_operating)):
                all_predictions.append(det_map.cpu().numpy())
                all_predictions_operating.append(det_operating.cpu().numpy())
                n_obj = int(batch["n_obj"][i].item())
                all_gt_boxes.append(batch["boxes"][i, :n_obj].cpu().numpy())
                all_gt_labels.append(batch["labels"][i, :n_obj].cpu().numpy())

    num_classes = len(DRIVING_CLASSES)
    total_tp = total_fp = total_gt = 0
    class_aps = {}
    class_metrics = {}

    # Evaluate each class
    for cls in range(num_classes):
        scores, matches = [], []
        n_gt = sum(
            int(np.sum(labels == cls)) for labels in all_gt_labels
        )
        total_gt += n_gt
        class_tp = class_fp = 0

        for pred, boxes, labels in zip(
            all_predictions, all_gt_boxes, all_gt_labels
        ):
            gt_indices = np.where(labels == cls)[0]
            used_gt = set()

            pred_cls = (
                pred[pred[:, 4].astype(int) == cls] if len(pred) else []
            )

            for det in sorted(pred_cls, key=lambda row: row[5], reverse=True):
                scores.append(float(det[5]))
                match = None
                if len(gt_indices):
                    overlaps = iou(det[None, :4], boxes[gt_indices])[0]
                    for rank in np.argsort(-overlaps):
                        gt_idx = int(gt_indices[rank])
                        if overlaps[rank] >= MATCH_IOU and gt_idx not in used_gt:
                            match = gt_idx
                            break
                if match is not None:
                    used_gt.add(match)
                    matches.append(1.0)
                else:
                    matches.append(0.0)

        # Operating-point precision/recall/F1 are computed separately at SCORE_THRESHOLD.
        class_tp = class_fp = 0
        for pred_op, boxes, labels in zip(all_predictions_operating, all_gt_boxes, all_gt_labels):
            gt_indices = np.where(labels == cls)[0]
            used_gt = set()
            pred_cls_op = pred_op[pred_op[:, 4].astype(int) == cls] if len(pred_op) else []
            for det in sorted(pred_cls_op, key=lambda row: row[5], reverse=True):
                match = None
                if len(gt_indices):
                    overlaps = iou(det[None, :4], boxes[gt_indices])[0]
                    for rank in np.argsort(-overlaps):
                        gt_idx = int(gt_indices[rank])
                        if overlaps[rank] >= MATCH_IOU and gt_idx not in used_gt:
                            match = gt_idx
                            break
                if match is not None:
                    used_gt.add(match)
                    class_tp += 1
                else:
                    class_fp += 1

        if n_gt > 0:
            class_aps[cls] = _average_precision(scores, matches, n_gt)
            cls_precision = class_tp / max(class_tp + class_fp, 1)
            cls_recall = class_tp / n_gt
            cls_f1 = (
                2 * cls_precision * cls_recall
                / max(cls_precision + cls_recall, 1e-12)
            )

            class_metrics[cls] = {
                "precision": cls_precision,
                "recall": cls_recall,
                "f1": cls_f1,
                "tp": class_tp,
                "fp": class_fp,
                "fn": n_gt - class_tp,
                "gt": n_gt,
            }

        total_tp += class_tp
        total_fp += class_fp

    # Calculate overall metrics AFTER all classes are processed
    valid_aps = [
        ap for ap in class_aps.values() if np.isfinite(ap)
    ]
    map50 = float(np.mean(valid_aps)) if valid_aps else 0.0

    total_fn = total_gt - total_tp
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    macro_f1 = (
        float(np.mean([m["f1"] for m in class_metrics.values()]))
        if class_metrics else 0.0
    )

    # Report overall metrics
    print(f"\nImages evaluated: {len(subset)}")
    print(
        f"AP score threshold: {MAP_SCORE_THRESHOLD}; operating score threshold: {SCORE_THRESHOLD}; "
        f"NMS IoU: {NMS_IOU}; match IoU: {MATCH_IOU}"
    )

    print(f"\nmAP@0.5:           {map50:.4f}")
    print(f"Overall precision: {precision:.4f}")
    print(f"Overall recall:    {recall:.4f}")
    print(f"Overall F1:        {f1:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")

    print("\nPer-class metrics:")
    print(
        f"{'Class':<16} {'AP@0.5':>8} {'Prec.':>8} "
        f"{'Recall':>8} {'F1':>8} {'TP':>7} "
        f"{'FP':>7} {'FN':>7}"
    )

    for cls in range(num_classes):
        if cls not in class_metrics:
            print(
                f"{DRIVING_CLASSES[cls]:<16} "
                f"{'N/A':>8} {'N/A':>8} {'N/A':>8} "
                f"{'N/A':>8} {'N/A':>7} {'N/A':>7} {'N/A':>7}"
            )
            continue

        m = class_metrics[cls]
        ap = class_aps.get(cls, float("nan"))

        print(
            f"{DRIVING_CLASSES[cls]:<16} "
            f"{ap:>8.4f} "
            f"{m['precision']:>8.4f} "
            f"{m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{m['tp']:>7} "
            f"{m['fp']:>7} "
            f"{m['fn']:>7}"
        )

    print(f"\nGround-truth objects: {total_gt}")
    print(f"True positives:       {total_tp}")
    print(f"False positives:      {total_fp}")
    print(f"False negatives:      {total_fn}")


if __name__ == "__main__":
    main()
