
"""Evaluation utilities for the RP1 lane and object detector."""

import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from calibration import ece, reliability_curve


def iou(boxes1, boxes2):
    """Pairwise IoU for boxes in (x1, y1, x2, y2) format."""
    boxes1 = np.asarray(boxes1, dtype=float).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=float).reshape(-1, 4)

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=float)

    left_top = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = np.clip(right_bottom - left_top, 0, None)
    intersection = wh[..., 0] * wh[..., 1]

    area1 = (
        np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None)
        * np.clip(boxes1[:, 3] - boxes1[:, 1], 0, None)
    )
    area2 = (
        np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None)
        * np.clip(boxes2[:, 3] - boxes2[:, 1], 0, None)
    )

    union = area1[:, None] + area2[None, :] - intersection
    return intersection / np.maximum(union, 1e-9)


def _average_precision(scores, matches, n_gt):
    """Compute interpolated AP using 101 recall thresholds."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0

    order = np.argsort(-np.asarray(scores))
    matches = np.asarray(matches, dtype=float)[order]

    tp = np.cumsum(matches)
    fp = np.cumsum(1.0 - matches)

    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)

    return float(
        np.mean([
            precision[recall >= r].max()
            if np.any(recall >= r) else 0.0
            for r in np.linspace(0.0, 1.0, 101)
        ])
    )


def compute_map(predictions, gt_boxes, gt_labels, iou_thr=0.5):
    """Class-aware mAP at a specified IoU threshold."""
    num_classes = 0
    for pred, labels in zip(predictions, gt_labels):
        if len(pred):
            num_classes = max(num_classes, int(pred[:, 4].max()) + 1)
        if len(labels):
            num_classes = max(num_classes, int(np.max(labels)) + 1)

    if num_classes == 0:
        return 0.0

    aps = []

    for cls in range(num_classes):
        scores, matches = [], []
        n_gt = 0

        # Count ground-truth objects of this class.
        for labels in gt_labels:
            n_gt += int(np.sum(np.asarray(labels) == cls))

        if n_gt == 0:
            continue

        # Match predictions to each image's ground truth.
        for pred, boxes, labels in zip(predictions, gt_boxes, gt_labels):
            pred = np.asarray(pred)
            boxes = np.asarray(boxes).reshape(-1, 4)
            labels = np.asarray(labels).reshape(-1)

            pred_cls = pred[pred[:, 4].astype(int) == cls] if len(pred) else pred
            used_gt = set()

            for det in sorted(pred_cls, key=lambda d: d[5], reverse=True):
                scores.append(float(det[5]))

                candidates = np.where(labels == cls)[0]
                if len(candidates) == 0:
                    matches.append(0.0)
                    continue

                overlaps = iou(det[None, :4], boxes[candidates])[0]
                ranked = np.argsort(-overlaps)

                matched = False
                for rank in ranked:
                    gt_idx = int(candidates[rank])
                    if overlaps[rank] >= iou_thr and gt_idx not in used_gt:
                        used_gt.add(gt_idx)
                        matched = True
                        break

                matches.append(1.0 if matched else 0.0)

        aps.append(_average_precision(scores, matches, n_gt))

    valid_aps = [ap for ap in aps if np.isfinite(ap)]
    return float(np.mean(valid_aps)) if valid_aps else 0.0


def gating_metrics(uep_max, error_flags):
    """Return ROC-AUC, false-positive rate, true-positive rate and thresholds."""
    uncertainty = np.asarray(uep_max, dtype=float)
    errors = np.asarray(error_flags, dtype=int)

    if len(uncertainty) != len(errors) or len(errors) == 0:
        raise ValueError("Uncertainty and error arrays must have equal nonzero length.")

    if len(np.unique(errors)) < 2:
        auc = float("nan")
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        thresholds = np.array([np.inf, -np.inf])
    else:
        auc = float(roc_auc_score(errors, uncertainty))
        fpr, tpr, thresholds = roc_curve(errors, uncertainty)

    return auc, fpr, tpr, thresholds


def evaluate_all(model, loader, calibrators, cfg_obj, cfg_cal, device,
                 map_score_thr=0.0001, map_iou_thr=0.6, map_max_dets=300):
    """Evaluate object AP/calibration and lane conformal coverage.

    Padded labels are trimmed with n_obj. Samples without labels for a task are
    excluded from that task's metrics rather than treated as background/negative.
    Object uncertainty is not available from this detector, so uncertainty-gating
    AUC is deliberately reported as NaN rather than using reserved zero columns.
    """
    model.eval()

    all_probs, all_correct = [], []
    cov_scores, cov_labels = [], []
    predictions, gt_boxes, gt_labels = [], [], []

    with torch.no_grad():
        for x, batch in loader:
            x = x.to(device)
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            out = model(x)
            dets = model.decode(out)  # deployment operating point for ECE
            dets_map = model.decode(out, map_score_thr, map_iou_thr, map_max_dets)

            for i, det_tensor in enumerate(dets):
                det = det_tensor.detach().cpu().numpy()
                det_map = dets_map[i].detach().cpu().numpy()

                # Object metrics are meaningful only where object annotations exist.
                object_labeled = bool(batch["has_object_labels"][i].item()) if "has_object_labels" in batch else True
                if object_labeled:
                    n_obj = int(batch["n_obj"][i].item()) if "n_obj" in batch else int(batch["boxes"][i].shape[0])
                    boxes = batch["boxes"][i, :n_obj].detach().cpu().numpy().reshape(-1, 4)
                    labels = batch["labels"][i, :n_obj].detach().cpu().numpy().reshape(-1)
                    predictions.append(det_map)
                    gt_boxes.append(boxes)
                    gt_labels.append(labels)

                    # Confidence reliability: greedy one-to-one match, requiring
                    # both class agreement and IoU >= 0.5.
                    used_gt = set()
                    for row in sorted(det, key=lambda r: float(r[5]), reverse=True):
                        cls = int(row[4])
                        candidates = np.where(labels == cls)[0]
                        matched = False
                        if len(candidates):
                            overlaps = iou(row[None, :4], boxes[candidates])[0]
                            for rank in np.argsort(-overlaps):
                                gt_idx = int(candidates[rank])
                                if overlaps[rank] >= 0.5 and gt_idx not in used_gt:
                                    used_gt.add(gt_idx)
                                    matched = True
                                    break
                        all_probs.append(float(row[5]))
                        all_correct.append(int(matched))

                # Lane calibration/coverage is computed only on lane-labeled samples.
                lane_labeled = bool(batch["has_lane_labels"][i].item()) if "has_lane_labels" in batch else True
                if lane_labeled:
                    pe = out["p_exist"][i].detach().cpu().numpy().reshape(-1)
                    lt = batch["lane_targets"][i].detach().cpu().numpy().reshape(-1)
                    cov_scores.append(pe)
                    cov_labels.append((lt > 0).astype(int))

    if all_probs:
        ece_val = ece(np.asarray(all_probs), np.asarray(all_correct, dtype=int), cfg_cal.n_bins_ece)
        confs, accs = reliability_curve(np.asarray(all_probs), np.asarray(all_correct, dtype=int), cfg_cal.n_bins_ece)
    else:
        ece_val = float("nan")
        confs, accs = np.array([]), np.array([])

    if cov_scores and "lane" in calibrators and calibrators["lane"].qhat is not None:
        coverage = calibrators["lane"].coverage(np.concatenate(cov_scores), np.concatenate(cov_labels))
    else:
        coverage = float("nan")

    map50 = compute_map(predictions, gt_boxes, gt_labels, iou_thr=0.5) if predictions else float("nan")

    return {
        "ece": ece_val,
        "coverage": coverage,
        "gating_auc": float("nan"),  # object uncertainty is not implemented by this model
        "mAP@50": map50,
        "conf_bins": confs,
        "acc_bins": accs,
        "gating_fpr": np.array([]),
        "gating_tpr": np.array([]),
        "gating_thresholds": np.array([]),
    }


def benchmark_latency(model, x, iters=100, device="cpu"):
    """Measure average forward-pass latency in milliseconds per batch."""
    if iters < 1:
        raise ValueError("iters must be at least 1.")

    model.eval()
    model.to(device)
    x = x.to(device)

    with torch.no_grad():
        for _ in range(5):
            model(x)

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(iters):
            model(x)

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return elapsed * 1000.0 / iters
