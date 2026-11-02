
import os
import sys
import json
import numpy as np
from collections import defaultdict

import torch

from models.yolov5_detection import DEFAULT_ANCHORS, build_targets


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"D:\MinorProjectData\BDD100K"
SPLIT = "train"
MAX_FILES = 3000
IMAGE_W = 640
IMAGE_H = 384

# Actual YOLOv5-style detection head strides
GRID_SHAPES = [
    (IMAGE_H // 8, IMAGE_W // 8),    # P3: 48 x 80
    (IMAGE_H // 16, IMAGE_W // 16),   # P4: 24 x 40
    (IMAGE_H // 32, IMAGE_W // 32),   # P5: 12 x 20
]

ANCHOR_T = 4.0

CLASS_NAMES = [
    "car", "truck", "bus", "pedestrian",
    "cyclist", "motorcycle", "traffic_light", "traffic_sign"
]

# ============================================================
# FIND BDD ANNOTATION FILES
# ============================================================


def find_annotation_file(data_dir, split):
    ann_dir = os.path.join(data_dir, split, "ann")

    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    json_files = [
        os.path.join(ann_dir, f)
        for f in os.listdir(ann_dir)
        if f.lower().endswith(".json")
    ]

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {ann_dir}")

    json_files.sort()
    json_files = json_files[:MAX_FILES]

    print(f"Annotation directory: {ann_dir}")
    print(f"Individual JSON files found: {len(json_files)}")

    return json_files


# ============================================================
# LOAD ANNOTATIONS
# ============================================================


def load_bdd_annotations(json_paths):
    records = []

    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        records.append(record)

    print(f"Successfully loaded annotation records: {len(records)}")

    return records

# ============================================================
# EXTRACT VALID OBJECT BOXES
# ============================================================



def extract_objects(records):
    boxes = []
    labels = []

    class_map = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "person": 3,
    "rider": 4,
    "motor": 5,
    "bike": 4,
    "traffic light": 6,
    "traffic sign": 7,
    }

    invalid = 0
    ignored = 0

    # Flatten records in case the loader returned nested lists
    flattened = []

    for item in records:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)

    for record in flattened:
        if not isinstance(record, dict):
            continue

        img_w = record.get("size", {}).get("width", 1280)
        img_h = record.get("size", {}).get("height", 720)

        for obj in record.get("objects", []):
            class_name = obj.get("classTitle", "").strip().lower()

            if class_name not in class_map:
                ignored += 1
                continue

            exterior = obj.get("points", {}).get("exterior", [])

            if len(exterior) < 2:
                invalid += 1
                continue

            x1, y1 = exterior[0]
            x2, y2 = exterior[1]

            x1 *= 640.0 / img_w
            x2 *= 640.0 / img_w
            y1 *= 384.0 / img_h
            y2 *= 384.0 / img_h

            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            if x2 <= x1 or y2 <= y1:
                invalid += 1
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(class_map[class_name])

    print(f"Valid boxes extracted: {len(boxes)}")
    print(f"Invalid boxes skipped: {invalid}")
    print(f"Other categories ignored: {ignored}")

    return boxes, labels

# ============================================================
# ANCHOR MATCHING STATISTICS
# ============================================================

def calculate_anchor_statistics(boxes, labels):

    anchors = DEFAULT_ANCHORS.cpu().float()

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2

    gt_wh = torch.stack([widths, heights], dim=1)

    # --------------------------------------------------------
    # Direct anchor matching (same ratio criterion as model)
    # --------------------------------------------------------

    matched_per_scale = torch.zeros(
        len(GRID_SHAPES), len(boxes), dtype=torch.long
    )

    matched_any = torch.zeros(len(boxes), dtype=torch.bool)

    for si, (gh, gw) in enumerate(GRID_SHAPES):

        scale = torch.tensor(
            [IMAGE_W / gw, IMAGE_H / gh],
            dtype=torch.float32
        )

        gwh = gt_wh / scale
        a_grid = anchors[si] / scale

        ratios = gwh[:, None, :] / a_grid[None, :, :]

        matches = torch.maximum(
            ratios, 1.0 / ratios
        ).amax(dim=2) < ANCHOR_T

        matched_per_scale[si] = matches.sum(dim=1)
        matched_any |= matches.any(dim=1)

    # --------------------------------------------------------
    # Actual build_targets assignment (including neighbor cells)
    # --------------------------------------------------------

    # Simulate one image containing all extracted GT objects
    batch_boxes = boxes.unsqueeze(0)
    batch_labels = labels.unsqueeze(0)
    n_obj = torch.tensor([len(boxes)], dtype=torch.long)

    targets = build_targets(
        boxes=batch_boxes,
        labels=batch_labels,
        n_obj=n_obj,
        grid_shapes=GRID_SHAPES,
        image_h=IMAGE_H,
        image_w=IMAGE_W,
        anchors=anchors,
        anchor_t=ANCHOR_T,
    )

    positives_per_scale = [
        len(target[0]) for target in targets
    ]

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    total = len(boxes)
    covered = matched_any.sum().item()

    print("\n" + "=" * 65)
    print("OVERALL ANCHOR COVERAGE")
    print("=" * 65)

    print(f"Total GT objects:             {total}")
    print(f"GT matched to >=1 anchor:    {covered}")
    print(f"GT with zero anchor matches:  {total - covered}")
    print(f"Anchor coverage:              {100 * covered / total:.2f}%")

    print("\n" + "=" * 65)
    print("POSITIVE ASSIGNMENTS PER SCALE")
    print("=" * 65)

    for i, name in enumerate(["P3 (stride 8)", "P4 (stride 16)", "P5 (stride 32)"]):
        print(
            f"{name:20s}: "
            f"{positives_per_scale[i]:8d} positives | "
            f"{matched_per_scale[i].sum().item():8d} direct matches"
        )

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("PER-CLASS ANCHOR COVERAGE")
    print("=" * 100)

    print(
        f"{'Class':18s} {'GT count':>10s} {'Covered':>10s} "
        f"{'Unmatched':>12s} {'Coverage %':>12s} "
        f"{'Avg anchors/GT':>16s}"
    )

    for cls_id, cls_name in enumerate(CLASS_NAMES):

        mask = labels == cls_id
        count = mask.sum().item()

        if count == 0:
            print(f"{cls_name:18s} {0:10d} (no objects)")
            continue

        class_covered = matched_any[mask].sum().item()
        unmatched = count - class_covered

        avg_matches = matched_per_scale[:, mask].sum().item() / count

        coverage = 100.0 * class_covered / count

        print(
            f"{cls_name:18s} {count:10d} {class_covered:10d} "
            f"{unmatched:12d} {coverage:11.2f}% "
            f"{avg_matches:15.3f}"
        )

    # --------------------------------------------------------
    # Box-size distribution
    # --------------------------------------------------------

    areas = widths * heights

    print("\n" + "=" * 65)
    print("GROUND-TRUTH BOX SIZE DISTRIBUTION")
    print("=" * 65)

    for name, values in [
        ("Width (pixels)", widths),
        ("Height (pixels)", heights),
        ("Area (pixels^2)", areas),
    ]:

        print(f"\n{name}:")
        print(
            f"  min={values.min().item():.1f}, "
            f"p10={torch.quantile(values, 0.10).item():.1f}, "
            f"p25={torch.quantile(values, 0.25).item():.1f}, "
            f"median={torch.quantile(values, 0.50).item():.1f}, "
            f"p75={torch.quantile(values, 0.75).item():.1f}, "
            f"p90={torch.quantile(values, 0.90).item():.1f}, "
            f"max={values.max().item():.1f}"
        )

    # --------------------------------------------------------
    # Size distribution by class
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("PER-CLASS BOX SIZE (MEDIAN WIDTH x HEIGHT)")
    print("=" * 100)

    print(
        f"{'Class':18s} {'Count':>10s} "
        f"{'Median W':>12s} {'Median H':>12s} "
        f"{'Median Area':>14s}"
    )

    for cls_id, cls_name in enumerate(CLASS_NAMES):

        mask = labels == cls_id
        count = mask.sum().item()

        if count == 0:
            continue

        cls_w = widths[mask]
        cls_h = heights[mask]
        cls_area = areas[mask]

        print(
            f"{cls_name:18s} {count:10d} "
            f"{cls_w.median().item():12.1f} "
            f"{cls_h.median().item():12.1f} "
            f"{cls_area.median().item():14.1f}"
        )


# ============================================================
# MAIN
# ============================================================


def main():
    annotation_files = find_annotation_file(DATA_DIR, SPLIT)

    if not annotation_files:
        print("ERROR: No annotation files found.")
        return

    records = load_bdd_annotations(annotation_files)

    # load_bdd_annotations returns a list of annotation dictionaries.
    # extract_objects processes those dictionaries and returns boxes/labels.
    boxes, labels = extract_objects(records)

    if len(boxes) == 0:
        print("ERROR: No valid boxes found.")
        return

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    calculate_anchor_statistics(boxes, labels)


if __name__ == "__main__":
    main()