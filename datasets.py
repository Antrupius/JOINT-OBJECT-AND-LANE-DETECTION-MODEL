"""Datasets for RP1: real parsers (KITTI / BDD100K / OpenLane) + synthetic smoke test."""
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from config import ModelConfig, DataConfig


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DRIVING_CLASSES = {0: "car", 1: "truck", 2: "bus", 3: "pedestrian",
                   4: "cyclist", 5: "motorcycle", 6: "traffic_light", 7: "traffic_sign"}


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x / 255.0 - MEAN) / STD


def boxes_to_coco(boxes: np.ndarray) -> np.ndarray:
    """(N,4) [x1,y1,x2,y2] -> (N,4) [x,y,w,h] COCO format."""
    return np.stack([boxes[:, 0], boxes[:, 1],
                     boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)



def lane_targets_from_lines(
    lines: List[List[Tuple[float, float]]],
    cfg: ModelConfig,
) -> np.ndarray:
    """Convert lane polylines into per-row anchor labels.
    0 = background; 1..grid_w = horizontal column labels.
    """
    H_row, W_feat = cfg.grid_h, cfg.grid_w
    targets = np.zeros((cfg.num_lanes, H_row), dtype=np.int64)

    for li, line in enumerate(lines):
        if li >= cfg.num_lanes:
            break
        if line is None or len(line) < 2:
            continue                      # empty slot (was `break`, which dropped later slots)

        points = np.asarray(line, dtype=np.float64)
        xs, ys = points[:, 0], points[:, 1]

        # Sort by vertical position before interpolation.
        order = np.argsort(ys)
        xs, ys = xs[order], ys[order]

        # np.interp needs increasing x-coordinates; average duplicate y values.
        unique_ys, inverse = np.unique(ys, return_inverse=True)
        if len(unique_ys) < 2:
            continue

        summed_xs = np.zeros(len(unique_ys), dtype=np.float64)
        counts = np.zeros(len(unique_ys), dtype=np.int64)
        np.add.at(summed_xs, inverse, xs)
        np.add.at(counts, inverse, 1)
        unique_xs = summed_xs / counts

        for r in range(H_row):
            y_pix = (r + 0.5) * (cfg.image_h / H_row)

            if y_pix < unique_ys[0] or y_pix > unique_ys[-1]:
                continue

            x_pix = float(np.interp(y_pix, unique_ys, unique_xs))
            col = int(round(x_pix / (cfg.image_w / W_feat)))

            if 0 <= col < W_feat:
                targets[li, r] = col + 1

    return targets



def assign_lane_slots(lanes, cfg: ModelConfig):
    """Ego-relative slot assignment (UFLD/CULane convention).
    slot 0 = 2nd lane left of centre, 1 = nearest left, 2 = nearest right, 3 = 2nd right.
    A slot has the same physical meaning in every image; empty slots stay None.
    (Old code: leftmost-4 lanes sorted by mean x -> slot meaning changed with lane count.)"""
    center = cfg.image_w / 2.0
    left, right = [], []
    for pts in lanes:
        if len(pts) < 2:
            continue
        x_near = max(pts, key=lambda p: p[1])[0]          # x at the lane point closest to the car
        (left if x_near < center else right).append((x_near, pts))
    left.sort(key=lambda t: -t[0])                          # nearest-to-centre first
    right.sort(key=lambda t: t[0])
    slots = [None] * cfg.num_lanes
    half = cfg.num_lanes // 2
    for k, (_, pts) in enumerate(left[:half]):
        slots[half - 1 - k] = pts                           # nearest left -> slot half-1
    for k, (_, pts) in enumerate(right[:half]):
        slots[half + k] = pts                               # nearest right -> slot half
    return slots



class SyntheticLaneObjectDataset(Dataset):
    """Procedurally drawn driving frames: road + lane polylines + object boxes.
    Lets the full RP1 pipeline run end-to-end before any real dataset is downloaded."""

    def __init__(self, n: int, cfg: ModelConfig, seed: int = 0):
        self.n, self.cfg, self.rng = n, cfg, random.Random(seed)
        self.H, self.W = cfg.image_h, cfg.image_w

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = random.Random(idx * 7919)
        img = Image.new("RGB", (self.W, self.H), (60, 64, 70))
        d = ImageDraw.Draw(img)
        horizon = rng.randint(140, 180)
        d.rectangle([0, horizon, self.W, self.H], fill=(85, 88, 92))       # road
        # --- lanes: 4 polylines with random offsets/slopes ---
        lanes = []
        for li in range(self.cfg.num_lanes):
            base = self.W * (0.15 + 0.23 * li) + rng.randint(-15, 15)
            slope = rng.uniform(-0.25, 0.25)
            pts = [(base + slope * (y - horizon), y) for y in range(horizon, self.H, 12)]
            lanes.append(pts)
            d.line(pts, fill=(255, 255, 255), width=4)
        # --- objects ---
        boxes, labels = [], []
        for _ in range(rng.randint(1, 5)):
            cls = rng.randint(0, self.cfg.num_classes - 1)
            w = rng.randint(40, 140); h = rng.randint(40, 140)
            x1 = rng.randint(0, self.W - w); y1 = rng.randint(horizon + 5, self.H - h)
            boxes.append([x1, y1, x1 + w, y1 + h]); labels.append(cls)
            d.rectangle([x1, y1, x1 + w, y1 + h], outline=(220, 60, 60), width=3)
        x = normalize(torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1))
        lane_targets = torch.from_numpy(lane_targets_from_lines(lanes, self.cfg))
        return x, {
            "boxes": torch.tensor(np.asarray(boxes, dtype=np.float32).reshape(-1, 4)),
            "labels": torch.tensor(np.asarray(labels, dtype=np.int64).reshape(-1)),
            "lane_targets": lane_targets,
            "has_object_labels": True,
            "has_lane_labels": True,
        }



class KittiOODataset(Dataset):
    """KITTI object detection dataset with explicit split handling."""

    CLASS_MAP = {
        "Car": 0,
        "Truck": 1,
        "Bus": 2,
        "Pedestrian": 3,
        "Cyclist": 4,
        "Motorcycle": 5,
    }

    def __init__(self, root: str, cfg: ModelConfig, split: str = "training"):
        self.root = root
        self.cfg = cfg

        split_aliases = {
            "train": "training",
            "training": "training",
            "test": "testing",
            "testing": "testing",
        }

        if split not in split_aliases:
            raise ValueError(
                f"Unsupported KITTI split: {split!r}. "
                "Use 'train'/'training' or 'test'/'testing'."
            )

        self.split = split_aliases[split]
        split_dir = os.path.join(root, self.split)

        self.image_dir = os.path.join(self.root, "image_2")
        self.label_dir = os.path.join(self.root, "label_2")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"KITTI image directory not found: {self.image_dir}"
            )

        self.has_labels = os.path.isdir(self.label_dir)

        if self.split == "training" and not self.has_labels:
            raise FileNotFoundError(
                f"KITTI training labels not found: {self.label_dir}"
            )

        self.ids = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(self.image_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if not self.ids:
            raise RuntimeError(
                f"No KITTI images found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.image_dir, self.ids[idx] + ".png"
        )

        # Support jpg/jpeg images as well.
        if not os.path.isfile(image_path):
            for ext in (".jpg", ".jpeg"):
                candidate = os.path.join(
                    self.image_dir, self.ids[idx] + ext
                )
                if os.path.isfile(candidate):
                    image_path = candidate
                    break

        img = Image.open(image_path).convert("RGB")
        original_w, original_h = img.size
        img = img.resize((self.cfg.image_w, self.cfg.image_h))

        boxes, labels = [], []
        label_path = os.path.join(
            self.label_dir, self.ids[idx] + ".txt"
        )

        if self.has_labels and os.path.isfile(label_path):
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 8 or parts[0] == "DontCare":
                        continue

                    cls = self.CLASS_MAP.get(parts[0])
                    if cls is None:
                        continue

                    x1, y1, x2, y2 = map(float, parts[4:8])

                    x1, x2 = sorted((
                        x1 * self.cfg.image_w / original_w,
                        x2 * self.cfg.image_w / original_w,
                    ))
                    y1, y2 = sorted((
                        y1 * self.cfg.image_h / original_h,
                        y2 * self.cfg.image_h / original_h,
                    ))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)

        image_tensor = normalize(
            torch.from_numpy(
                np.asarray(img, dtype=np.float32)
            ).permute(2, 0, 1)
        )

        lane_targets = torch.zeros(
            (self.cfg.num_lanes, self.cfg.grid_h),
            dtype=torch.long,
        )

        return image_tensor, {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "lane_targets": lane_targets,
            "has_object_labels": self.has_labels,
            "has_lane_labels": False,
        }


class BDD100KDataset(Dataset):
    """Dataset Ninja BDD100K images with object and lane annotations."""

    CLASS_MAP = {
        "car": 0,
        "truck": 1,
        "bus": 2,
        "person": 3,
        "rider": 4,
        "motor": 5,
        "bike": 4,
        "traffic light": 6,      # matches DRIVING_CLASSES / KITTI ordering
        "traffic sign": 7,
    }

    def __init__(self, root: str, cfg: ModelConfig, split: str = "train"):
        self.root = root
        self.cfg = cfg
        self.split = split

        
        folder = "val" if split in ("val", "validation") else split
        self.image_dir = os.path.join(root, folder, "img")
        self.annotation_dir = os.path.join(root, folder, "ann")


        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"BDD100K image directory not found: {self.image_dir}")
        if not os.path.isdir(self.annotation_dir):
            raise FileNotFoundError(f"BDD100K annotation directory not found: {self.annotation_dir}")

        self.ids = sorted(
            p.name for p in os.scandir(self.image_dir)
            if p.is_file() and p.name.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.isfile(os.path.join(self.annotation_dir, p.name + ".json"))
        )

        if not self.ids:
            raise RuntimeError(f"No matching BDD100K image/annotation pairs in {split}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        image_path = os.path.join(self.image_dir, name)
        annotation_path = os.path.join(self.annotation_dir, name + ".json")

        img = Image.open(image_path).convert("RGB")
        original_w, original_h = img.size
        img = img.resize((self.cfg.image_w, self.cfg.image_h))

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        boxes, labels = [], []
        lane_points = []

        for obj in annotation.get("objects", []):
            category = obj.get("classTitle", "").strip().lower()
            geometry = obj.get("geometryType", "").lower()
            exterior = obj.get("points", {}).get("exterior", [])

            
            
            if category == "lane":
                if geometry == "line" and len(exterior) >= 2:
                    scaled_points = [
                        (
                            float(point[0]) * self.cfg.image_w / original_w,
                            float(point[1]) * self.cfg.image_h / original_h,
                        )
                        for point in exterior
                        if len(point) >= 2
                    ]
                    if len(scaled_points) >= 2:
                        lane_points.append(scaled_points)
                continue



            if category not in self.CLASS_MAP or geometry != "rectangle" or len(exterior) < 2:
                continue

            (x1, y1), (x2, y2) = exterior[:2]
            x1, x2 = sorted((x1 * self.cfg.image_w / original_w,
                             x2 * self.cfg.image_w / original_w))
            y1, y2 = sorted((y1 * self.cfg.image_h / original_h,
                             y2 * self.cfg.image_h / original_h))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.CLASS_MAP[category])

        lane_targets = lane_targets_from_lines(lane_points, self.cfg)

        x = normalize(
            torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)
        )

        return x, {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "lane_targets": lane_targets,
            "has_object_labels": True,
            "has_lane_labels": bool((lane_targets > 0).any()),
        }




class OpenLaneDataset(Dataset):
    """OpenLane dataset with reliable image-annotation path matching."""

    def __init__(self, root: str, cfg: ModelConfig, json_path: str):
        self.cfg = cfg
        self.image_dir = os.path.join(root, "images")
        self.annotation_files = []
        self.image_files = []

        if not os.path.isdir(json_path):
            raise FileNotFoundError(
                f"OpenLane annotation directory not found: {json_path}"
            )

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"OpenLane image directory not found: {self.image_dir}"
            )

        json_count = 0
        missing_count = 0

        # Index JPGs once. The segment folder is the immediate parent of
        # each image, including images under images_validation_* wrappers.
        image_index = {}
        for image_dirpath, _, image_filenames in os.walk(self.image_dir):
            segment_name = os.path.basename(image_dirpath).lower()
            for image_filename in image_filenames:
                if not image_filename.lower().endswith(".jpg"):
                    continue
                key = (segment_name, image_filename.lower())
                image_path = os.path.join(image_dirpath, image_filename)
                image_index.setdefault(key, []).append(image_path)

        # Match annotations by segment folder and frame filename.
        for dirpath, _, filenames in os.walk(json_path):
            for filename in filenames:
                if not filename.lower().endswith(".json"):
                    continue

                json_count += 1
                annotation_path = os.path.join(dirpath, filename)
                segment_name = os.path.basename(dirpath).lower()
                image_filename = os.path.splitext(filename)[0] + ".jpg"
                candidates = image_index.get(
                    (segment_name, image_filename.lower()), []
                )

                if candidates:
                    # Deterministic selection if duplicate copies exist.
                    image_path = min(
                        candidates,
                        key=lambda path: (path.count(os.sep), path.lower()),
                    )
                    self.annotation_files.append(annotation_path)
                    self.image_files.append(image_path)
                else:
                    missing_count += 1

        # Keep annotation-image pairs in the same sorted order.
        pairs = sorted(zip(self.annotation_files, self.image_files))
        self.annotation_files = [pair[0] for pair in pairs]
        self.image_files = [pair[1] for pair in pairs]

        print(
            f"OpenLane split: {json_path}\n"
            f"  JSON annotations found: {json_count}\n"
            f"  Matched image-annotation pairs: {len(self.annotation_files)}\n"
            f"  Annotations without matching images: {missing_count}"
        )

        if not self.annotation_files:
            raise RuntimeError(
                "No matching OpenLane images found.\n"
                f"Images: {self.image_dir}\n"
                f"Annotations: {json_path}"
            )

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        annotation_path = self.annotation_files[idx]
        image_path = self.image_files[idx]

        with open(annotation_path, "r", encoding="utf-8") as f:
            rec = json.load(f)

        with Image.open(image_path) as source:
            img = source.convert("RGB")

        original_w, original_h = img.size
        img = img.resize((self.cfg.image_w, self.cfg.image_h))

        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        lanes = []

        for lane in rec.get("lane_lines", []):
            uv = lane.get("uv")
            if not isinstance(uv, (list, tuple)) or len(uv) < 2:
                continue

            xs, ys = uv[0], uv[1]
            if len(xs) != len(ys):
                continue

            pts = []
            for px, py in zip(xs, ys):
                if px is None or py is None:
                    continue

                x = float(px) * self.cfg.image_w / original_w
                y = float(py) * self.cfg.image_h / original_h

                if (
                    0 <= x < self.cfg.image_w
                    and 0 <= y < self.cfg.image_h
                ):
                    pts.append((x, y))

            if len(pts) >= 2:
                lanes.append(pts)

        # Ego-relative slots: same slot = same physical lane in every image.
        slots = assign_lane_slots(lanes, self.cfg)
        lane_targets = torch.from_numpy(
            lane_targets_from_lines(slots, self.cfg)
        ).long()

        image_tensor = normalize(
            torch.from_numpy(
                np.asarray(img, dtype=np.float32)
            ).permute(2, 0, 1)
        )

        return image_tensor, {
            "boxes": boxes,
            "labels": labels,
            "lane_targets": lane_targets,
            "has_object_labels": False,
            "has_lane_labels": bool(lanes),
        }




def make_dataset(name: str, cfg: ModelConfig, dcfg: DataConfig, split: str = "train"):
    if name == "synthetic":
        return SyntheticLaneObjectDataset(dcfg.synthetic_n, cfg)

    if name == "kitti":
        return KittiOODataset(dcfg.kitti_dir, cfg, split=split)

    if name == "bdd100k":
        return BDD100KDataset(
            dcfg.bdd_dir,
            cfg,
            split=split
        )

    if name == "openlane":
        annotation_dir = os.path.join(
            dcfg.openlane_annotations_dir,
            "training" if split == "train" else "validation"
        )
        return OpenLaneDataset(
            dcfg.openlane_dir,
            cfg,
            annotation_dir
        )

    raise ValueError(f"unknown dataset {name}")



def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    B = len(batch)

    max_n = max(b[1]["boxes"].shape[0] for b in batch)
    boxes = torch.zeros((B, max_n, 4), dtype=torch.float32)
    labels = torch.zeros((B, max_n), dtype=torch.int64)
    n_obj = torch.zeros((B,), dtype=torch.int64)
    lane = torch.stack([
    torch.as_tensor(b[1]["lane_targets"], dtype=torch.long)
    for b in batch
    ])

    has_object_labels = torch.tensor(
        [b[1]["has_object_labels"] for b in batch],
        dtype=torch.bool
    )
    has_lane_labels = torch.tensor(
        [b[1]["has_lane_labels"] for b in batch],
        dtype=torch.bool
    )

    for i, b in enumerate(batch):
        n = b[1]["boxes"].shape[0]
        if n > 0:
            boxes[i, :n] = b[1]["boxes"]
            labels[i, :n] = b[1]["labels"]
        n_obj[i] = n

    return xs, {
        "boxes": boxes,
        "labels": labels,
        "n_obj": n_obj,
        "lane_targets": lane,
        "has_object_labels": has_object_labels,
        "has_lane_labels": has_lane_labels,
    }

