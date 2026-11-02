"""Deterministic image-level class-aware sampling for BDD100K.

Builds one sampling weight per BDD100K training image from the classes present
in its annotation. Sampling is with replacement; each epoch still contains
len(dataset) sampled entries. The original dataset and annotations are never
modified.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

SAMPLER_VERSION = 1
CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "person": 3,
    "rider": 4,
    "bike": 4,
    "motor": 5,
    "traffic light": 6,
    "traffic sign": 7,
}
CLASS_NAMES = [
    "car", "truck", "bus", "pedestrian", "cyclist", "motorcycle",
    "traffic_light", "traffic_sign",
]
RARE_CLASS_IDS = {1, 2, 3, 4, 5}


def _annotation_classes(annotation_path: Path) -> set[int]:
    with annotation_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    present: set[int] = set()
    for obj in ann.get("objects", []):
        category = obj.get("classTitle", "").strip().lower()
        if category == "lane" or category not in CLASS_MAP:
            continue
        if obj.get("geometryType", "").lower() != "rectangle":
            continue
        exterior = obj.get("points", {}).get("exterior", [])
        if len(exterior) < 2:
            continue
        present.add(CLASS_MAP[category])
    return present


def build_image_weights(dataset, data_root: str, cache_path: str,
                        car_only_weight: float = 0.35,
                        max_class_boost: float = 3.0):
    """Return float weights and annotation statistics, using a validated cache."""
    if not (0.0 < car_only_weight <= 1.0):
        raise ValueError("car_only_weight must be in (0, 1].")
    if max_class_boost < 1.0:
        raise ValueError("max_class_boost must be >= 1.")

    ann_dir = Path(data_root) / "train" / "ann"
    ids = list(dataset.ids)
    ann_paths = [ann_dir / (name + ".json") for name in ids]
    missing = [str(p) for p in ann_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing BDD annotations, first example: {missing[0]}")

    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    signatures = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in ann_paths]
    class_sets = None
    if cache.is_file():
        try:
            saved = torch.load(cache, map_location="cpu", weights_only=False)
            if (saved.get("sampler_version") == SAMPLER_VERSION
                    and saved.get("signatures") == signatures
                    and saved.get("ids") == ids):
                class_sets = saved["class_sets"]
                print(f"Loaded annotation index cache: {cache}", flush=True)
        except Exception as exc:
            print(f"Ignoring unreadable sampler cache ({exc}); rebuilding.", flush=True)

    if class_sets is None:
        class_sets = []
        for i, path in enumerate(ann_paths, 1):
            class_sets.append(sorted(_annotation_classes(path)))
            if i % 10000 == 0:
                print(f"Indexed annotations: {i}/{len(ann_paths)}", flush=True)
        tmp = Path(str(cache) + ".tmp")
        torch.save({
            "sampler_version": SAMPLER_VERSION,
            "ids": ids,
            "signatures": signatures,
            "class_sets": class_sets,
        }, tmp)
        os.replace(tmp, cache)
        print(f"Saved annotation index cache: {cache}", flush=True)

    image_freq = [0] * len(CLASS_NAMES)
    for classes in class_sets:
        for class_id in classes:
            image_freq[class_id] += 1

    nonzero = [n for n in image_freq if n > 0]
    if not nonzero:
        raise RuntimeError("No supported object classes were found in BDD100K annotations.")
    reference_freq = max(nonzero)
    class_boost = [1.0] * len(CLASS_NAMES)
    for class_id, freq in enumerate(image_freq):
        if freq:
            # Square-root inverse-frequency scaling is deliberately moderated.
            class_boost[class_id] = min(max_class_boost, (reference_freq / freq) ** 0.5)

    weights = []
    group_counts = {"car_only": 0, "minority_present": 0, "common_only": 0, "no_objects": 0}
    for classes in class_sets:
        present = set(classes)
        if not present:
            weight = 0.35
            group_counts["no_objects"] += 1
        elif present == {0}:
            weight = car_only_weight
            group_counts["car_only"] += 1
        else:
            boost = max(class_boost[c] for c in present)
            rare_count = len(present & RARE_CLASS_IDS)
            # A small composition bonus rewards images with several minority classes.
            composition_bonus = 1.0 + 0.12 * max(0, rare_count - 1)
            weight = min(max_class_boost, boost * composition_bonus)
            if rare_count:
                group_counts["minority_present"] += 1
            else:
                group_counts["common_only"] += 1
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.double)
    if not torch.isfinite(weights_tensor).all() or (weights_tensor <= 0).any():
        raise RuntimeError("Sampler produced invalid image weights.")

    stats = {
        "class_image_frequency": dict(zip(CLASS_NAMES, image_freq)),
        "class_boost": dict(zip(CLASS_NAMES, [round(x, 3) for x in class_boost])),
        "image_groups": group_counts,
        "weight_min": float(weights_tensor.min()),
        "weight_max": float(weights_tensor.max()),
        "weight_mean": float(weights_tensor.mean()),
        "sampler_version": SAMPLER_VERSION,
    }
    return weights_tensor, class_sets, stats


def make_epoch_indices(weights: torch.Tensor, epoch_seed: int, num_samples: int):
    """Deterministic weighted epoch order; repeated indices are intentional."""
    generator = torch.Generator(device="cpu").manual_seed(int(epoch_seed))
    return torch.multinomial(weights, num_samples=num_samples,
                             replacement=True, generator=generator).tolist()


def print_sampling_report(weights: torch.Tensor, class_sets, stats: dict,
                          num_samples: int):
    print("\n=== BDD100K class-aware sampling report ===", flush=True)
    print("Image-level class frequencies:", flush=True)
    for name, count in stats["class_image_frequency"].items():
        print(f"  {name:15s}: {count:6d} images | boost={stats['class_boost'][name]:.2f}", flush=True)
    print("Image composition groups:", flush=True)
    for name, count in stats["image_groups"].items():
        print(f"  {name:18s}: {count:6d}", flush=True)
    probs = weights / weights.sum()
    expected_class_draws = [0.0] * len(CLASS_NAMES)
    for i, classes in enumerate(class_sets):
        for class_id in classes:
            expected_class_draws[class_id] += float(probs[i]) * num_samples
    print("Expected class-containing sampled images per epoch (overlap allowed):", flush=True)
    for name, count in zip(CLASS_NAMES, expected_class_draws):
        print(f"  {name:15s}: {count:8.1f}", flush=True)
    print(f"Weights: min={stats['weight_min']:.3f}, mean={stats['weight_mean']:.3f}, max={stats['weight_max']:.3f}", flush=True)
    print("Note: expected class counts overlap because one image can contain multiple classes.\n", flush=True)
