
import os
import json
import glob
import argparse
from collections import Counter, defaultdict

# Your project's 8 detection classes
CLASSES = [
    "car", "truck", "bus", "pedestrian",
    "cyclist", "motorcycle", "traffic_light", "traffic_sign"
]

# Map dataset-specific labels to your model's classes
ALIASES = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "rider": "cyclist",
    "cyclist": "cyclist",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
}


def normalize_class(name):
    return ALIASES.get(name.lower().strip())


def print_distribution(name, instance_counts, image_counts, total_images):
    print(f"\n{'=' * 65}")
    print(f"DATASET: {name}")
    print(f"Total images: {total_images}")
    print(f"Total instances: {sum(instance_counts.values())}")
    print("-" * 65)
    print(f"{'Class':<18} {'Instances':>12} {'Images':>12}")
    print("-" * 65)

    for cls in CLASSES:
        print(
            f"{cls:<18} "
            f"{instance_counts[cls]:>12} "
            f"{image_counts[cls]:>12}"
        )

    print("=" * 65)
    

def count_bdd(ann_dir):
    instances = Counter()
    images = Counter()
    total_images = 0

    files = glob.glob(os.path.join(ann_dir, "*.json"))

    print(f"Found {len(files)} BDD annotation files.")

    for file in files:
        total_images += 1
        image_classes = set()

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # BDD individual annotation format
        labels = data.get("labels", [])

        for obj in labels:
            category = obj.get("category", "")
            cls = normalize_class(category)

            if cls is not None:
                instances[cls] += 1
                image_classes.add(cls)

        for cls in image_classes:
            images[cls] += 1

    return instances, images, total_images



def count_kitti(label_dir):
    instances = Counter()
    images = Counter()
    total_images = 0

    files = glob.glob(os.path.join(label_dir, "*.txt"))

    for file in files:
        total_images += 1
        image_classes = set()

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                # KITTI format: class is the first field
                category = parts[0]
                cls = normalize_class(category)

                if cls is not None:
                    instances[cls] += 1
                    image_classes.add(cls)

        for cls in image_classes:
            images[cls] += 1

    return instances, images, total_images


def inspect_bdd_cache(cache_path):
    print("\nInspecting existing BDD annotation cache...")

    if not os.path.exists(cache_path):
        print("Cache not found:", cache_path)
        return

    import torch
    data = torch.load(cache_path, map_location="cpu", weights_only=False)

    print("Cache found:", cache_path)
    print("Object type:", type(data))

    if isinstance(data, dict):
        print("Cache keys:", list(data.keys()))

        for key, value in data.items():
            if isinstance(value, (list, tuple, dict)):
                print(f"  {key}: {type(value).__name__}, length={len(value)}")
            else:
                print(f"  {key}: {type(value).__name__}")

    elif isinstance(data, (list, tuple)):
        print("Cache length:", len(data))
        if len(data) > 0:
            print("First entry type:", type(data[0]))
            print("First entry preview:", str(data[0])[:500])

    print("\nCache structure printed. We can adapt the script to extract")
    print("class counts directly if the cache contains class annotations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bdd-json", type=str, default=None,
                        help="Path to BDD100K labels JSON")
    parser.add_argument("--kitti-labels", type=str, default=None,
                        help="Path to KITTI training label_2 folder")
    parser.add_argument("--bdd-cache", type=str, default=None,
                        help="Path to existing BDD annotation index cache")

    args = parser.parse_args()

    if args.bdd_json:
        inst, imgs, total = count_bdd(args.bdd_json)
        print_distribution("BDD100K", inst, imgs, total)

    if args.kitti_labels:
        inst, imgs, total = count_kitti(args.kitti_labels)
        print_distribution("KITTI", inst, imgs, total)

    if args.bdd_cache:
        inspect_bdd_cache(args.bdd_cache)
