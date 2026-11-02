
import os
import json
from collections import Counter

BDD_DIR = r"D:\MinorProjectData\BDD100K\train\ann"

CLASSES = [
    "car", "truck", "bus", "person",
    "bike", "motor", "traffic light", "traffic sign"
]

instance_counts = Counter()
image_counts = Counter()

print("Scanning BDD100K training annotations...\n")

files = [f for f in os.listdir(BDD_DIR) if f.endswith(".json")]
print(f"Total annotation files: {len(files)}")

valid_images = 0

for i, filename in enumerate(files):

    path = os.path.join(BDD_DIR, filename)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        objects = data.get("objects", [])
        present_classes = set()

        for obj in objects:
            category = obj.get("classTitle")

            if category in CLASSES:
                instance_counts[category] += 1
                present_classes.add(category)

        for category in present_classes:
            image_counts[category] += 1

        valid_images += 1

    except Exception as e:
        print(f"Error in {filename}: {e}")

    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1}/{len(files)}")

print("\n" + "=" * 65)
print("BDD100K CLASS DISTRIBUTION")
print("=" * 65)

print(f"Valid images scanned: {valid_images}\n")

print(f"{'Class':<20}{'Instances':<18}{'Images containing class'}")
print("-" * 65)

for category in CLASSES:
    print(
        f"{category:<20}"
        f"{instance_counts[category]:<18}"
        f"{image_counts[category]}"
    )

print("\n" + "=" * 65)
print("CLASS BALANCE SUMMARY")
print("=" * 65)

min_images = min(image_counts[c] for c in CLASSES)

rarest_class = min(CLASSES, key=lambda c: image_counts[c])

print(f"Rarest class: {rarest_class}")
print(f"Images containing rarest class: {min_images}")
print(f"Simple theoretical estimate: {min_images * len(CLASSES)}")

print("\nScan complete.")
