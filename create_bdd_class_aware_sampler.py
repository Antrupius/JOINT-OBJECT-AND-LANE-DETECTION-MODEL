
import os
import json
import random
import csv
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

ANN_DIR = r"D:\MinorProjectData\BDD100K\train\ann"

OUTPUT_CSV = "bdd_class_aware_sampling.csv"

SEED = 42

CLASSES = [
    "car",
    "truck",
    "bus",
    "person",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
]

# Repeat factors based on previous class AP results
REPEAT_FACTORS = {
    "car": 1,
    "truck": 2,
    "bus": 2,
    "person": 2,
    "bike": 3,
    "motor": 3,
    "traffic light": 2,
    "traffic sign": 2,
}

# =========================================================
# SCAN ANNOTATIONS
# =========================================================

print("Scanning BDD100K annotations...")

files = [
    f for f in os.listdir(ANN_DIR)
    if f.endswith(".json")
]

print(f"Total images: {len(files)}")

image_labels = {}
class_image_counts = Counter()
class_instance_counts = Counter()

for i, filename in enumerate(files):

    path = os.path.join(ANN_DIR, filename)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        objects = data.get("objects", [])

        present_classes = set()

        for obj in objects:

            category = obj.get("classTitle")

            if category in CLASSES:

                class_instance_counts[category] += 1
                present_classes.add(category)

        image_labels[filename] = present_classes

        for category in present_classes:
            class_image_counts[category] += 1

    except Exception as e:
        print(f"Error reading {filename}: {e}")

    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1}/{len(files)}")

# =========================================================
# CALCULATE IMAGE REPEAT COUNTS
# =========================================================

print("\nCalculating class-aware repeat counts...")

sampling_list = []

for filename, present_classes in image_labels.items():

    if not present_classes:
        repeat = 1

    else:
        # Highest repeat factor among classes present
        repeat = max(
            REPEAT_FACTORS[c]
            for c in present_classes
        )

    sampling_list.extend([filename] * repeat)

# =========================================================
# SHUFFLE
# =========================================================

random.seed(SEED)
random.shuffle(sampling_list)

# =========================================================
# SAVE CSV
# =========================================================

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)
    writer.writerow(["filename"])

    for filename in sampling_list:
        writer.writerow([filename])

# =========================================================
# REPORT
# =========================================================

print("\n" + "=" * 60)
print("CLASS-AWARE SAMPLING SUMMARY")
print("=" * 60)

print(f"Original unique images: {len(image_labels)}")
print(f"Total sampled exposures: {len(sampling_list)}")
print(f"Additional exposures: {len(sampling_list) - len(image_labels)}")

print("\nClass repeat factors:")

for category in CLASSES:
    print(
        f"{category:<18} "
        f"{REPEAT_FACTORS[category]}x | "
        f"{class_image_counts[category]} images"
    )

print(f"\nSaved sampling list to: {OUTPUT_CSV}")
print("Complete.")
