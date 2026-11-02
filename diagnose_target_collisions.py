
import torch
from collections import defaultdict
from config import ModelConfig, DataConfig
from datasets import make_dataset, collate_fn
from torch.utils.data import DataLoader
from models.yolov5_detection import (
    DEFAULT_ANCHORS,
    build_targets,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATASET = "bdd100k"
SPLIT = "train"
NUM_IMAGES = 1000
BATCH_SIZE = 4

CLASS_NAMES = [
    "car", "truck", "bus", "pedestrian",
    "cyclist", "motorcycle", "traffic_light", "traffic_sign"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    mc = ModelConfig()
    dc = DataConfig()
    dc.dataset_name = "bdd100k"
    dc.bdd_dir = r"D:\MinorProjectData\BDD100K"

    dataset = make_dataset(DATASET, mc, dc, split=SPLIT)
    indices = list(range(min(NUM_IMAGES, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    anchors = torch.tensor(
        DEFAULT_ANCHORS, dtype=torch.float32, device=DEVICE
    )

    grid_shapes = [
        (48, 80),  # P3
        (24, 40),  # P4
        (12, 20),  # P5
    ]

    collisions = [defaultdict(list) for _ in grid_shapes]
    total_assignments = [0] * len(grid_shapes)
    unique_slots = [0] * len(grid_shapes)

    gt_count = [0] * 8
    matched_gt = [0] * 8
    unmatched_gt = [0] * 8

    for batch_idx, batch in enumerate(loader):
        images, targets = batch

        boxes = targets["boxes"].to(DEVICE)
        labels = targets["labels"].to(DEVICE)
        n_obj = targets["n_obj"].to(DEVICE)

        boxes = boxes.to(DEVICE)
        labels = labels.to(DEVICE)
        n_obj = n_obj.to(DEVICE)

        B = boxes.shape[0]

        for b in range(B):
            for j in range(int(n_obj[b])):
                cls = int(labels[b, j])
                if 0 <= cls < 8:
                    gt_count[cls] += 1

        targets = build_targets(
            boxes,
            labels,
            n_obj,
            grid_shapes,
            image_h=384,
            image_w=640,
            anchors=anchors,
            anchor_t=4.0,
        )

        # Track which GT object generated each assignment.
        # We reproduce anchor matching to recover GT indices.
        for si, (gh, gw) in enumerate(grid_shapes):
            scale = torch.tensor(
                [640 / gw, 384 / gh],
                device=DEVICE,
                dtype=torch.float32,
            )

            for b in range(B):
                valid = torch.arange(
                    boxes.shape[1], device=DEVICE
                ) < n_obj[b]

                obj_indices = valid.nonzero(as_tuple=True)[0]

                if obj_indices.numel() == 0:
                    continue

                tb = boxes[b, obj_indices].float()
                cls_ids = labels[b, obj_indices].long()

                w = (tb[:, 2] - tb[:, 0]).clamp_min(1.0)
                h = (tb[:, 3] - tb[:, 1]).clamp_min(1.0)

                cx = (tb[:, 0] + tb[:, 2]) / 2
                cy = (tb[:, 1] + tb[:, 3]) / 2

                gxy = torch.stack([cx, cy], dim=1) / scale
                gwh = torch.stack([w, h], dim=1) / scale

                a_grid = anchors[si] / scale
                r = gwh[:, None, :] / a_grid[None, :, :]

                match = torch.maximum(r, 1.0 / r).amax(dim=2) < 4.0
                ti, ai = match.nonzero(as_tuple=True)

                if ti.numel() == 0:
                    continue

                gxy_m = gxy[ti]
                b_m = torch.full_like(ti, b)
                cls_m = cls_ids[ti]
                gt_m = obj_indices[ti]

                gxi = torch.tensor(
                    [gw, gh], device=DEVICE, dtype=torch.float32
                ) - gxy_m

                j = (gxy_m % 1.0 < 0.5) & (gxy_m > 1.0)
                l = (gxi % 1.0 < 0.5) & (gxi > 1.0)

                keep = torch.stack([
                    torch.ones_like(j[:, 0]),
                    j[:, 0],
                    j[:, 1],
                    l[:, 0],
                    l[:, 1],
                ])

                m = ti.numel()

                def rep(x):
                    return x.unsqueeze(0).expand(5, *x.shape)[keep]

                gt_r = rep(gt_m)
                cls_r = rep(cls_m)
                b_r = rep(b_m)
                a_r = rep(ai)

                offsets = torch.tensor(
                    [
                        [0, 0],
                        [1, 0],
                        [0, 1],
                        [-1, 0],
                        [0, -1],
                    ],
                    device=DEVICE,
                    dtype=torch.float32,
                )[:, None, :].expand(5, m, 2)[keep] * 0.5

                gij = (gxy_m.unsqueeze(0).expand(
                    5, m, 2
                )[keep] - offsets).long()

                gi = gij[:, 0].clamp(0, gw - 1)
                gj = gij[:, 1].clamp(0, gh - 1)

                for k in range(gt_r.numel()):
                    key = (
                        int(b_r[k]),
                        int(a_r[k]),
                        int(gj[k]),
                        int(gi[k]),
                    )

                    collisions[si][key].append(
                        (int(gt_r[k]), int(cls_r[k]))
                    )

    print("\n" + "=" * 75)
    print("TARGET COLLISION DIAGNOSTIC")
    print("=" * 75)

    for si, name in enumerate(["P3", "P4", "P5"]):
        slots = collisions[si]

        total = sum(len(v) for v in slots.values())
        unique = len(slots)
        collided = sum(1 for v in slots.values() if len(v) > 1)

        print(f"\n{name}:")
        print(f"  Total positive assignments: {total}")
        print(f"  Unique anchor-grid slots: {unique}")
        print(f"  Collided slots: {collided}")
        print(f"  Collision rate: {collided / max(unique, 1) * 100:.4f}%")

        class_collisions = defaultdict(int)

        for values in slots.values():
            if len(values) > 1:
                for _, cls in values:
                    class_collisions[cls] += 1

        print("  Classes involved in collisions:")
        for cls in sorted(class_collisions):
            print(
                f"    {CLASS_NAMES[cls]:16s}: "
                f"{class_collisions[cls]}"
            )

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
