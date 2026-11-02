
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ModelConfig
from datasets import OpenLaneDataset

cfg = ModelConfig()
dataset = OpenLaneDataset(
    r"D:\MinorProjectData\OpenLane",
    cfg,
    r"D:\MinorProjectData\lane3d_300\training",
)

indices = [0, len(dataset) // 3, (2 * len(dataset)) // 3]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, idx in zip(axes, indices):
    image, target = dataset[idx]

    # Reverse the project's normalization for display.
    img = image.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.clip(img * std + mean, 0, 1)

    ax.imshow(img)

    grid = target["lane_targets"].cpu().numpy()
    for lane_idx, row_idx in zip(*np.where(grid > 0)):
        col = grid[lane_idx, row_idx] - 1
        x = (col + 0.5) * cfg.image_w / cfg.grid_w
        y = (row_idx + 0.5) * cfg.image_h / cfg.grid_h
        ax.scatter(x, y, s=10)

    ax.set_title(f"Sample {idx}: {int((grid > 0).sum())} target points")
    ax.set_xlim(0, cfg.image_w)
    ax.set_ylim(cfg.image_h, 0)
    ax.axis("off")

plt.tight_layout()
output = Path("lane_target_visualization.png")
plt.savefig(output, dpi=140, bbox_inches="tight")
print("Saved:", output.resolve())
plt.show()
