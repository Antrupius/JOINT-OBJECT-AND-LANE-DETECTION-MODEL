from dataclasses import dataclass, field
from typing import Tuple, Optional

import torch


@dataclass
class ModelConfig:
    backbone: str = "repvit"  # "repvit" | "mobilenetv3"
    image_h: int = 384
    image_w: int = 640
    num_classes: int = 8
    num_lanes: int = 4
    h_steps: int = 48
    n_offsets: int = 80
    repvit_depths: Tuple[int, ...] = (2, 2, 4, 2)
    repvit_dims: Tuple[int, ...] = (32, 64, 128, 256)
    width_mult: float = 1.0
    det_score_thr: float = 0.25
    det_nms_iou: float = 0.45
    max_dets: int = 50

    @property
    def grid_h(self) -> int:
        return self.image_h // 8

    @property
    def grid_w(self) -> int:
        return self.image_w // 8



@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 3
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    box_loss_weight: float = 0.05  # YOLO-style box localization loss weight
    # Positive-class weights for [car, truck, bus, pedestrian, cyclist,
    # motorcycle, traffic_light, traffic_sign]. These encourage the classifier
    # to learn less frequent classes; validate their effect on held-out data.
    class_weights: torch.Tensor = field(
        default_factory=lambda: torch.tensor(
            [1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0],
            dtype=torch.float32,
        )
    )
    # Process BDD100K in sequential subsets without resetting model/optimizer.
    chunk_size: int = 7000
    det_loss_weight: float = 4.0
    val_split: float = 0.2
    calib_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CalibConfig:
    alpha: float = 0.1
    n_bins_ece: int = 15
    ece_max_bin: float = 1.0
    tau_epistemic: float = 0.6
    cov_monitor_ema: float = 0.9
    cov_drift_thresh: float = 0.03



@dataclass
class DataConfig:
    root: str = "./data"
    dataset_name: str = "openlane"
    kitti_dir: Optional[str] = None
    bdd_dir: Optional[str] = None
    openlane_dir: Optional[str] = r"D:\MinorProjectData\OpenLane"
    openlane_annotations_dir: Optional[str] = r"D:\MinorProjectData\lane3d_300"
    synthetic_n: int = 2000
    num_workers: int = 2


@dataclass
class EvalConfig:
    iou_thresholds: Tuple[float, ...] = (0.5,)
    ig_steps: int = 50
    deletion_steps: int = 20
    blur_kernel: int = 25
    blur_sigma: float = 8.0
    latency_iters: int = 100