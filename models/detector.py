"""Joint detector with YOLOv5-style multi-scale object detection and preserved lane head."""
import torch
import torch.nn as nn

from models.backbone import build_backbone
from models.heads import LaneHead
from models.yolov5_detection import YOLOv5StyleDetectionHead
from config import ModelConfig


class JointDetector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg.backbone,
                                       img_h=cfg.image_h, width_mult=cfg.width_mult)
        dims = self.backbone.out_dims
        self.obj_head = YOLOv5StyleDetectionHead(dims, cfg.num_classes, cfg.image_h, cfg.image_w)
        self.lane_head = LaneHead(dims, cfg.num_lanes, cfg.n_offsets, cfg.grid_h, cfg.grid_w)


    def features(self, x: torch.Tensor):
        return self.backbone(x)                                # [f_s8, f_s16, f_s32]

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.features(x)
        det_raw = self.obj_head(feats)
        lane = self.lane_head(feats)
        # Compatibility aliases for existing analysis hooks; canonical training
        # and decoding use det_raw and the YOLOv5-style head.
        first = det_raw[0]
        return {
            "det_raw": det_raw,
            "obj_logit": first[..., 4],
            "cls_logits": first[..., 5:].permute(0, 1, 4, 2, 3).contiguous(),
            **lane,
        }

    def target_score(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar target for integrated gradients: max foreground class logit over the grid."""
        out = self.forward(x)
        return out["cls_logits"].max()

    def decode(self, out: dict, score_thr=None, iou_thr=None, max_dets=None):
        """Defaults are the deployment operating point from ModelConfig.
        For mAP pass score_thr=0.001, iou_thr=0.6, max_dets=300."""
        return self.obj_head.decode(
            out["det_raw"],
            self.cfg.det_score_thr if score_thr is None else score_thr,
            self.cfg.det_nms_iou if iou_thr is None else iou_thr,
            self.cfg.max_dets if max_dets is None else max_dets,
        )
