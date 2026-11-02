"""Lane-detection head and evidential uncertainty utilities."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import ConvBnAct


def softplus_evidence(logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(logits)


def dirichlet_stats(logits: torch.Tensor, num_classes: int):
    """Evidential deep learning quantities from class logits.
    Returns expected probs, vacuity (epistemic-style), aleatoric (expected entropy)."""
    ev = softplus_evidence(logits)                     # evidence >= 0
    alpha = 1.0 + ev                                   # Dirichlet params
    S = alpha.sum(dim=1, keepdim=True)                 # Dirichlet strength
    prob = alpha / S                                   # expected class prob
    vacuity = num_classes / S.squeeze(1)               # u = K / S
    ent = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1)
    return prob, vacuity, ent, alpha, ev


class LaneHead(nn.Module):
    """Row-wise lane classification that preserves horizontal position."""

    def __init__(self, in_dims, num_lanes: int, n_offsets: int,
                 feat_h: int, feat_w: int):
        super().__init__()
        self.L = num_lanes
        self.O = n_offsets
        self.O1 = n_offsets + 1
        self.H = feat_h
        self.W = feat_w

        self.encoders = nn.ModuleList([
            nn.Sequential(ConvBnAct(d, 128, 1, 1))
            for d in in_dims
        ])
        self.merge = nn.Sequential(
            ConvBnAct(128 * len(in_dims), 128, 3, 1),
            ConvBnAct(128, 128, 3, 1),
        )

        # Classify each complete row while preserving its horizontal features.
        self.row_classifier = nn.Linear(
            128 * self.W, self.L * self.O1
        )

    def forward(self, feats):
        ups = [
            F.interpolate(
                enc(feat),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            )
            for feat, enc in zip(feats, self.encoders)
        ]

        x = self.merge(torch.cat(ups, dim=1))  # (B, 128, H, W)
        b = x.shape[0]

        # Keep all horizontal positions; do not average across width.
        rows = x.permute(0, 2, 1, 3).reshape(b, self.H, -1)
        logits = self.row_classifier(rows)
        logits = logits.reshape(b, self.H, self.L, self.O1)
        logits = logits.permute(0, 2, 1, 3).contiguous()

        probs = F.softmax(logits, dim=-1)
        p_exist = 1.0 - probs[..., 0]

        _, vacuity, alea, _, _ = dirichlet_stats(
            logits.reshape(-1, self.O1), self.O1
        )

        return {
            "lane_logits": logits,
            "p_exist": p_exist,
            "vacuity_lane": vacuity.reshape(b, self.L, self.H),
            "alea_lane": alea.reshape(b, self.L, self.H),
        }

    def existence(self, out: dict, thr: float = 0.5):
        return (out["p_exist"] > thr).float()
