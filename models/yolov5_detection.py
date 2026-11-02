"""YOLOv5-inspired multi-scale detection branch for the existing RepViT backbone.

This is an original compatible implementation of the YOLOv5 design pattern (PAN/FPN
neck, anchor-based multi-scale detection, objectness + class logits), not a drop-in
copy of Ultralytics source and not pretrained YOLOv5 weights.

Changes versus the previous version (state_dict keys are UNCHANGED, so old
checkpoints still load):
  * Target assignment is now YOLOv5-style: every GT is matched to *all* anchors
    whose size ratio is < anchor_t, on its own cell plus the two nearest
    neighbouring cells.  Previously exactly ONE (scale, anchor, cell) per GT was
    positive, which gave ~5-9x fewer positives than YOLOv5 and left objectness
    sitting near the "everything is background" prior.
  * Objectness target is the (detached) CIoU of the matched prediction instead of
    a hard 1.0, and the per-scale objectness losses use the YOLOv5 balance
    weights (4.0, 1.0, 0.4) and are summed over scales, not averaged.
  * Detection-head biases are initialised like YOLOv5 (low prior objectness,
    uniform class prior) so training does not start from sigmoid(0)=0.5.
  * Loss is vectorised: no per-GT Python loop, no .item() GPU syncs.
  * Loss maths is done in float32 (pixel coordinates up to 640 lose ~0.5px of
    precision in fp16 under autocast).
  * decode() takes a pre-NMS top-k so a very low score threshold (needed for
    mAP) stays cheap.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from models.backbone import ConvBnAct


# Anchors in pixels (P3/8, P4/16, P5/32), adapted from common YOLOv5 anchors to
# a 640x384 input.  Single source of truth for the head AND the loss.
DEFAULT_ANCHORS = torch.tensor([
    [[10., 8.], [16., 18.], [33., 14.]],
    [[30., 37.], [62., 27.], [59., 71.]],
    [[116., 54.], [156., 119.], [373., 196.]],
])


class Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.cv1 = ConvBnAct(channels, hidden, 1, 1)
        self.cv2 = ConvBnAct(hidden, channels, 3, 1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class C3Lite(nn.Module):
    def __init__(self, in_ch, out_ch, repeats=1):
        super().__init__()
        hidden = max(out_ch // 2, 8)
        self.a = ConvBnAct(in_ch, hidden, 1, 1)
        self.b = ConvBnAct(in_ch, hidden, 1, 1)
        self.blocks = nn.Sequential(*[Bottleneck(hidden) for _ in range(repeats)])
        self.out = ConvBnAct(hidden * 2, out_ch, 1, 1)

    def forward(self, x):
        return self.out(torch.cat((self.blocks(self.a(x)), self.b(x)), dim=1))


class YOLOv5StyleDetectionHead(nn.Module):
    """Three-scale anchor-based detector with YOLOv5-style decode and class-aware NMS."""
    def __init__(self, in_dims, num_classes, image_h=384, image_w=640):
        super().__init__()
        if len(in_dims) != 3:
            raise ValueError(f"Expected three backbone scales, got {in_dims}")
        self.nc = int(num_classes)
        self.no = self.nc + 5
        self.image_h, self.image_w = int(image_h), int(image_w)
        c3, c4, c5 = in_dims
        # FPN top-down path.
        self.p5_reduce = ConvBnAct(c5, c4, 1, 1)
        self.p4_fuse = C3Lite(c4 + c4, c4, 2)
        self.p4_reduce = ConvBnAct(c4, c3, 1, 1)
        self.p3_fuse = C3Lite(c3 + c3, c3, 2)
        # PAN bottom-up path.
        self.p3_down = ConvBnAct(c3, c3, 3, 2)
        self.p4_pan = C3Lite(c3 + c4, c4, 2)
        self.p4_down = ConvBnAct(c4, c4, 3, 2)
        self.p5_pan = C3Lite(c4 + c5, c5, 2)
        self.detect = nn.ModuleList([
            nn.Conv2d(c3, 3 * self.no, 1),
            nn.Conv2d(c4, 3 * self.no, 1),
            nn.Conv2d(c5, 3 * self.no, 1),
        ])
        self.register_buffer("anchors", DEFAULT_ANCHORS.clone())
        self.strides = (8, 16, 32)
        self._init_detect_biases()

    def _init_detect_biases(self):
        """YOLOv5 bias init: low prior objectness (~8 objects/image), uniform class prior."""
        with torch.no_grad():
            for conv, s in zip(self.detect, self.strides):
                b = conv.bias.view(3, self.no)          # view -> edits conv.bias in place
                cells = (self.image_w / s) * (self.image_h / s)
                b[:, 4] += math.log(8.0 / cells)
                b[:, 5:] += math.log(0.6 / (self.nc - 0.99))

    def forward(self, feats):
        f3, f4, f5 = feats
        p5 = self.p5_reduce(f5)
        p4 = self.p4_fuse(torch.cat((f4, F.interpolate(p5, size=f4.shape[-2:], mode='nearest')), 1))
        p4r = self.p4_reduce(p4)
        p3 = self.p3_fuse(torch.cat((f3, F.interpolate(p4r, size=f3.shape[-2:], mode='nearest')), 1))
        p4out = self.p4_pan(torch.cat((self.p3_down(p3), p4), 1))
        p5out = self.p5_pan(torch.cat((self.p4_down(p4out), f5), 1))
        pyramid = (p3, p4out, p5out)
        raw = []
        for i, (feature, conv) in enumerate(zip(pyramid, self.detect)):
            b, _, h, w = feature.shape
            raw.append(conv(feature).view(b, 3, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous())
        return raw

    def decode(self, raw, score_thr=0.25, iou_thr=0.45, max_dets=50, pre_nms_topk=3000):
        """Rows: (x1, y1, x2, y2, class, score, 0, 0) in input-pixel coordinates.

        score = sigmoid(obj) * max_c sigmoid(cls_c).  Use score_thr ~0.001 and
        max_dets ~300 when computing mAP; 0.25 is a deployment operating point.
        """
        device = raw[0].device
        batch = raw[0].shape[0]
        all_boxes, all_scores, all_classes = [], [], []
        for scale, pred in enumerate(raw):
            pred = pred.float()
            _, _, h, w, _ = pred.shape
            yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            grid = torch.stack((xx, yy), dim=-1).view(1, 1, h, w, 2).float()
            stride = pred.new_tensor([self.image_w / w, self.image_h / h])
            xy = (pred[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) * stride
            wh = (pred[..., 2:4].sigmoid() * 2.0).pow(2) * self.anchors[scale].to(pred.dtype).view(1, 3, 1, 1, 2)
            obj = pred[..., 4].sigmoid()
            cls = pred[..., 5:].sigmoid()
            scores, classes = (obj.unsqueeze(-1) * cls).max(dim=-1)
            xyxy = torch.cat((xy - wh / 2, xy + wh / 2), dim=-1)
            all_boxes.append(xyxy.reshape(batch, -1, 4))
            all_scores.append(scores.reshape(batch, -1))
            all_classes.append(classes.reshape(batch, -1))
        boxes_all = torch.cat(all_boxes, dim=1)
        boxes_all = torch.stack((boxes_all[..., 0].clamp(0, self.image_w),
                                 boxes_all[..., 1].clamp(0, self.image_h),
                                 boxes_all[..., 2].clamp(0, self.image_w),
                                 boxes_all[..., 3].clamp(0, self.image_h)), dim=-1)
        scores_all = torch.cat(all_scores, dim=1)
        classes_all = torch.cat(all_classes, dim=1)

        results = []
        for b in range(batch):
            mask = scores_all[b] >= score_thr
            boxes, sc, cl = boxes_all[b][mask], scores_all[b][mask], classes_all[b][mask]
            if sc.numel() == 0:
                results.append(torch.zeros((0, 8), device=device))
                continue
            if sc.numel() > pre_nms_topk:
                top = sc.topk(pre_nms_topk).indices
                boxes, sc, cl = boxes[top], sc[top], cl[top]
            keep = batched_nms(boxes, sc, cl, iou_thr)[:max_dets]
            # Preserve the existing 8-column inference contract. The final two
            # columns are reserved but unavailable for this sigmoid detector;
            # do not disguise low confidence as epistemic/aleatoric uncertainty.
            unavailable = torch.zeros_like(sc[keep])
            results.append(torch.cat((boxes[keep], cl[keep, None].float(), sc[keep, None],
                                      unavailable[:, None], unavailable[:, None]), dim=1))
        return results


# ----------------------------------------------------------------------------
# Target assignment + loss
# ----------------------------------------------------------------------------
def _empty_targets(device):
    z = torch.zeros(0, dtype=torch.long, device=device)
    zf = torch.zeros(0, dtype=torch.float32, device=device)

    return (
        z, z, z, z,
        torch.zeros(0, 2, dtype=torch.float32, device=device),
        torch.zeros(0, 2, dtype=torch.float32, device=device),
        z, zf
    )

@torch.no_grad()
def build_targets(boxes, labels, n_obj, grid_shapes, image_h, image_w, anchors, anchor_t=4.0):
    """YOLOv5-style assignment.

    boxes  : (B, M, 4) float, [x1, y1, x2, y2] in INPUT PIXELS (what datasets.py produces)
    labels : (B, M) int class ids;  n_obj : (B,) number of valid boxes per image
    grid_shapes : [(gh, gw), ...] per scale;  anchors : (S, 3, 2) pixels

    Returns one tuple per scale: (b, a, gj, gi, gxy, gwh, cls) where gxy/gwh are the
    GT centre / size in GRID units of that scale.  A GT is positive for every anchor
    with max(w/aw, aw/w, h/ah, ah/h) < anchor_t, in its own cell and in the two
    nearest neighbouring cells.
    """
    device = boxes.device
    anchors = anchors.to(device=device, dtype=torch.float32)
    boxes = boxes.float()
    B, M = boxes.shape[:2]
    if M == 0:
        return [_empty_targets(device) for _ in grid_shapes]
    valid = torch.arange(M, device=device)[None, :] < n_obj.to(device)[:, None]
    bi, mi = valid.nonzero(as_tuple=True)
    if bi.numel() == 0:
        return [_empty_targets(device) for _ in grid_shapes]
    tb = boxes[bi, mi]
    tc = labels.to(device)[bi, mi].long()
    w = (tb[:, 2] - tb[:, 0]).clamp_min(1.0)
    h = (tb[:, 3] - tb[:, 1]).clamp_min(1.0)
    cx = (tb[:, 0] + tb[:, 2]) / 2
    cy = (tb[:, 1] + tb[:, 3]) / 2
    offs = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=device, dtype=torch.float32) * 0.5

    out = []
    for si, (gh, gw) in enumerate(grid_shapes):
        scale = torch.tensor([image_w / gw, image_h / gh], device=device, dtype=torch.float32)
        gxy = torch.stack((cx, cy), dim=1) / scale
        gwh = torch.stack((w, h), dim=1) / scale
        a_grid = anchors[si] / scale                                   # (3, 2) in grid units
        r = gwh[:, None, :] / a_grid[None, :, :]                      # (n, 3, 2)
        match = torch.maximum(r, 1.0 / r).amax(dim=2) < anchor_t      # (n, 3)
        ti, ai = match.nonzero(as_tuple=True)
        # Anchor-fit quality: closer to 1.0 means a better anchor match.
        match_quality = torch.maximum(r, 1.0 / r).amax(dim=2)
        quality_m = match_quality[ti, ai]
        if ti.numel() == 0:
            out.append(_empty_targets(device))
            continue
        gxy_m, gwh_m, b_m, c_m = gxy[ti], gwh[ti], bi[ti], tc[ti]
        m = ti.numel()
        gxi = torch.tensor([gw, gh], device=device, dtype=torch.float32) - gxy_m
        j = (gxy_m % 1.0 < 0.5) & (gxy_m > 1.0)                        # may use left / upper neighbour
        l = (gxi % 1.0 < 0.5) & (gxi > 1.0)                            # may use right / lower neighbour
        keep = torch.stack((torch.ones_like(j[:, 0]), j[:, 0], j[:, 1], l[:, 0], l[:, 1]))  # (5, m)

        def rep(x):
            return x.unsqueeze(0).expand(5, *x.shape)[keep]

        gxy_r = rep(gxy_m)
        gwh_r = rep(gwh_m)
        b_r = rep(b_m)
        a_r = rep(ai)
        c_r = rep(c_m)
        quality_r = rep(quality_m)
        offsets = offs[:, None, :].expand(5, m, 2)[keep]
        gij = (gxy_r - offsets).long()
        gi = gij[:, 0].clamp(0, gw - 1)
        gj = gij[:, 1].clamp(0, gh - 1)
        out.append((b_r, a_r, gj, gi, gxy_r, gwh_r, c_r, quality_r))
    return out


def yolo_object_loss(raw, batch, image_h=384, image_w=640, box_weight=0.05, obj_weight=1.0,
                     cls_weight=None, anchors=None, anchor_t=4.0, balance=(4.0, 1.0, 0.4),
                     return_components=False, class_weights=None, class_mask=None):
    """YOLOv5-style loss: multi-positive assignment, CIoU box loss, IoU-aware objectness, BCE classes.

    Defaults follow YOLOv5 hyper-parameters (box 0.05, obj 1.0, cls 0.5*nc/80*3/nl).
    Returns a scalar tensor (or (tensor, dict) with return_components=True).
    """
    from torchvision.ops import complete_box_iou_loss
    device = raw[0].device
    nc = raw[0].shape[-1] - 5
    nl = len(raw)
    if cls_weight is None:
        cls_weight = 0.5 * nc / 80.0 * 3.0 / nl
    anchors = (DEFAULT_ANCHORS if anchors is None else anchors).to(device=device, dtype=torch.float32)
    bal = tuple(balance) if nl == len(balance) else (1.0,) * nl
    grid_shapes = [tuple(p.shape[2:4]) for p in raw]
    targets = build_targets(batch['boxes'].to(device), batch['labels'].to(device),
                            batch['n_obj'].to(device), grid_shapes, image_h, image_w,
                            anchors, anchor_t)

    lbox = raw[0].new_zeros((), dtype=torch.float32)
    lobj = raw[0].new_zeros((), dtype=torch.float32)
    lcls = raw[0].new_zeros((), dtype=torch.float32)
    n_pos = 0
    for si, (pred, tg) in enumerate(zip(raw, targets)):
        pred = pred.float()
        na, gh, gw = pred.shape[1], pred.shape[2], pred.shape[3]
        b, a, gj, gi, gxy, gwh, c, quality = tg
        tobj = torch.zeros(pred.shape[:4], device=device, dtype=torch.float32)
        if b.numel():
            n_pos += int(b.numel())
            
            # --------------------------------------------------
            # Resolve anchor-grid collisions for box/class loss
            # --------------------------------------------------
            # Slot ID uniquely identifies:
            # (batch, anchor, grid_y, grid_x)
            # --------------------------------------------------
            # Resolve anchor-grid collisions for box/class loss
            # --------------------------------------------------
            # Slot ID uniquely identifies:
            # (batch, anchor, grid_y, grid_x)
            flat = ((b * na + a) * gh + gj) * gw + gi

            # Sort by quality first, then stably group by prediction slot.
            # Within each slot, the best-quality assignment comes first.
            order = torch.argsort(quality, stable=True)
            order = order[torch.argsort(flat[order], stable=True)]

            flat_sorted = flat[order]

            # Keep exactly one GT per occupied prediction slot.
            first = torch.ones_like(flat_sorted, dtype=torch.bool)
            first[1:] = flat_sorted[1:] != flat_sorted[:-1]

            keep_idx = order[first]

            # Filter box/class targets.
            # Objectness continues using ALL original assignments below.
            b_box = b[keep_idx]
            a_box = a[keep_idx]
            gj_box = gj[keep_idx]
            gi_box = gi[keep_idx]

            gxy_box = gxy[keep_idx]
            gwh_box = gwh[keep_idx]
            c_box = c[keep_idx]

            ps = pred[b_box, a_box, gj_box, gi_box]
                                                # (K, no)

            scale = torch.tensor(
                [image_w / gw, image_h / gh],
                device=device,
                dtype=torch.float32,
            )
            a_grid = anchors[si] / scale

            pxy = (
                ps[:, 0:2].sigmoid() * 2.0 - 0.5
                + torch.stack((gi_box, gj_box), dim=1).float()
            )

            pwh = (ps[:, 2:4].sigmoid() * 2.0).pow(2) * a_grid[a_box]

            pbox = torch.cat(
                (pxy - pwh / 2, pxy + pwh / 2),
                dim=1,
            )

            tbox = torch.cat(
                (gxy_box - gwh_box / 2, gxy_box + gwh_box / 2),
                dim=1,
            )

            closs = complete_box_iou_loss(
                pbox, tbox, reduction="none"
            )

            lbox = lbox + closs.mean()

            
            # Objectness IoU from ALL original assignments
            ps_all = pred[b, a, gj, gi]

            scale = torch.tensor(
                [image_w / gw, image_h / gh],
                device=device,
                dtype=torch.float32,
            )

            a_grid = anchors[si] / scale

            pxy_all = (
                ps_all[:, 0:2].sigmoid() * 2.0 - 0.5
                + torch.stack((gi, gj), dim=1).float()
            )

            pwh_all = (
                (ps_all[:, 2:4].sigmoid() * 2.0).pow(2)
                * a_grid[a]
            )

            pbox_all = torch.cat(
                (
                    pxy_all - pwh_all / 2,
                    pxy_all + pwh_all / 2,
                ),
                dim=1,
            )

            tbox_all = torch.cat(
                (gxy - gwh / 2, gxy + gwh / 2),
                dim=1,
            )

            iou_all = (
                1.0 - complete_box_iou_loss(
                    pbox_all, tbox_all, reduction="none"
                )
            ).detach().clamp(0.0, 1.0)

            flat_all = ((b * na + a) * gh + gj) * gw + gi

            tobj.view(-1).scatter_reduce_(
                0, flat_all, iou_all, reduce="amax", include_self=True
            )

            tcls = F.one_hot(c_box, num_classes=nc).to(ps.dtype)
            if class_weights is None:
                per_class_bce = F.binary_cross_entropy_with_logits(
                    ps[:, 5:], tcls, reduction="none"
                )
            else:
                positive_weights = class_weights.to(
                    device=device, dtype=ps.dtype
                )
                per_class_bce = F.binary_cross_entropy_with_logits(
                    ps[:, 5:], tcls,
                    pos_weight=positive_weights,
                    reduction="none"
                )

            if class_mask is not None:
                mask = class_mask.to(
                    device=device, dtype=ps.dtype
                ).view(1, -1)

                if mask.shape[-1] != nc:
                    raise ValueError(
                        f"class_mask must have {nc} entries, "
                        f"got {mask.shape[-1]}"
                    )

                cls_bce = (
                    (per_class_bce * mask).sum()
                    / mask.sum().clamp_min(1.0)
                )
            else:
                cls_bce = per_class_bce.mean()

            lcls = lcls + cls_bce
        lobj = lobj + bal[si] * F.binary_cross_entropy_with_logits(pred[..., 4], tobj)

    loss = box_weight * lbox + obj_weight * lobj + cls_weight * lcls
    if return_components:
        return loss, {
            "box": lbox.detach().item(),
            "obj": lobj.detach().item(),
            "cls": lcls.detach().item(),
            "n_pos": n_pos
        }
    return loss
