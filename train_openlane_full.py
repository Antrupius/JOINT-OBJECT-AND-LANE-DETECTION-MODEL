"""
Full OpenLane lane-head training for the joint detector.

Safety guarantees:
- Does NOT modify the source object-detection checkpoint.
- Freezes the YOLO object-detection head.
- Optimizes only the shared backbone and lane_head.
- Writes all new checkpoints into a separate output directory.

Run from the project root:
python train_openlane_full.py --init-ckpt "experiments\\class_aware_conservative\\bdd_epoch10_backup.pt"
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ModelConfig, DataConfig
from models.detector import JointDetector
from datasets import make_dataset, collate_fn
from losses import ufld_focal_softbin_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_compatible_weights(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = extract_state_dict(ckpt)
    own = model.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in own and hasattr(v, "shape") and own[k].shape == v.shape
    }
    missing = sorted(set(own) - set(compatible))
    unexpected = sorted(set(state) - set(own))
    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(own)} compatible tensors from: {path}")
    if missing:
        print(f"Not initialized from source checkpoint: {len(missing)} tensors "
              f"(expected lane-head-specific/new tensors among them).")
    if unexpected:
        print(f"Ignored {len(unexpected)} source tensors not present in this model.")
    return ckpt


def lane_metrics(logits, targets):
    # Cell-level occupancy metrics (not official OpenLane benchmark metrics).
    pred = logits.argmax(dim=-1) > 0
    gt = targets > 0
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return tp, fp, fn, precision, recall, f1


def run_epoch(model, loader, optimizer, scaler, device, train, args, epoch):
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    totals = np.zeros(3, dtype=np.float64)  # TP, FP, FN

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=f"{'Train' if train else 'Val'} {epoch}", leave=True)
        for batch in pbar:
            images, targets = batch

            images = images.to(device, non_blocking=True)
            lane_targets = targets["lane_targets"].to(device, non_blocking=True).long()
            has_lane = targets["has_lane_labels"].to(device).bool()

            # Skip batches without usable lane annotations.
            if not has_lane.any():
                continue

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                output = model(images)
                logits = output["lane_logits"]
                loss = ufld_focal_softbin_loss(
                    logits[has_lane],
                    lane_targets[has_lane],
                    gamma=args.gamma,
                    smoothing=args.smoothing,
                    foreground_weight=args.foreground_weight,
                )

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            n_batches += 1
            with torch.no_grad():
                tp, fp, fn, _, _, _ = lane_metrics(
                    logits[has_lane].detach(), lane_targets[has_lane]
                )
                totals += (tp, fp, fn)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    tp, fp, fn = totals
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return avg_loss, precision, recall, f1


def save_checkpoint(path, model, optimizer, scaler, epoch, val_loss, best_val, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val_loss": best_val,
        "args": vars(args),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-ckpt", required=True,
                        help="Working object-detection checkpoint used to initialize the joint model.")
    parser.add_argument("--resume-ckpt", default="runs/openlane_full/latest_resume.pt")
    parser.add_argument("--out-dir", default="runs/openlane_full")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--foreground-weight", type=float, default=4.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    mc = ModelConfig()
    dc = DataConfig()
    dc.openlane_dir = r"D:\\MinorProjectData\\OpenLane"
    dc.openlane_annotations_dir = r"D:\\MinorProjectData\\lane3d_300"
    dc.num_workers = args.workers

    train_ds = make_dataset("openlane", mc, dc, "train")
    val_ds = make_dataset("openlane", mc, dc, "val")
    print(f"OpenLane: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn, drop_last=False,
    )

    model = JointDetector(mc).to(device)

    # Load compatible source weights; lane-head weights are expected to be absent
    # or shape-incompatible and will remain freshly initialized.
    load_compatible_weights(model, args.init_ckpt, device)

    # Explicitly freeze the complete object-detection head.
    for p in model.obj_head.parameters():
        p.requires_grad = False

    # Optimize ONLY the shared backbone and lane head.
    trainable = list(model.backbone.parameters()) + list(model.lane_head.parameters())
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.lane_head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val = float("inf")
    resume_path = Path(args.resume_ckpt)

    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume requested, but not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", float("inf")))
        print(f"Resuming at epoch {start_epoch}; best val loss={best_val:.6f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, tr_p, tr_r, tr_f1 = run_epoch(
            model, train_loader, optimizer, scaler, device, True, args, epoch
        )
        val_loss, va_p, va_r, va_f1 = run_epoch(
            model, val_loader, optimizer, scaler, device, False, args, epoch
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.5f} P/R/F1={tr_p:.4f}/{tr_r:.4f}/{tr_f1:.4f} | "
            f"val loss={val_loss:.5f} P/R/F1={va_p:.4f}/{va_r:.4f}/{va_f1:.4f}"
        )

        # Always preserve the most recent state for continuation.
        save_checkpoint(
            out_dir / "latest_resume.pt", model, optimizer, scaler,
            epoch, val_loss, best_val, args
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "best_lane_validation.pt", model, optimizer, scaler,
                epoch, val_loss, best_val, args
            )
            print(f"Saved new best lane checkpoint (val loss={best_val:.6f})")


if __name__ == "__main__":
    main()
