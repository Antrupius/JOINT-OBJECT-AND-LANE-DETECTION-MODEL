"""Small OpenLane lane-only fine-tuning test for the current joint model.

The OpenLane dataset parser/target generation is imported unchanged from datasets.py.
Only the backbone and lane head are optimized; the object head is frozen.
Default run is deliberately small: 256 training images, 64 validation images,
100 optimizer steps, batch size 2. This is a smoke/quality check, not final training.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import ModelConfig, DataConfig, TrainConfig
from datasets import make_dataset, collate_fn
from models.detector import JointDetector
from losses import ufld_focal_softbin_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lane_metrics(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    probs, truths = [], []
    with torch.inference_mode():
        for x, batch in loader:
            x = x.to(device, non_blocking=True)
            y = batch["lane_targets"].to(device).long()
            amp = str(device).startswith("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)["lane_logits"]
            p = logits.float().softmax(-1)
            pred = p.argmax(-1)
            gt_fg, pred_fg = y > 0, pred > 0
            tp += int((gt_fg & pred_fg).sum())
            fp += int(((~gt_fg) & pred_fg).sum())
            fn += int((gt_fg & (~pred_fg)).sum())
            probs.append((1 - p[..., 0]).cpu().flatten())
            truths.append(gt_fg.cpu().flatten())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    result = {"precision": precision, "recall": recall, "f1": f1,
              "tp": tp, "fp": fp, "fn": fn}
    try:
        from sklearn.metrics import roc_auc_score
        result["auroc"] = float(roc_auc_score(torch.cat(truths).numpy(), torch.cat(probs).numpy()))
    except Exception:
        result["auroc"] = None
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=r"D:\MinorProjectData\OpenLane")
    ap.add_argument("--annotations-root", default=r"D:\MinorProjectData\lane3d_300")
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0, help="Keep 0 for a conservative Windows run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="runs/openlane_small_test")
    ap.add_argument("--init-ckpt", default=None, help="Optional compatible joint-model state dict")
    args = ap.parse_args()
    if min(args.n_train, args.n_val, args.steps, args.batch_size) < 1:
        ap.error("n-train, n-val, steps and batch-size must be positive")

    seed_all(args.seed)
    cfg, dcfg = ModelConfig(), DataConfig()
    tcfg = TrainConfig()
    device = tcfg.device
    dcfg.dataset_name = "openlane"
    dcfg.openlane_dir = args.data_root
    dcfg.openlane_annotations_dir = args.annotations_root
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; AMP: {str(device).startswith('cuda')}")
    print("Loading OpenLane using the project's existing dataset parser...")
    train_ds = make_dataset("openlane", cfg, dcfg, split="train")
    val_ds = make_dataset("openlane", cfg, dcfg, split="validation")
    if len(train_ds) < 1 or len(val_ds) < 1:
        raise RuntimeError(f"OpenLane splits empty: train={len(train_ds)}, val={len(val_ds)}")
    n_train, n_val = min(args.n_train, len(train_ds)), min(args.n_val, len(val_ds))
    gen = torch.Generator().manual_seed(args.seed)
    train_ids = torch.randperm(len(train_ds), generator=gen)[:n_train].tolist()
    val_ids = torch.arange(n_val).tolist()
    pin = str(device).startswith("cuda")
    workers = max(0, args.workers)
    common = {"batch_size": args.batch_size, "collate_fn": collate_fn,
              "num_workers": workers, "pin_memory": pin,
              "persistent_workers": workers > 0}
    if workers > 0:
        common["prefetch_factor"] = 2
    train_loader = DataLoader(Subset(train_ds, train_ids), shuffle=True, **common)
    val_loader = DataLoader(Subset(val_ds, val_ids), shuffle=False, **common)
    print(f"Subset: train={n_train}, validation={n_val}, optimizer_steps={args.steps}")

    model = JointDetector(cfg).to(device)
    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        print(f"Loaded compatible checkpoint: {args.init_ckpt}")
    for p in model.obj_head.parameters():
        p.requires_grad_(False)
    params = [p for p in model.backbone.parameters() if p.requires_grad] + list(model.lane_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=pin)
    iterator = iter(train_loader)
    model.train()
    running = 0.0
    for step in range(1, args.steps + 1):
        try:
            x, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, target = next(iterator)
        x = x.to(device, non_blocking=True)
        y = target["lane_targets"].to(device).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=pin):
            logits = model(x)["lane_logits"]
            loss = ufld_focal_softbin_loss(logits, y, gamma=2.0, smoothing=0.1, foreground_weight=4.0)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite lane loss at step {step}: {loss.detach().item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach())
        if step % 20 == 0 or step == args.steps:
            print(f"step {step}/{args.steps} | mean recent lane loss={running / min(step % 20 or 20, step):.4f}")
            running = 0.0

    metrics = lane_metrics(model, val_loader, device)
    print("OpenLane subset metrics (cell-level; not official lane benchmark metrics):")
    print(json.dumps(metrics, indent=2))
    ckpt_path = out_dir / "openlane_subset_joint_checkpoint.pt"
    torch.save({"model": model.state_dict(), "metrics": metrics,
                "config": vars(cfg), "train_subset": n_train, "val_subset": n_val,
                "steps": args.steps}, ckpt_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {ckpt_path.resolve()}")
    print(f"Saved metrics: {(out_dir / 'metrics.json').resolve()}")


if __name__ == "__main__":
    main()
