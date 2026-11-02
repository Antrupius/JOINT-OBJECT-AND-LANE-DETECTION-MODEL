"""One-epoch BDD100K head-only training using explicit repeated CSV exposures.

- Freezes RepViT backbone and lane head; trains only YOLO object head.
- CSV must contain a column named `filename`, with BDD annotation filenames
  (e.g. `abc.jpg.json`) repeated according to the class-aware plan.
- Resume checkpoint records the cursor into the shuffled CSV exposure list.
- Original training scripts are not modified.
"""
import argparse
import csv
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import ModelConfig, DataConfig, TrainConfig
from datasets import make_dataset, collate_fn
from models.detector import JointDetector
from losses import total_loss
from validation_utils import validate_loss
from train import set_seed, _move_batch
from main import load_checkpoint


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_exposure_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError("Sampling CSV must have a 'filename' column.")
        names = [row["filename"].strip() for row in reader if row.get("filename", "").strip()]
    if not names:
        raise ValueError("Sampling CSV contains no exposures.")
    digest = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    return names, digest


def map_csv_exposures_to_dataset(train_ds, exposure_names):
    # BDD100KDataset.ids contains image names; annotation path is name + ".json".
    name_to_index = {name + ".json": i for i, name in enumerate(train_ds.ids)}
    mapped = []
    missing = []
    for name in exposure_names:
        # Support CSV values that may be image names instead of annotation names.
        key = name if name.endswith(".json") else name + ".json"
        idx = name_to_index.get(key)
        if idx is None:
            missing.append(name)
        else:
            mapped.append(idx)
    if missing:
        sample = missing[:8]
        raise ValueError(
            f"{len(missing)} CSV filename(s) did not match BDD dataset IDs. "
            f"Examples: {sample}. Check that CSV filenames correspond to train/ann."
        )
    return mapped


def save_resume(path, model, optimizer, scaler, state):
    payload = dict(state)
    payload.update({
        "checkpoint_type": "bdd100k_csv_class_aware_head_only_v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    })
    atomic_save(payload, path)


def save_best(path, model, val_loss, images_seen):
    atomic_save({
        "model_state_dict": model.state_dict(),
        "val_loss": float(val_loss),
        "images_seen": int(images_seen),
        "selection": "lowest loss on fixed validation subset",
        "experiment": "bdd100k_csv_class_aware_head_only",
    }, path)


def main():
    p = argparse.ArgumentParser(description="One epoch of CSV-driven class-aware BDD head-only training.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--sampling-csv", required=True)
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--best-ckpt", default="experiments/bdd_csv_class_aware_head_only/best.pt")
    p.add_argument("--resume-ckpt", default="experiments/bdd_csv_class_aware_head_only/resume.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--save-every-images", type=int, default=500)
    p.add_argument("--validate-every-images", type=int, default=1000)
    p.add_argument("--validation-images", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if min(args.batch_size, args.save_every_images, args.validate_every_images, args.validation_images) < 1:
        p.error("batch size and intervals must be positive.")
    if args.resume and args.init_ckpt:
        p.error("Use --resume OR --init-ckpt, not both.")
    resume_path = Path(args.resume_ckpt)
    if args.resume and not resume_path.is_file():
        p.error(f"Resume checkpoint not found: {resume_path}")
    if not args.resume and resume_path.exists():
        p.error(f"Resume checkpoint already exists: {resume_path}. Use --resume or move it aside.")
    if not args.resume and not args.init_ckpt:
        p.error("Provide --init-ckpt for a fresh run, or --resume.")

    set_seed(args.seed)
    mc, dc, tc = ModelConfig(), DataConfig(), TrainConfig()
    dc.dataset_name = "bdd100k"
    dc.bdd_dir = str(args.data_dir)
    dc.num_workers = 0
    tc.batch_size = args.batch_size
    tc.lr = args.lr
    tc.device = "cuda" if torch.cuda.is_available() else "cpu"
    tc.num_classes = mc.num_classes
    tc.image_h, tc.image_w = mc.image_h, mc.image_w
    amp = tc.device.startswith("cuda")

    print(f"Device: {tc.device}")
    print("Experiment: CSV CLASS-AWARE / OBJECT-HEAD-ONLY / ONE EPOCH")
    print(f"LR={args.lr}; batch={args.batch_size}; AMP={amp}")

    train_ds = make_dataset("bdd100k", mc, dc, split="train")
    val_ds = make_dataset("bdd100k", mc, dc, split="validation")
    if not len(train_ds) or not len(val_ds):
        raise RuntimeError("BDD train and validation datasets must be non-empty.")

    exposure_names, csv_hash = read_exposure_csv(args.sampling_csv)
    exposure_indices = map_csv_exposures_to_dataset(train_ds, exposure_names)
    total_exposures = len(exposure_indices)
    print(f"Unique training images: {len(train_ds):,}")
    print(f"CSV exposures (this one epoch): {total_exposures:,}")
    print(f"Extra repeat exposures: {total_exposures - len(train_ds):,}")
    print(f"Sampling CSV SHA256: {csv_hash}")

    val_n = min(args.validation_images, len(val_ds))
    val_ids = random.Random(args.seed + 9001).sample(range(len(val_ds)), val_n)
    common = dict(batch_size=args.batch_size, collate_fn=collate_fn,
                 num_workers=0, pin_memory=amp)
    val_loader_small = DataLoader(Subset(val_ds, val_ids), shuffle=False, **common)
    val_loader_full = DataLoader(val_ds, shuffle=False, **common)

    model = JointDetector(mc).to(tc.device)
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.lane_head.parameters():
        param.requires_grad = False
    for param in model.obj_head.parameters():
        param.requires_grad = True
    trainable = [x for x in model.parameters() if x.requires_grad]
    print(f"Trainable object-head params: {sum(x.numel() for x in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=tc.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_path = Path(args.best_ckpt)
    start_pos = 0
    images_seen = 0
    best_val = float("inf")
    last_val_at = 0

    if args.resume:
        state = torch_load(resume_path)
        if state.get("checkpoint_type") != "bdd100k_csv_class_aware_head_only_v1":
            raise ValueError("Checkpoint type mismatch; not this CSV class-aware experiment.")
        if state.get("sampling_csv_sha256") != csv_hash:
            raise ValueError("Sampling CSV differs from the one saved in the resume checkpoint.")
        if int(state.get("total_exposures", -1)) != total_exposures:
            raise ValueError("Exposure count mismatch in resume checkpoint.")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        scaler.load_state_dict(state.get("scaler_state_dict", {}))
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_state_all"]])
        start_pos = int(state["exposure_cursor"])
        images_seen = int(state["images_seen"])
        best_val = float(state["best_val"])
        last_val_at = int(state.get("last_val_at", 0))
        print(f"Resuming at exposure cursor {start_pos:,}/{total_exposures:,}; images_seen={images_seen:,}")
    else:
        load_checkpoint(model, str(args.init_ckpt), tc.device)
        print(f"Initialized weights from: {args.init_ckpt}")

    def cursor_state(pos):
        return {
            "exposure_cursor": pos,
            "images_seen": images_seen,
            "best_val": best_val,
            "last_val_at": last_val_at,
            "total_exposures": len(exposure_indices),
            "sampling_csv": str(Path(args.sampling_csv).resolve()),
            "sampling_csv_sha256": csv_hash,
            "seed": args.seed,
            "lr": args.lr,
            "experiment": "bdd100k_csv_class_aware_head_only",
        }

    if start_pos >= total_exposures:
        print("Resume checkpoint indicates the CSV epoch is already complete.")
        return

    # Build batches from the exact CSV sequence. Repeated CSV filenames become
    # repeated dataset indices; no implicit random sampler is applied.
    model.train()
    tc._epoch = 0
    next_save_at = ((images_seen // args.save_every_images) + 1) * args.save_every_images
    next_val_at = ((images_seen // args.validate_every_images) + 1) * args.validate_every_images

    pos = start_pos
    while pos < total_exposures:
        end = min(pos + args.batch_size, total_exposures)
        batch_indices = exposure_indices[pos:end]
        loader = DataLoader(Subset(train_ds, batch_indices), shuffle=False, **common)
        for images, batch in loader:
            images = images.to(tc.device, non_blocking=amp)
            batch = _move_batch(batch, tc.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                outputs = model(images)
                loss = total_loss(outputs, batch, tc)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite loss at CSV exposure cursor {pos}")
            if not loss.requires_grad:
                raise RuntimeError("Loss has no gradient graph.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n = int(images.shape[0])
            pos += n
            images_seen += n

            if images_seen >= next_val_at:
                val = validate_loss(model, val_loader_small, tc)
                improved = val < best_val
                if improved:
                    best_val = val
                    save_best(best_path, model, val, images_seen)
                print(f"[Validation] exposures={pos:,}/{total_exposures:,} val_loss={val:.5f} best={best_val:.5f} improved={improved}")
                last_val_at = images_seen
                next_val_at += args.validate_every_images
                save_resume(resume_path, model, optimizer, scaler, cursor_state(pos))

            if images_seen >= next_save_at:
                save_resume(resume_path, model, optimizer, scaler, cursor_state(pos))
                print(f"[Resume checkpoint] saved at {images_seen:,} exposures (CSV cursor {pos:,}).")
                next_save_at += args.save_every_images
        del loader

    if images_seen != last_val_at:
        val = validate_loss(model, val_loader_small, tc)
        improved = val < best_val
        if improved:
            best_val = val
            save_best(best_path, model, val, images_seen)
        print(f"[Final fixed-subset validation: {val:.5f}; best={best_val:.5f}; improved={improved}")
        last_val_at = images_seen

    full_val = validate_loss(model, val_loader_full, tc)
    save_resume(resume_path, model, optimizer, scaler, cursor_state(total_exposures))
    print(f"[ONE CSV EPOCH COMPLETE] exposures={total_exposures:,}; full validation loss={full_val:.5f}")
    print(f"Latest/resume checkpoint: {resume_path}")
    print(f"Best fixed-subset checkpoint: {best_path}")


if __name__ == "__main__":
    main()
