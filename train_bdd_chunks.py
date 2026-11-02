"""Resumable BDD100K chunk trainer.

Runs a configurable number of 7,000-image chunks per invocation. The latest
checkpoint stores model, optimizer, AMP scaler, RNG states, and the exact
training cursor so the next invocation can continue with the next chunk.
A separate best checkpoint is selected using a fixed validation subset.
"""
import argparse
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


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _save_resume(path, model, optimizer, scaler, state):
    payload = dict(state)
    payload.update({
        "checkpoint_type": "bdd100k_resumable_training_state",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    })
    _atomic_torch_save(payload, path)


def _load_resume(path, model, optimizer, scaler, device, lr):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError(f"Not a resumable checkpoint: {path}")
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])

    # Explicitly enforce the learning rate supplied through CLI.
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print(f"[Resume checkpoint] Learning rate set to {lr}")
    scaler.load_state_dict(state.get("scaler_state_dict", {}))
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(
            [rng_state.cpu() for rng_state in state["cuda_rng_state_all"]]
        )
    return state


def _save_best(path, model, val_loss, epoch, images_seen):
    payload = {
        "model_state_dict": model.state_dict(),
        "val_loss": float(val_loss),
        "epoch": int(epoch) + 1,
        "images_seen": int(images_seen),
        "selection": "lowest loss on the fixed validation subset",
    }
    _atomic_torch_save(payload, path)


def _validate_and_maybe_save(model, loader, cfg, best_path, best_loss,
                             epoch, images_seen):
    value = validate_loss(model, loader, cfg)
    improved = value < best_loss
    if improved:
        _save_best(best_path, model, value, epoch, images_seen)
        best_loss = value
    print(
        f"[Validation checkpoint] images_seen={images_seen} "
        f"subset_loss={value:.5f} best={best_loss:.5f} "
        f"best_updated={improved}",
        flush=True,
    )
    return best_loss, value


def run_chunked_training(
    data_dir,
    best_path,
    resume_path,
    init_ckpt=None,
    epochs=2,
    chunk_size=7000,
    batch_size=2,
    lr=1e-4,
    save_every_images=500,
    validate_every_images=1000,
    validation_images=500,
    chunks_per_run=1,
    resume=False,
    seed=42,
):
    if min(epochs, chunk_size, batch_size, save_every_images,
           validate_every_images, validation_images, chunks_per_run) < 1:
        raise ValueError("All counts and intervals must be positive.")

    best_path = Path(best_path)
    resume_path = Path(resume_path)
    if resume and not resume_path.is_file():
        raise FileNotFoundError(f"Cannot resume; checkpoint not found: {resume_path}")
    if not resume and resume_path.exists():
        raise FileExistsError(
            f"Resume file already exists: {resume_path}\n"
            "Use --resume to continue it. To intentionally restart, move or "
            "rename that resume file first."
        )

    set_seed(seed)
    mc = ModelConfig()
    dc = DataConfig()
    tc = TrainConfig()
    dc.dataset_name = "bdd100k"
    dc.bdd_dir = str(data_dir)
    dc.num_workers = 0
    tc.epochs = epochs
    tc.batch_size = batch_size
    tc.lr = lr
    tc.chunk_size = chunk_size
    tc.device = "cuda" if torch.cuda.is_available() else "cpu"
    tc.num_classes = mc.num_classes
    tc.image_h, tc.image_w = mc.image_h, mc.image_w

    print(f"Device: {tc.device}", flush=True)
    print("Loading BDD100K TEST dataset...", flush=True)
    train_ds = make_dataset("bdd100k", mc, dc, split="test")
    val_ds = train_ds
    if not len(train_ds) or not len(val_ds):
        raise RuntimeError("BDD100K training and validation sets must be non-empty.")

    # Use a fixed, seeded validation subset for frequent best-checkpoint
    # selection. Full validation loss is also reported at each epoch boundary.
    val_n = min(validation_images, len(val_ds))
    val_indices = random.Random(seed + 9001).sample(range(len(val_ds)), val_n)
    val_subset = Subset(val_ds, val_indices)
    amp_enabled = tc.device.startswith("cuda")
    common = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": 0,
        "pin_memory": amp_enabled,
    }
    val_loader_small = DataLoader(val_subset, shuffle=False, **common)
    val_loader_full = DataLoader(val_ds, shuffle=False, **common)

    model = JointDetector(mc).to(tc.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if resume:
        state = _load_resume(resume_path, model, optimizer, scaler, tc.device, lr)
        epoch_idx = int(state["epoch_idx"])
        chunk_idx = int(state["chunk_idx"])
        offset_in_chunk = int(state["offset_in_chunk"])
        images_seen_total = int(state["images_seen_total"])
        best_val = float(state["best_val"])
        last_validation_count = int(state.get("last_validation_count", 0))
        print(
            f"Resumed at epoch {epoch_idx + 1}, chunk {chunk_idx + 1}, "
            f"offset {offset_in_chunk}; images_seen={images_seen_total}; "
            f"best_subset_loss={best_val:.5f}",
            flush=True,
        )
    else:
        if init_ckpt:
            load_checkpoint(model, str(init_ckpt), tc.device)
            print(f"Initialized from existing checkpoint: {init_ckpt}", flush=True)
        else:
            print("No initial checkpoint supplied; training from initialized weights.", flush=True)
        epoch_idx, chunk_idx, offset_in_chunk = 0, 0, 0
        images_seen_total, best_val, last_validation_count = 0, float("inf"), 0

    def cursor_state():
        return {
            "epoch_idx": epoch_idx,
            "chunk_idx": chunk_idx,
            "offset_in_chunk": offset_in_chunk,
            "images_seen_total": images_seen_total,
            "best_val": best_val,
            "last_validation_count": last_validation_count,
            "epochs_target": epochs,
            "chunk_size": chunk_size,
            "train_size": len(train_ds),
            "seed": seed,
        }

    if epoch_idx >= epochs:
        print("Training is already complete according to the resume checkpoint.")
        print(f"Best model: {best_path}")
        return

    chunks_completed_this_run = 0
    while epoch_idx < epochs and chunks_completed_this_run < chunks_per_run:
        # A fresh deterministic permutation for each epoch; it is reproducible
        # when resuming because it depends only on the configured seed and epoch.
        order_gen = torch.Generator().manual_seed(seed + epoch_idx)
        order = torch.randperm(len(train_ds), generator=order_gen).tolist()
        num_chunks = (len(order) + chunk_size - 1) // chunk_size

        if chunk_idx >= num_chunks:
            # Defensive recovery for a cursor saved exactly at epoch boundary.
            full_value = validate_loss(model, val_loader_full, tc)
            print(f"Epoch {epoch_idx + 1} full validation loss: {full_value:.5f}", flush=True)
            epoch_idx += 1
            chunk_idx, offset_in_chunk = 0, 0
            _save_resume(resume_path, model, optimizer, scaler, cursor_state())
            continue

        chunk_indices = order[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
        if offset_in_chunk > len(chunk_indices):
            raise ValueError("Resume cursor is beyond the current chunk.")
        remaining_indices = chunk_indices[offset_in_chunk:]
        chunk_number_for_log = chunk_idx + 1
        print(
            f"\n=== Epoch {epoch_idx + 1}/{epochs} | "
            f"chunk {chunk_number_for_log}/{num_chunks} | "
            f"remaining images in chunk: {len(remaining_indices)} ===",
            flush=True,
        )

        chunk_loader = DataLoader(
            Subset(train_ds, remaining_indices),
            shuffle=False,
            **common,
        )
        model.train()
        tc._epoch = epoch_idx

        for images, batch in chunk_loader:
            images = images.to(tc.device, non_blocking=amp_enabled)
            batch = _move_batch(batch, tc.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                outputs = model(images)
                loss = total_loss(outputs, batch, tc)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch_idx + 1}, chunk {chunk_idx + 1}, "
                    f"offset {offset_in_chunk}"
                )
            if not loss.requires_grad:
                raise RuntimeError("Training loss has no gradient graph.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_n = int(images.shape[0])
            offset_in_chunk += batch_n
            images_seen_total += batch_n

            if images_seen_total - last_validation_count >= validate_every_images:
                best_val, _ = _validate_and_maybe_save(
                    model, val_loader_small, tc, best_path, best_val,
                    epoch_idx, images_seen_total,
                )
                last_validation_count = images_seen_total
                _save_resume(resume_path, model, optimizer, scaler, cursor_state())

            if images_seen_total % save_every_images == 0:
                _save_resume(resume_path, model, optimizer, scaler, cursor_state())
                print(
                    f"[Resume checkpoint] saved after {images_seen_total} training images.",
                    flush=True,
                )

        del chunk_loader

        # Evaluate the fixed validation subset at the chunk boundary if the
        # interval did not already trigger on the final batch.
        if images_seen_total != last_validation_count:
            best_val, _ = _validate_and_maybe_save(
                model, val_loader_small, tc, best_path, best_val,
                epoch_idx, images_seen_total,
            )
            last_validation_count = images_seen_total

        chunk_idx += 1
        offset_in_chunk = 0
        _save_resume(resume_path, model, optimizer, scaler, cursor_state())
        chunks_completed_this_run += 1
        print(
            f"[Chunk complete] epoch={epoch_idx + 1}/{epochs}, "
            f"completed chunk={chunk_number_for_log}/{num_chunks}, "
            f"images_seen_total={images_seen_total}",
            flush=True,
        )

        if chunk_idx >= num_chunks:
            full_value = validate_loss(model, val_loader_full, tc)
            print(
                f"\n[Epoch complete] {epoch_idx + 1}/{epochs}; "
                f"train images this epoch={len(train_ds)}; "
                f"full validation loss={full_value:.5f}; "
                f"best fixed-subset validation loss={best_val:.5f}",
                flush=True,
            )
            epoch_idx += 1
            chunk_idx, offset_in_chunk = 0, 0
            _save_resume(resume_path, model, optimizer, scaler, cursor_state())

    print(
        f"\nThis run completed {chunks_completed_this_run} chunk(s). "
        f"Next cursor: epoch {min(epoch_idx + 1, epochs)}/{epochs}, "
        f"chunk {chunk_idx + 1}.",
        flush=True,
    )
    print(f"Best model checkpoint: {best_path}", flush=True)
    print(f"Resume checkpoint: {resume_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train BDD100K in resumable 7,000-image chunks."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--best-ckpt", default="checkpoints/bdd100k_full_best.pt")
    parser.add_argument("--resume-ckpt", default="checkpoints/bdd100k_full_resume.pt")
    parser.add_argument("--init-ckpt", default="checkpoints/bdd100k_7000_1000_final_audit_v1.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=7000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunks-per-run", type=int, default=1)
    parser.add_argument("--save-every-images", type=int, default=500)
    parser.add_argument("--validate-every-images", type=int, default=1000)
    parser.add_argument("--validation-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.resume:
        init_ckpt = None
    else:
        init_ckpt = args.init_ckpt
        if init_ckpt and not Path(init_ckpt).is_file():
            parser.error(f"Initial checkpoint not found: {init_ckpt}")

    run_chunked_training(
        data_dir=args.data_dir,
        best_path=args.best_ckpt,
        resume_path=args.resume_ckpt,
        init_ckpt=init_ckpt,
        epochs=args.epochs,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every_images=args.save_every_images,
        validate_every_images=args.validate_every_images,
        validation_images=args.validation_images,
        chunks_per_run=args.chunks_per_run,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
