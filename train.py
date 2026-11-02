"""Direct training utilities for the joint perception model."""
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from config import TrainConfig, DataConfig, ModelConfig
from datasets import make_dataset, collate_fn
from models.detector import JointDetector
from losses import total_loss
from validation_utils import validate_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch, device):
    non_blocking = str(device).startswith("cuda")
    return {key: value.to(device, non_blocking=non_blocking)
            if torch.is_tensor(value) else value for key, value in batch.items()}


def train_epoch(model, loader, optimizer, cfg: TrainConfig, epoch: int = 0, scaler=None, chunk_label=None):
    """Train over one loader; optionally reuse a GradScaler across chunks."""
    model.train()
    cfg._epoch = epoch
    total, batches, samples_seen = 0.0, 0, 0
    amp_enabled = str(cfg.device).startswith("cuda") and torch.cuda.is_available()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_samples = len(loader.dataset)
    next_report = 10000
    label = f" | {chunk_label}" if chunk_label else ""
    print(f"\n[Joint] Starting epoch {epoch + 1}{label} | Total training images: {total_samples}", flush=True)

    for images, batch in loader:
        images = images.to(cfg.device, non_blocking=amp_enabled)
        batch = _move_batch(batch, cfg.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            outputs = model(images)
            loss = total_loss(outputs, batch, cfg)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {loss.detach().item()}")
        if not loss.requires_grad:
            raise RuntimeError("Training loss has no gradient graph.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * images.size(0)
        batches += 1
        samples_seen += images.size(0)
        while samples_seen >= next_report:
            print(f"[Joint] Epoch {epoch + 1}: {min(next_report, total_samples)}/{total_samples} images processed", flush=True)
            next_report += 10000

    if samples_seen < total_samples:
        print(f"[Joint] Epoch {epoch + 1}: {samples_seen}/{total_samples} images processed", flush=True)
    if batches == 0:
        raise RuntimeError("No training batches were produced.")
    print(f"[Joint] Epoch {epoch + 1} completed: {samples_seen}/{total_samples} images", flush=True)
    return total / samples_seen


def _make_loaders(dcfg, mcfg, tcfg):
    """Create explicit train and validation loaders."""
    if dcfg.dataset_name not in ("bdd100k", "openlane"):
        raise ValueError(f"{dcfg.dataset_name!r} does not expose a separate validation dataset through make_dataset(). Use BDD100K or OpenLane.")
    train_ds = make_dataset(dcfg.dataset_name, mcfg, dcfg, split="train")
    val_ds = make_dataset(dcfg.dataset_name, mcfg, dcfg, split="validation")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Training and validation datasets must be non-empty.")
    generator = torch.Generator().manual_seed(tcfg.seed)
    workers = max(0, int(dcfg.num_workers))
    common = {"num_workers": workers, "pin_memory": str(tcfg.device).startswith("cuda"), "persistent_workers": workers > 0}
    if workers > 0:
        common["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, generator=generator, collate_fn=collate_fn, **common)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, collate_fn=collate_fn, **common)
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    return train_loader, val_loader



def _save_best(model, path, best_loss, val_loss, epoch=None):
    """Atomically save improved model weights and training metadata."""
    if not math.isfinite(val_loss):
        raise FloatingPointError(
            f"Cannot save checkpoint for non-finite validation loss: {val_loss}"
        )

    if val_loss < best_loss:
        temp_path = f"{path}.tmp"
        try:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": float(val_loss),
            }
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return val_loss, True

    return best_loss, False




def _restore_best_checkpoint(model, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Best checkpoint was not found: {path}")

    device = next(model.parameters()).device
    checkpoint = torch.load(
        path, map_location=device, weights_only=True
    )

    # Support both the new checkpoint dictionary and older state_dict files.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def fit_joint_model(dcfg: DataConfig, mcfg: ModelConfig, tcfg: TrainConfig, tag: str = "joint_model") -> JointDetector:
    """Train one joint model directly from its own task losses."""
    from datasets import SyntheticLaneObjectDataset
    model = JointDetector(mcfg).to(tcfg.device)
    tcfg.num_classes = mcfg.num_classes
    tcfg.image_h, tcfg.image_w = mcfg.image_h, mcfg.image_w
    if dcfg.dataset_name == "synthetic":
        full = SyntheticLaneObjectDataset(dcfg.synthetic_n, mcfg, seed=tcfg.seed)
        n_val = max(1, min(len(full) - 1, int(tcfg.val_split * len(full))))
        n_train = len(full) - n_val
        if n_train < 1:
            raise ValueError("Synthetic training requires at least two samples.")
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(tcfg.seed))
        train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    else:
        train_loader, val_loader = _make_loaders(dcfg, mcfg, tcfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    best_val = float("inf")
    best_path = f"{tag}.pt"
    if os.path.exists(best_path):
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {best_path}. Choose a new output name.")
    amp_enabled = str(tcfg.device).startswith("cuda") and torch.cuda.is_available()
    # Keep one scaler for the complete run, including all chunks of an epoch.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(max(1, int(tcfg.epochs))):
        if dcfg.dataset_name == "bdd100k":
            # BDD100K is streamed in sequential 7,000-image chunks. The model,
            # optimizer, and AMP scaler persist across every chunk; validation
            # runs only after the complete training set has been processed.
            train_ds = train_loader.dataset
            chunk_size = max(1, int(getattr(tcfg, "chunk_size", 7000)))
            generator = torch.Generator().manual_seed(int(tcfg.seed) + epoch)
            order = torch.randperm(len(train_ds), generator=generator).tolist()
            weighted_loss, samples_seen = 0.0, 0
            num_chunks = (len(order) + chunk_size - 1) // chunk_size

            for chunk_number, start in enumerate(range(0, len(order), chunk_size), start=1):
                chunk_indices = order[start:start + chunk_size]
                chunk_ds = Subset(train_ds, chunk_indices)
                chunk_loader = DataLoader(
                    chunk_ds,
                    batch_size=tcfg.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=0,
                    pin_memory=amp_enabled,
                )
                print(
                    f"\n[Joint] Epoch {epoch + 1}/{tcfg.epochs}: "
                    f"chunk {chunk_number}/{num_chunks} ({len(chunk_ds)} images)",
                    flush=True,
                )
                chunk_loss = train_epoch(
                    model, chunk_loader, optimizer, tcfg,
                    epoch=epoch, scaler=scaler,
                    chunk_label=f"chunk {chunk_number}/{num_chunks}",
                )
                weighted_loss += chunk_loss * len(chunk_ds)
                samples_seen += len(chunk_ds)
                del chunk_loader, chunk_ds

            if samples_seen != len(train_ds):
                raise RuntimeError(
                    f"Epoch processed {samples_seen} images; expected {len(train_ds)}."
                )
            train_value = weighted_loss / samples_seen
        else:
            train_value = train_epoch(
                model, train_loader, optimizer, tcfg,
                epoch=epoch, scaler=scaler,
            )

        val_value = validate_loss(model, val_loader, tcfg)
        best_val, saved = _save_best(
            model, best_path, best_val, val_value, epoch=epoch + 1
        )
        print(
            f"[{tag}] epoch {epoch + 1}/{tcfg.epochs} "
            f"train_loss={train_value:.4f} val_loss={val_value:.4f} "
            f"best_checkpoint_saved={saved}",
            flush=True,
        )
    return _restore_best_checkpoint(model, best_path)
