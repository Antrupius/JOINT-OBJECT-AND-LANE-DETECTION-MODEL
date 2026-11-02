
"""
KITTI Object Detection Fine-Tuning
----------------------------------
Initializes from the BDD100K Epoch 10 checkpoint.

Trainable:
    - Shared backbone (RepViT)
    - Object detection head (YOLOv5-style)

Frozen:
    - Lane detection head

KITTI classification supervision:
    Classes 0-5 enabled
    Classes 6-7 masked

Original BDD checkpoints are never overwritten.
"""

import os
import random
import shutil
from datetime import datetime
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent


# ============================================================
# STEP 1: SAFELY PATCH LOSS FUNCTIONS
# ============================================================

def backup_file(path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(
        path.name + f".backup_kitti_{timestamp}"
    )
    shutil.copy2(path, backup)
    print(f"Backup created: {backup}")


def patch_yolo_loss():

    path = ROOT / "models" / "yolov5_detection.py"

    text = path.read_text(encoding="utf-8")

    if "class_mask=None" in text:
        print("YOLO loss already supports class_mask.")
        return

    old_signature = (
        "return_components=False, class_weights=None):"
    )

    new_signature = (
        "return_components=False, class_weights=None, "
        "class_mask=None):"
    )

    old_block = '''            if class_weights is None:
                cls_bce = F.binary_cross_entropy_with_logits(
                    ps[:, 5:], tcls
                )
            else:
                positive_weights = class_weights.to(
                    device=device, dtype=ps.dtype
                )
                cls_bce = F.binary_cross_entropy_with_logits(
                    ps[:, 5:], tcls, pos_weight=positive_weights
                )
            lcls = lcls + cls_bce'''

    # The uploaded current file may have formatting with a compact call.
    old_block_compact = '''            if class_weights is None:
                cls_bce = F.binary_cross_entropy_with_logits(ps[:, 5:], tcls)
            else:
                positive_weights = class_weights.to(
                    device=device, dtype=ps.dtype
                )
                cls_bce = F.binary_cross_entropy_with_logits(
                    ps[:, 5:], tcls, pos_weight=positive_weights
                )
            lcls = lcls + cls_bce'''

    new_block = '''            if class_weights is None:
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

            lcls = lcls + cls_bce'''

    if old_signature not in text:
        raise RuntimeError(
            "Unexpected YOLO loss signature. "
            "No patch was applied. Inspect the function manually."
        )

    if old_block in text:
        matched_block = old_block
    elif old_block_compact in text:
        matched_block = old_block_compact
    else:
        raise RuntimeError(
            "Could not find the expected class BCE block. "
            "No patch was applied."
        )

    backup_file(path)

    text = text.replace(
        old_signature,
        new_signature,
        1
    )

    text = text.replace(
        matched_block,
        new_block,
        1
    )

    path.write_text(text, encoding="utf-8")

    print("YOLO loss patched successfully.")


def patch_total_loss():

    path = ROOT / "losses.py"

    text = path.read_text(encoding="utf-8")

    if (
        "def total_loss(out: dict, batch: dict, cfg, class_mask=None)"
        in text
    ):
        print("total_loss already supports class_mask.")
        return

    old_signature = (
        "def total_loss(out: dict, batch: dict, cfg) "
        "-> torch.Tensor:"
    )

    new_signature = (
        "def total_loss(out: dict, batch: dict, cfg, "
        "class_mask=None) -> torch.Tensor:"
    )

    old_call = '''            class_weights=getattr(cfg, "class_weights", None),
        )'''

    new_call = '''            class_weights=getattr(cfg, "class_weights", None),
            class_mask=class_mask,
        )'''

    if old_signature not in text:
        raise RuntimeError(
            "Unexpected total_loss signature. "
            "No patch applied."
        )

    if old_call not in text:
        raise RuntimeError(
            "Could not locate YOLO loss call in losses.py. "
            "No patch applied."
        )

    backup_file(path)

    text = text.replace(
        old_signature,
        new_signature,
        1
    )

    text = text.replace(
        old_call,
        new_call,
        1
    )

    path.write_text(text, encoding="utf-8")

    print("total_loss patched successfully.")


# ============================================================
# STEP 2: TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kitti-dir",
        default=r"D:\MinorProjectData\KITTI\training"
    )

    parser.add_argument(
        "--init-ckpt",
        required=True,
        help="Path to BDD100K Epoch 10 checkpoint"
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument(
        "--out-dir",
        default="experiments/kitti_finetune"
    )

    args = parser.parse_args()

    # Patch only after confirming expected source structure.
    patch_yolo_loss()
    patch_total_loss()

    # Import after patching.
    from config import ModelConfig, TrainConfig, DataConfig
    from datasets import make_dataset, collate_fn
    from models.detector import JointDetector
    from main import load_checkpoint
    from losses import total_loss

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    mc = ModelConfig()
    tc = TrainConfig()
    dc = DataConfig()

    dc.dataset_name = "kitti"
    dc.kitti_dir = args.kitti_dir

    tc.device = device
    tc.batch_size = args.batch_size
    tc.epochs = args.epochs
    tc.lr = args.lr

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = make_dataset(
        "kitti",
        mc,
        dc,
        split="train"
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "KITTI dataset is empty. Check --kitti-dir."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False
    )

    # --------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------

    model = JointDetector(mc).to(device)

    load_checkpoint(
        model,
        args.init_ckpt,
        device
    )

    print(
        f"\nInitialized from BDD checkpoint:\n"
        f"{args.init_ckpt}"
    )

    # --------------------------------------------------------
    # Freeze lane head
    # --------------------------------------------------------

    for parameter in model.lane_head.parameters():
        parameter.requires_grad = False

    for parameter in model.backbone.parameters():
        parameter.requires_grad = True

    for parameter in model.obj_head.parameters():
        parameter.requires_grad = True

    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=getattr(
            tc, "weight_decay", 1e-4
        )
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # --------------------------------------------------------
    # KITTI class mask
    # --------------------------------------------------------

    # Class IDs:
    # 0 Car
    # 1 Truck
    # 2 Bus
    # 3 Pedestrian
    # 4 Cyclist
    # 5 Motorcycle
    # 6 Traffic light -> masked
    # 7 Traffic sign  -> masked

    class_mask = torch.tensor(
        [1, 1, 1, 1, 1, 1, 0, 0],
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_dir = ROOT / args.out_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    latest_checkpoint = (
        output_dir / "kitti_finetune_latest.pt"
    )

    global_step = 0

    print("\n========== KITTI FINE-TUNING ==========")
    print(f"Device          : {device}")
    print(f"Training images : {len(dataset)}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Learning rate  : {args.lr}")
    print("Trainable      : Backbone + Object Head")
    print("Frozen         : Lane Head")
    print(f"Class mask     : {class_mask.tolist()}")
    print(f"Output folder  : {output_dir}")
    print("====================================\n")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        model.train()

        # Prevent frozen lane head from updating running statistics.
        model.lane_head.eval()

        total_loss_value = 0.0
        total_seen = 0

        for images, batch in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                outputs = model(images)

                loss = total_loss(
                    outputs,
                    batch,
                    tc,
                    class_mask=class_mask
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at "
                    f"epoch {epoch}, step {global_step}"
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=10.0
            )

            scaler.step(optimizer)
            scaler.update()

            batch_size_actual = images.shape[0]

            total_loss_value += (
                float(loss.detach()) * batch_size_actual
            )

            total_seen += batch_size_actual

            global_step += 1

            if global_step % 100 == 0:

                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Step {global_step} | "
                    f"Images {total_seen}/{len(dataset)} | "
                    f"Loss {loss.item():.5f}"
                )

        # ----------------------------------------------------
        # Epoch summary
        # ----------------------------------------------------

        mean_loss = (
            total_loss_value / max(total_seen, 1)
        )

        print("\n------------------------------------")
        print(f"Epoch {epoch} completed")
        print(f"Images processed : {total_seen}")
        print(f"Mean loss       : {mean_loss:.6f}")
        print("------------------------------------")

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "epoch_loss": mean_loss,

            "source_checkpoint":
                str(args.init_ckpt),

            "dataset": "KITTI",

            "class_mask":
                class_mask.cpu().tolist(),

            "frozen_module": "lane_head"
        }

        epoch_checkpoint = (
            output_dir /
            f"kitti_finetune_epoch{epoch}.pt"
        )

        torch.save(
            checkpoint,
            epoch_checkpoint
        )

        torch.save(
            checkpoint,
            latest_checkpoint
        )

        print(f"Saved epoch checkpoint: {epoch_checkpoint}")
        print(f"Updated latest checkpoint: {latest_checkpoint}")

    print("\nKITTI fine-tuning completed successfully.")
    print("Your original BDD checkpoint remains untouched.")


if __name__ == "__main__":
    main()
