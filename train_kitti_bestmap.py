import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# KITTI class mask
# BDD class order:
# car, truck, bus, pedestrian, cyclist, motorcycle,
# traffic_light, traffic_sign
# ============================================================

CLASS_MASK = torch.tensor(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
)


# ============================================================
# Checkpoint utilities
# ============================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]

    return checkpoint


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    state = extract_state_dict(checkpoint)

    if not isinstance(state, dict):
        raise TypeError(
            f"Checkpoint does not contain a valid state dictionary: "
            f"{checkpoint_path}"
        )

    model.load_state_dict(state, strict=True)

    return checkpoint


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    val_map,
    best_map,
    patience_counter,
    args
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),

            "val_map": val_map,
            "best_map": best_map,
            "patience_counter": patience_counter,

            "class_mask": CLASS_MASK.tolist(),
            "seed": args.seed,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
        path
    )


# ============================================================
# BDD100K mAP@0.5 evaluation
#
# Uses the SAME deterministic 1000-image validation subset
# and evaluation protocol as evaluate_bdd_checkpoint.py.
# ============================================================

def evaluate_bdd_map(
    model,
    val_loader,
    device,
    score_threshold=0.0001,
    nms_iou=0.6,
    max_dets=300,
    match_iou=0.5
):
    from evaluate import iou, _average_precision
    from datasets import DRIVING_CLASSES

    model.eval()

    num_classes = len(DRIVING_CLASSES)

    all_scores = {c: [] for c in range(num_classes)}
    all_matches = {c: [] for c in range(num_classes)}
    total_gt = {c: 0 for c in range(num_classes)}

    with torch.inference_mode():

        for images, targets in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            predictions = model.decode(
                outputs,
                score_threshold,
                nms_iou,
                max_dets
            )

            batch_size = images.shape[0]

            for b in range(batch_size):

                pred = predictions[b].cpu().numpy()

                gt = targets

                n_obj = int(gt["n_obj"][b].item())

                gt_boxes = gt["boxes"][b, :n_obj].cpu().numpy()
                gt_labels = gt["labels"][b, :n_obj].cpu().numpy()

                # Count GT instances per class.
                for cls in range(num_classes):
                    total_gt[cls] += int(
                        np.sum(gt_labels == cls)
                    )

                # Evaluate predictions class-wise.
                for cls in range(num_classes):

                    cls_gt_boxes = gt_boxes[
                        gt_labels == cls
                    ]

                    cls_pred = (
                        pred[pred[:, 4].astype(int) == cls]
                        if len(pred)
                        else np.empty((0, 6))
                    )

                    # Highest confidence first.
                    cls_pred = sorted(
                        cls_pred,
                        key=lambda row: row[5],
                        reverse=True
                    )

                    matched_gt = set()

                    for det in cls_pred:

                        score = float(det[5])
                        box = det[:4]

                        all_scores[cls].append(score)

                        match = None

                        if len(cls_gt_boxes):

                            overlaps = iou(
                                box[None, :4],
                                cls_gt_boxes
                            )[0]

                            for rank in np.argsort(-overlaps):

                                if overlaps[rank] < match_iou:
                                    continue

                                if int(rank) not in matched_gt:
                                    match = int(rank)
                                    break

                        if match is not None:

                            matched_gt.add(match)
                            all_matches[cls].append(1.0)

                        else:

                            all_matches[cls].append(0.0)

    # AP for classes that have GT instances.
    class_aps = {}

    for cls in range(num_classes):

        if total_gt[cls] == 0:
            continue

        class_aps[cls] = _average_precision(
            all_scores[cls],
            all_matches[cls],
            total_gt[cls]
        )

    valid_aps = [
        ap for ap in class_aps.values()
        if np.isfinite(ap)
    ]

    mean_ap = (
        float(np.mean(valid_aps))
        if valid_aps
        else 0.0
    )

    return mean_ap, class_aps


# ============================================================
# Main training
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train object detection on KITTI while selecting "
            "checkpoints using fixed-subset BDD100K mAP@0.5."
        )
    )

    parser.add_argument(
        "--kitti-dir",
        type=str,
        default=r"D:\MinorProjectData\KITTI\training"
    )

    parser.add_argument(
        "--bdd-dir",
        type=str,
        default=r"D:\MinorProjectData\BDD100K"
    )

    parser.add_argument(
        "--init-ckpt",
        type=str,
        required=True
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"experiments\kitti_bestmap"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=6
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("KITTI TRAINING — BDD100K BEST mAP CHECKPOINT")
    print("=" * 70)

    print("Device:", device)
    print("Initial checkpoint:", args.init_ckpt)
    print("KITTI directory:", args.kitti_dir)
    print("BDD100K directory:", args.bdd_dir)
    print("Output directory:", args.out_dir)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print("Early stopping patience:", args.patience)

    os.makedirs(args.out_dir, exist_ok=True)

    best_path = os.path.join(
        args.out_dir,
        "best_map.pt"
    )

    latest_path = os.path.join(
        args.out_dir,
        "latest.pt"
    )

    # --------------------------------------------------------
    # Project imports
    # --------------------------------------------------------

    from config import (
        ModelConfig,
        TrainConfig,
        DataConfig
    )

    from datasets import (
        make_dataset,
        collate_fn
    )

    from models.detector import JointDetector

    from losses import total_loss

    mc = ModelConfig()
    dc = DataConfig()
    tc = TrainConfig()

    # --------------------------------------------------------
    # Configure datasets
    # --------------------------------------------------------

    dc.kitti_dir = args.kitti_dir
    dc.bdd_dir = args.bdd_dir

    # ========================================================
    # KITTI TRAINING DATASET
    # ========================================================

    print("\nLoading KITTI training dataset...")

    train_dataset = make_dataset(
        "kitti",
        mc,
        dc,
        split="training"
    )

    print("KITTI training images:", len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False
    )

    # ========================================================
    # BDD100K VALIDATION DATASET
    #
    # EXACT SAME selection logic as evaluate_bdd_checkpoint.py
    # ========================================================

    print("\nLoading BDD100K validation dataset...")

    bdd_dataset = make_dataset(
        "bdd100k",
        mc,
        dc,
        split="validation"
    )

    NUM_VALIDATION_IMAGES = 1000

    n_images = min(
        NUM_VALIDATION_IMAGES,
        len(bdd_dataset)
    )

    if n_images < NUM_VALIDATION_IMAGES:

        print(
            f"WARNING: Requested {NUM_VALIDATION_IMAGES} "
            f"BDD validation images, but only {n_images} exist."
        )

    if n_images == 0:
        raise RuntimeError(
            "BDD100K validation dataset is empty."
        )

    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(bdd_dataset),
        generator=generator
    )[:n_images].tolist()

    val_dataset = Subset(
        bdd_dataset,
        indices
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False
    )

    print(
        f"BDD100K validation subset: "
        f"{len(val_dataset)} / {len(bdd_dataset)} images"
    )

    print(
        "Validation subset selection: "
        "torch.randperm(len(dataset), "
        "generator=torch.Generator().manual_seed(42))[:1000]"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = JointDetector(mc).to(device)

    print("\nLoading initial BDD checkpoint...")

    load_checkpoint(
        model,
        args.init_ckpt,
        device
    )

    print("Initial checkpoint loaded successfully.")

    # --------------------------------------------------------
    # Freeze lane head.
    # Train shared backbone + object detection head.
    # --------------------------------------------------------

    for name, param in model.named_parameters():

        if (
            "lane_head" in name
            or "lane" in name.lower()
        ):

            param.requires_grad = False

        else:

            param.requires_grad = True

    if hasattr(model, "lane_head"):
        model.lane_head.eval()

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if not trainable_params:
        raise RuntimeError(
            "No trainable model parameters found."
        )

    print(
        "Trainable parameters:",
        sum(p.numel() for p in trainable_params)
    )

    print(
        "Frozen parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if not p.requires_grad
        )
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = GradScaler(
        enabled=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Resume or initialize
    # --------------------------------------------------------

    start_epoch = 0
    best_map = -1.0
    patience_counter = 0

    if args.resume:

        if not os.path.exists(latest_path):

            raise FileNotFoundError(
                "--resume was specified, but latest checkpoint "
                f"does not exist: {latest_path}"
            )

        print("\nResuming from latest checkpoint...")

        ckpt = torch.load(
            latest_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            ckpt["model_state_dict"],
            strict=True
        )

        if "optimizer_state_dict" in ckpt:

            optimizer.load_state_dict(
                ckpt["optimizer_state_dict"]
            )

        if "scaler_state_dict" in ckpt:

            scaler.load_state_dict(
                ckpt["scaler_state_dict"]
            )

        start_epoch = int(
            ckpt["epoch"]
        )

        best_map = float(
            ckpt.get("best_map", -1.0)
        )

        patience_counter = int(
            ckpt.get("patience_counter", 0)
        )

        print("Resuming after epoch:", start_epoch)
        print("Previous best BDD mAP@0.5:", best_map)

    # ========================================================
    # END OF PART 1
    # Continue directly with PART 2 below.
    # ========================================================

        # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs
    ):

        model.train()

        # Keep frozen lane head in evaluation mode.
        if hasattr(model, "lane_head"):
            model.lane_head.eval()

        running_loss = 0.0
        n_batches = 0

        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 70)

        # ----------------------------------------------------
        # KITTI training
        # ----------------------------------------------------

        for batch_idx, (images, targets) in enumerate(
            train_loader
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # Forward pass and loss calculation.
            with autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):

                outputs = model(images)

                loss = total_loss(
                    outputs,
                    targets,
                    tc,
                    class_mask=CLASS_MASK.to(device)
                )

                # Support dictionary-style loss output.
                if isinstance(loss, dict):

                    if "total_loss" in loss:
                        loss = loss["total_loss"]

                    elif "loss" in loss:
                        loss = loss["loss"]

                    else:
                        raise RuntimeError(
                            "Unexpected total_loss dictionary keys: "
                            + str(loss.keys())
                        )

            if not torch.is_tensor(loss):
                raise TypeError(
                    "total_loss must return a Tensor "
                    "or a dictionary containing the loss Tensor."
                )

            # Backpropagation.
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=5.0
            )

            scaler.step(optimizer)
            scaler.update()

            running_loss += float(
                loss.detach().item()
            )

            n_batches += 1

            if (batch_idx + 1) % 100 == 0:

                avg_loss = (
                    running_loss / max(n_batches, 1)
                )

                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Average loss: {avg_loss:.5f}"
                )

        epoch_loss = (
            running_loss / max(n_batches, 1)
        )

        print(
            f"\nEpoch {epoch + 1} KITTI training loss: "
            f"{epoch_loss:.6f}"
        )

        # ====================================================
        # BDD100K VALIDATION
        # ====================================================

        print(
            "\nEvaluating on the fixed "
            "1000-image BDD100K validation subset..."
        )

        val_map, class_aps = evaluate_bdd_map(
            model=model,
            val_loader=val_loader,
            device=device,
            score_threshold=0.0001,
            nms_iou=0.6,
            max_dets=300,
            match_iou=0.5
        )

        print(
            "\nBDD100K validation mAP@0.5:",
            f"{val_map:.6f}"
        )

        # ----------------------------------------------------
        # Per-class AP
        # ----------------------------------------------------

        from datasets import DRIVING_CLASSES

        print("\nPer-class AP:")

        for cls, class_name in enumerate(DRIVING_CLASSES):

            if cls in class_aps:
                print(
                    f"  Class {cls:<4}: "
                    f"{class_aps[cls]:.6f}"
                )
            else:
                print(
                    f"  Class {cls:<4}: N/A"
                )

        # ====================================================
        # BEST CHECKPOINT
        #
        # Save ONLY if BDD100K mAP strictly improves.
        # ====================================================

        improved = (
            val_map > best_map + args.min_delta
        )

        if improved:

            previous_best = best_map

            best_map = val_map
            patience_counter = 0

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                epoch + 1,
                val_map,
                best_map,
                patience_counter,
                args
            )

            print("\n*** NEW BEST BDD mAP CHECKPOINT SAVED ***")

            if previous_best >= 0:

                print(
                    f"Previous best mAP: {previous_best:.6f}"
                )

            else:

                print("No previous BDD mAP checkpoint.")

            print(
                f"New best mAP:      {best_map:.6f}"
            )

            print(
                "Saved to:",
                best_path
            )

        else:

            patience_counter += 1

            print("\nBDD mAP did not improve.")

            print(
                f"Best mAP remains: {best_map:.6f}"
            )

            print(
                f"Early stopping patience: "
                f"{patience_counter}/{args.patience}"
            )

        # ====================================================
        # LATEST CHECKPOINT
        #
        # Always save latest weights after each completed epoch.
        # ====================================================

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scaler,
            epoch + 1,
            val_map,
            best_map,
            patience_counter,
            args
        )

        print(
            "\nLatest checkpoint saved:",
            latest_path
        )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if patience_counter >= args.patience:

            print(
                "\nEarly stopping triggered: BDD100K validation "
                "mAP did not improve for "
                f"{args.patience} consecutive epochs."
            )

            break

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("KITTI TRAINING FINISHED")
    print("=" * 70)

    print(
        "Best BDD100K mAP@0.5:",
        f"{best_map:.6f}"
    )

    print(
        "Best checkpoint:",
        best_path
    )

    print(
        "Latest checkpoint:",
        latest_path
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()