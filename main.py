
"""RP1 entry point: smoke, train, eval, and export."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from config import (
    ModelConfig,
    TrainConfig,
    CalibConfig,
    DataConfig,
)
from datasets import (
    make_dataset,
    collate_fn,
    SyntheticLaneObjectDataset,
)
from models.detector import JointDetector
from train import fit_joint_model, set_seed
from calibration import TemperatureScaling, ConformalBinaryCalibrator
from evaluate import evaluate_all, benchmark_latency
from pipeline import InferencePipeline
from deploy import export_onnx, quantize_int8



def fit_calibrators(model, dcfg, mcfg, tcfg, calcfg):
    """Fit lane conformal calibration using all valid calibration batches."""

    dataset_name = (
        "synthetic" if dcfg.root == "smoke" else dcfg.dataset_name
    )

    if dataset_name == "kitti":
        raise ValueError(
            "KITTI calibration requires lane labels; "
            "use BDD100K or OpenLane."
        )

    ds = make_dataset(
        dataset_name, mcfg, dcfg, split="validation"
    )

    n = len(ds)
    n_cal = int(n * tcfg.calib_split)

    if n_cal < 1 or n_cal >= n:
        raise ValueError(
            "Dataset is too small for the requested calibration split."
        )

    cal_ds, _ = random_split(
        ds,
        [n_cal, n - n_cal],
        generator=torch.Generator().manual_seed(tcfg.seed),
    )

    loader = DataLoader(
        cal_ds,
        batch_size=tcfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=dcfg.num_workers,
    )

    model.eval()
    scores_all = []
    labels_all = []
    valid_images = 0

    with torch.no_grad():
        for x, batch in loader:
            x = x.to(tcfg.device)
            valid = batch["has_lane_labels"].to(
                tcfg.device, dtype=torch.bool
            )

            if not valid.any():
                continue

            out = model(x)

            scores = out["p_exist"][valid]
            targets = batch["lane_targets"].to(
                tcfg.device
            )[valid]

            lane_labels = (targets > 0).long()

            scores = scores.reshape(-1)
            lane_labels = lane_labels.reshape(-1)

            if scores.numel() != lane_labels.numel():
                raise ValueError(
                    "Calibration shape mismatch: "
                    f"scores={scores.numel()}, "
                    f"labels={lane_labels.numel()}"
                )

            if not torch.isfinite(scores).all():
                raise ValueError(
                    "Calibration scores contain NaN or infinity."
                )

            scores_all.append(scores.cpu().numpy())
            labels_all.append(lane_labels.cpu().numpy())
            valid_images += int(valid.sum().item())

    if not scores_all:
        raise ValueError(
            "No lane-labeled samples were found for calibration."
        )

    scores = np.concatenate(scores_all)
    labels = np.concatenate(labels_all)

    print(
        f"Calibration: {valid_images} lane-labeled images, "
        f"{scores.size} lane-row scores."
    )

    conformal = ConformalBinaryCalibrator(calcfg.alpha)
    conformal.fit(scores, labels)

    # Temperature scaling remains unfitted until correctly matched
    # object logits and class labels are collected.
    temperature = TemperatureScaling()

    return temperature, conformal

def load_checkpoint(model, path, device):
    """Load a raw state dictionary or a checkpoint containing model weights."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(
            path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        # Support all checkpoint formats emitted by this project and common
        # external training wrappers.
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint.get("model", checkpoint)),
        )
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")
    model.load_state_dict(state, strict=True)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="RP1 autonomous-driving perception project"
    )

    parser.add_argument(
        "--mode",
        choices=["smoke", "train", "eval", "export"],
        default="smoke",
    )
    
    parser.add_argument(
        "--data",
        choices=["synthetic", "kitti", "bdd100k", "openlane"],
        default="synthetic",
    )

    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--ckpt", default=None, help="Checkpoint path for eval/export; output name for train")
    parser.add_argument("--epochs", type=int, default=None)

    args = parser.parse_args()


    mcfg = ModelConfig()
    tcfg = TrainConfig()
    calcfg = CalibConfig()
    dcfg = DataConfig()
    dcfg.dataset_name = args.data

    # Apply the user-supplied data path before validating it.
    if args.data_dir:
        if args.data == "kitti":
            dcfg.kitti_dir = args.data_dir
        elif args.data == "bdd100k":
            dcfg.bdd_dir = args.data_dir
        elif args.data == "openlane":
            dcfg.openlane_dir = args.data_dir
        else:
            dcfg.root = args.data_dir

    # Validate the resolved BDD100K root.
    if args.data == "bdd100k":
        if not dcfg.bdd_dir or not os.path.isdir(dcfg.bdd_dir):
            parser.error(
                "BDD100K directory not found. "
                "Pass --data_dir with the BDD100K root directory."
            )


    if args.data == "synthetic":
        dcfg.root = "smoke"

    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be at least 1")

        tcfg.epochs = args.epochs

    set_seed(tcfg.seed)

    if not torch.cuda.is_available():
        tcfg.device = "cpu"

    print(f"Mode: {args.mode}")
    print(f"Device: {tcfg.device}")

    if args.mode == "smoke":
        # Quick integration test; not final dataset training.
        tcfg.epochs = 1
        dcfg.synthetic_n = 400
        model = fit_joint_model(dcfg, mcfg, tcfg, "joint_model_smoke")

        temperature, conformal = fit_calibrators(
            model, dcfg, mcfg, tcfg, calcfg
        )

        print(f"Conformal qhat: {conformal.qhat:.4f}")

        ds = SyntheticLaneObjectDataset(64, mcfg)
        loader = DataLoader(
            ds,
            batch_size=4,
            collate_fn=collate_fn,
        )

        results = evaluate_all(
            model,
            loader,
            {"lane": conformal},
            mcfg,
            calcfg,
            tcfg.device,
        )

        print("[Smoke evaluation]")
        print({
            k: v
            for k, v in results.items()
            if k not in (
                "conf_bins",
                "acc_bins",
                "gating_fpr",
                "gating_tpr",
                "gating_thresholds",
            )
        })

        pipeline = InferencePipeline(
            model,
            temperature,
            conformal,
            tau=calcfg.tau_epistemic,
        )

        x, _ = ds[0]
        output = pipeline(x.unsqueeze(0))

        print("Summary:", output["summary"])
        print(
            "Low-confidence flag:",
            output["low_confidence_flag"],
        )

        latency = benchmark_latency(
            model,
            x.unsqueeze(0),
            iters=20,
            device=tcfg.device,
        )

        print(f"Inference latency: {latency:.2f} ms/batch")

    elif args.mode == "train":
        fit_joint_model(dcfg, mcfg, tcfg, os.path.splitext(args.ckpt)[0] if args.ckpt else "joint_model")

    elif args.mode == "eval":
        checkpoint_path = args.ckpt or "joint_model.pt"

        model = JointDetector(mcfg).to(tcfg.device)
        load_checkpoint(model, checkpoint_path, tcfg.device)

        # Final evaluation must use held-out data, never the training split.
        ds = make_dataset(args.data, mcfg, dcfg, split="validation")

        loader = DataLoader(
            ds,
            batch_size=4,
            collate_fn=collate_fn,
            num_workers=dcfg.num_workers,
        )

        results = evaluate_all(
            model,
            loader,
            {"lane": ConformalBinaryCalibrator(calcfg.alpha)},
            mcfg,
            calcfg,
            tcfg.device,
        )

        print("Evaluation results:", results)

    elif args.mode == "export":
        checkpoint_path = args.ckpt or "joint_model.pt"

        model = JointDetector(mcfg).cpu()
        load_checkpoint(model, checkpoint_path, "cpu")

        export_onnx(model, "rp1.onnx", mcfg)
        quantize_int8("rp1.onnx", "rp1_int8.onnx")


if __name__ == "__main__":
    main()
