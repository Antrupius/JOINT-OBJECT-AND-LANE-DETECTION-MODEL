# Joint Autonomous-Driving Perception

A compact joint model for object detection and lane detection on BDD100K-style data.

## What's in this package

- `models/backbone.py` — compact RepViT-style feature extractor.
- `models/yolov5_detection.py` — multi-scale YOLOv5-inspired detection head, target assignment, decoding, and detection loss.
- `models/heads.py` — lane-detection head.
- `models/detector.py` — joint model wiring.
- `datasets.py` — BDD100K, OpenLane, KITTI, and synthetic dataset adapters.
- `losses.py`, `train.py`, `validation_utils.py` — joint loss and training/validation.
- `evaluate.py`, `evaluate_bdd_checkpoint.py` — evaluation utilities.
- `visualize_bdd_predictions.py`, `visualize_lanes.py` — prediction and lane visualizations.
- `pipeline.py`, `calibration.py`, `explain.py`, `explain_text.py`, `deploy.py` — inference, calibration, explainability, and export utilities.
- `checkpoints/bdd100k_7000_1000_final_audit_v1.pt` — retained **baseline checkpoint** for comparison. It is not the result of the final full-dataset run.

Experimental trainers, one-off diagnostics, duplicate checkpoints, backup copies, Python caches, and test/check scripts have been excluded from this clean package.

## Requirements

Use the existing `MinorProject` Conda environment if it already has PyTorch and torchvision installed. Otherwise install the dependencies for your Python/CUDA setup, then install the remaining packages:

```bash
pip install -r requirements.txt
```

PyTorch and torchvision must be compatible with each other and with your CUDA runtime. If CUDA is unavailable, the code falls back to CPU, but full BDD100K training will be much slower.

## Expected BDD100K directory layout

Set `--data_dir` to the folder that contains the `train` and `val` directories:

```text
BDD100K/
  train/
    img/
    ann/
  val/
    img/
    ann/
```

Each image in `img` must have a matching JSON annotation at `ann/<image filename>.json`. The current dataset adapter reads object rectangles and lane polylines from the `objects` list.

## Resumable chunk-by-chunk BDD100K training

Use `train_bdd_chunks.py` for the final run when you want to process **one 7,000-image chunk per command**, then resume with the next chunk. It initializes from the retained baseline checkpoint, unless you choose another `--init-ckpt`.

First invocation (starts from the baseline and trains the first chunk):

```bat
python train_bdd_chunks.py --data-dir "D:\MinorProjectData\BDD100K" --epochs 2 --chunks-per-run 1 --best-ckpt "checkpoints\bdd100k_full_best.pt" --resume-ckpt "checkpoints\bdd100k_full_resume.pt" --init-ckpt "checkpoints\bdd100k_7000_1000_final_audit_v1.pt" --save-every-images 500 --validate-every-images 1000 --validation-images 500
```

Every later invocation uses the **same command plus `--resume`**:

```bat
python train_bdd_chunks.py --data-dir "D:\MinorProjectData\BDD100K" --epochs 2 --chunks-per-run 1 --best-ckpt "checkpoints\bdd100k_full_best.pt" --resume-ckpt "checkpoints\bdd100k_full_resume.pt" --save-every-images 500 --validate-every-images 1000 --validation-images 500 --resume
```

Run the second command again for each next chunk. With about 70,000 training images, expect 10 chunks per epoch and 20 chunks over two epochs. The last chunk may be smaller if the dataset size is not divisible by 7,000.

### Checkpoint behavior

- **Resume checkpoint:** saved every 500 processed training images and at each chunk boundary. It contains model weights, optimizer state, AMP scaler, RNG state, epoch/chunk cursor, and image offset. If interrupted, resume from the latest saved point; at most the work since the last save must be repeated.
- **Best checkpoint:** validation runs on the same seeded, fixed 500-image validation subset every 1,000 training images and at chunk boundaries if needed. `checkpoints\bdd100k_full_best.pt` is updated when this validation loss improves.
- **Full validation:** the complete validation split is evaluated at the end of each epoch for reporting.
- **Best means lowest validation loss on the fixed subset**, not guaranteed highest mAP. After training, evaluate the best checkpoint using the detection evaluator and inspect per-class AP and visual predictions.
- The training set is deterministically shuffled once per epoch and divided into chunks. Model, optimizer, and AMP scaler persist between chunks and between command invocations.

Do not delete or rename the resume checkpoint while you intend to continue the same run. To intentionally start over, move the resume file out of the way and use the first command again. Existing best checkpoints are not overwritten unless validation improves.

## Evaluate the final checkpoint

Evaluate a deterministic subset of 1,000 validation images:

```bat
python evaluate_bdd_checkpoint.py --ckpt "checkpoints\bdd100k_full_2ep_final.pt" --data_dir "D:\MinorProjectData\BDD100K" --images 1000
```

For a full validation-set run, use `--images` equal to the validation-set size. This may take considerably longer.

## Visualize lane predictions

```bat
python visualize_lanes.py --ckpt "checkpoints\\bdd100k_full_2ep_final.pt" --data_dir "D:\\MinorProjectData\\BDD100K" --images 12
```

Green lane anchors indicate ground truth; red anchors indicate model predictions. Outputs are saved under `results/lane_visualizations/` by default.

## Visualize detections

```bat
python visualize_bdd_predictions.py --ckpt "checkpoints\bdd100k_full_2ep_final.pt" --data_dir "D:\MinorProjectData\BDD100K" --images 6
```

Green boxes indicate ground truth and red boxes indicate model predictions. To change the display confidence threshold, pass `--score_thr`, for example `--score_thr 0.15`.

## Class order

The model uses this fixed class order:

0. car
1. truck
2. bus
3. pedestrian (`person` in BDD annotations)
4. cyclist (`rider` / `bike` in BDD annotations)
5. motorcycle (`motor` in BDD annotations)
6. traffic light
7. traffic sign

## Important limitations

- The backbone is a project implementation inspired by RepViT design principles; it does not load official pretrained RepViT weights.
- The detection head is YOLOv5-inspired, not the official Ultralytics YOLOv5 implementation.
- The included baseline was trained on a 7,000-image training subset and evaluated on a 1,000-image validation subset. It is retained for comparison only.
- The class weights are an experimental imbalance mitigation, not a guarantee of better minority-class AP. Use the held-out metrics to assess their actual effect.
- The model does not currently provide validated object epistemic/aleatoric uncertainty. Do not report its reserved compatibility columns as calibrated uncertainty.
- The monocular distance summary, where used, is a heuristic and not a validated depth estimate.
- No BDD100K dataset is included in this ZIP. The final two-epoch training and its evaluation must be run on the project machine with the dataset available.

See `BASELINE_RESULTS.md` for the previously recorded subset evaluation.
