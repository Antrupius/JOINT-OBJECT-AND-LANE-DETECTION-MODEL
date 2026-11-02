# Consolidation and code audit

## Active architecture

- `JointDetector` uses the custom RepViT-style backbone from `models/backbone.py`.
- The object branch is `YOLOv5StyleDetectionHead` in `models/yolov5_detection.py`, with three feature scales and three anchors per scale.
- The lane branch is `LaneHead` in `models/heads.py`.
- `losses.py` calls `yolo_object_loss` for object-labelled samples and the lane loss for lane-labelled samples.
- Class weights are applied as the positive weights of the class BCE loss; they do not change objectness or box regression directly.

## Consolidated changes

- Preserved the audited model, target assignment, loss, dataset, and checkpoint-loading implementations from the supplied current project.
- Kept the current class weights: `[1, 3, 3, 3, 3, 3, 2, 2]`.
- Added `TrainConfig.chunk_size = 7000`.
- Updated the original `train.py` so BDD100K training processes the full shuffled dataset in sequential 7,000-image subsets per epoch. The same model, optimizer, and gradient scaler persist across all chunks; validation occurs after the complete epoch.
- Made the main BDD100K evaluator and prediction visualizer accept checkpoint and dataset paths as command-line arguments.
- Retained one baseline checkpoint and its recorded metrics for comparison.
- Removed experiment-only training scripts, one-off check/diagnostic scripts, duplicate checkpoint variants, stale backup copies, Python bytecode caches, and generated OpenLane subset-run files.

## Verification performed in the packaging environment

- `python -m compileall -q .` passed.
- `python -m unittest tests.test_stage1_smoke -v` passed before the test scripts were removed from the deliverable.
- `python -m unittest tests.test_audit_regressions -v` passed before the test scripts were removed from the deliverable.
- `python test_yolov5_detection_loss.py` passed before the diagnostic scripts were removed.
- `python test_detector_wiring.py` passed before the diagnostic scripts were removed.
- `python test_losses_smoke.py` passed before the diagnostic scripts were removed.

These checks validate syntax, model wiring, basic loss/gradient behavior, and selected evaluation/checkpoint regressions. They do not establish real BDD100K accuracy.

## Still to do on the project machine

- Run the final two epochs against the actual BDD100K dataset using the command in `README.md`.
- Evaluate the final checkpoint using the same 1,000-image validation subset, then compare mAP@0.5 and per-class AP with `BASELINE_RESULTS.md`.
- Inspect the new predictions visually and report limitations honestly.

No full-dataset training was run while preparing this ZIP because the dataset is not included in the upload.
