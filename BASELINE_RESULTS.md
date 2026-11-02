# Baseline evaluation record

This record documents the existing `bdd100k_7000_1000_final_audit_v1.pt` checkpoint before the planned full-dataset training. It is not a claim about the new final checkpoint.

## Evaluation setup

- Training subset: 7,000 BDD100K images
- Validation subset: 1,000 BDD100K images
- AP score floor: 0.0001
- Operating confidence threshold: 0.25
- NMS IoU: 0.6
- Detection matching IoU: 0.5
- Ground-truth objects: 18,618

## Recorded object metrics

| Metric | Value |
|---|---:|
| mAP@0.5 | 0.0332 |
| Precision | 0.3468 |
| Recall | 0.1328 |
| F1 | 0.1920 |
| Macro F1 | 0.0372 |
| True positives | 2,472 |
| False positives | 4,657 |
| False negatives | 16,146 |

## Per-class AP@0.5

| Class | AP@0.5 |
|---|---:|
| car | 0.2182 |
| truck | 0.0141 |
| bus | 0.0000 |
| pedestrian | 0.0109 |
| cyclist | 0.0000 |
| motorcycle | 0.0000 |
| traffic light | 0.0161 |
| traffic sign | 0.0066 |

The result shows that the earlier checkpoint was weak, especially on non-car classes. These numbers motivate the final full-data run, but they do not prove that class weighting or two additional epochs will fix the issue. After training, rerun the evaluator on the same deterministic validation subset and compare per-class metrics and visual predictions.
