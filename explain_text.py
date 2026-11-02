"""RP1 Stage 3: deterministic rule-based decision summary (no generative text, so every
clause is traceable to a structured model output — faithfulness by construction)."""
from typing import Dict, List, Tuple

CLASS_NAMES = {0: "car", 1: "truck", 2: "bus", 3: "pedestrian",
               4: "cyclist", 5: "motorcycle", 6: "traffic light", 7: "traffic sign"}

# heuristic prior heights (m) per class for monocular depth estimation
CLASS_HEIGHTS = {0: 1.5, 1: 3.2, 2: 3.2, 3: 1.7, 4: 1.7, 5: 1.2, 6: 0.8, 7: 0.9}


def estimate_depth(box, cls, img_h: int, focal_ratio: float = 1.0) -> float:
    """Heuristic monocular depth from box height (f = focal/100 scaled by image height).
    Used only for the summary template; replace with the 3D head when extended."""
    h_px = max(box[3] - box[1], 8.0)
    f_eff = focal_ratio * img_h
    return float(f_eff * CLASS_HEIGHTS.get(int(cls), 1.5) / h_px)


def build_summary(dets: List, lane_sets, lane_vacuity, img_h: int, u_epi_thr: float = 0.6) -> str:
    """dets: rows with (x1,y1,x2,y2, class, detection score, reserved columns).
    Object uncertainty is not estimated by this detector.
    lane_sets: (num_lanes, h_steps) booleans for 'lane present' per row."""
    clauses = []
    # --- lane clauses ---
    for li, sets in enumerate(lane_sets):
        present_frac = float(sets.mean()) if hasattr(sets, "mean") else float(np.mean(sets))
        vac = float(np.mean(lane_vacuity[li])) if lane_vacuity is not None else 0.0
        if present_frac < 0.3:
            clauses.append(f"lane {li+1} absent from field of view (coverage set excludes it, vacuity {vac:.2f})")
        elif vac > u_epi_thr:
            clauses.append(f"lane {li+1} uncertain: model lacks evidence (vacuity {vac:.2f})")
    # --- object clauses ---
    for d in dets[:4]:
        x1, y1, x2, y2, cls, score = d[:6]
        depth = estimate_depth((x1, y1, x2, y2), cls, img_h)
        name = CLASS_NAMES.get(int(cls), "object")
        confidence_note = "low detection score" if score < 0.5 else f"detection score {score:.2f}"
        clauses.append(f"{name} at ~{depth:.0f} m with {confidence_note}; depth is a heuristic estimate")
    if not clauses:
        return "free road: no lanes exiting coverage, no critical objects."
    return "; ".join(clauses)


import numpy as np
