
"""RP1 inference pipeline and consumer agent."""
import numpy as np
import torch

from explain import integrated_gradients
from explain_text import build_summary


class InferencePipeline:
    def __init__(self, model, temperature, conformal,
                 tau=0.6, ig_steps=50):
        self.model = model
        self.T = temperature
        self.conformal = conformal
        self.tau = tau
        self.ig_steps = ig_steps

    def __call__(self, x: torch.Tensor) -> dict:
        self.model.eval()

        with torch.no_grad():
            out = self.model(x)
            dets = self.model.decode(out)

            # Conformal lane prediction sets
            pe = out["p_exist"][0].cpu().numpy()
            lane_sets = []

            for li in range(pe.shape[0]):
                inc1, _ = self.conformal.predict_set(pe[li])
                lane_sets.append(inc1)

            det0 = dets[0].cpu().numpy()
            vac_lane = out["vacuity_lane"][0].cpu().numpy()

        # Integrated Gradients requires gradients, so run outside no_grad.
        saliency = integrated_gradients(
            self.model, x, steps=self.ig_steps
        )

        summary = build_summary(
            det0, lane_sets, vac_lane,
            self.model.cfg.image_h, self.tau
        )

        # Object epistemic/aleatoric uncertainty is not estimated by this head.
        # Use a score threshold for object confidence and keep lane gating separate.
        object_scores_low = bool((det0[:, 5] < 0.5).any()) if len(det0) else False
        low_confidence = (
            object_scores_low
            or bool((vac_lane > self.tau).mean() > 0.5)
        )

        return {
            "detections": det0,
            "lane_sets": lane_sets,
            "saliency": saliency[0],
            "summary": summary,
            "u_alea": None,
            "u_epi": None,
            "object_uncertainty_available": False,
            "low_confidence_flag": low_confidence,
        }


class ConsumerAgent:
    """Applies safety gating and monitors streaming coverage."""

    def __init__(self, alpha, ema=0.9, drift_thresh=0.03):
        self.alpha = alpha
        self.ema = ema
        self.drift = drift_thresh
        self.cov_ema = None

    def update(self, pipeline_out, observed_coverage):
        if self.cov_ema is None:
            self.cov_ema = observed_coverage
        else:
            self.cov_ema = (
                self.ema * self.cov_ema
                + (1 - self.ema) * observed_coverage
            )

        if pipeline_out["low_confidence_flag"]:
            return (
                "ESCALATE: low-confidence scene "
                "(epistemic uncertainty above tau)."
            )

        if self.cov_ema < (1 - self.alpha) - self.drift:
            return "RECALIBRATE: streaming coverage drifted below 1 - alpha."

        return "NOMINAL"
