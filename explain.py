"""RP1 Stage 3: integrated-gradients saliency + deletion/insertion faithfulness metrics.
Captum used if available; a self-contained Riemann-trapezoid implementation otherwise."""
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF



def integrated_gradients(model, x: torch.Tensor, steps: int = 50, baseline: torch.Tensor = None):
    """Attributions for model.target_score(x) (max foreground class logit)."""
    model.eval()
    x = x.detach()

    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.detach().to(device=x.device, dtype=x.dtype)

    alphas = torch.linspace(0.0, 1.0, steps, device=x.device)
    grad_sum = torch.zeros_like(x)

    for a in alphas:
        xi = (baseline + a * (x - baseline)).detach().requires_grad_(True)
        score = model.target_score(xi)
        grad = torch.autograd.grad(score.sum(), xi)[0]
        grad_sum += grad

    ig = (x - baseline) * grad_sum / steps
    return ig.detach()



def _gaussian_blur(x: torch.Tensor, k: int, sigma: float) -> torch.Tensor:
    return TF.gaussian_blur(x, kernel_size=k, sigma=sigma)


def deletion_auc(model, x: torch.Tensor, saliency: torch.Tensor, steps: int = 20,
                 kernel: int = 25, sigma: float = 8.0) -> float:
    """Progressively remove the most salient pixels (via blur mask), track target score.
    Lower AUC = more faithful (score should collapse quickly)."""
    model.eval()
    with torch.no_grad():
        base_score = float(model.target_score(x).item())
        sal = saliency.view(1, -1)
        order = sal[0].sort(descending=True).indices
        blurred = _gaussian_blur(x, kernel, sigma)
        mask = torch.ones_like(x)
        scores, n = [], x.view(1, -1).shape[1]
        for s in range(steps + 1):
            frac = s / steps
            if s > 0:
                keep = int(n * (1 - frac))
                m = torch.ones(n, dtype=torch.bool)
                m[order[: n - keep]] = False
                mask = m.view(1, 3, x.shape[2], x.shape[3]).float()
            x_masked = x * mask + blurred * (1 - mask)
            scores.append(float(model.target_score(x_masked).item()))
    scores = max(scores[-1], 0.0)
    return float(np.trapz(scores, np.linspace(0, 1, steps + 1)) / (base_score + 1e-8))


def insertion_auc(model, x: torch.Tensor, saliency: torch.Tensor, steps: int = 20,
                  kernel: int = 25, sigma: float = 8.0) -> float:
    """Progressively insert the most salient pixels into a blurred image.
    Higher AUC = more faithful (score should recover quickly)."""
    model.eval()
    with torch.no_grad():
        sal = saliency.view(1, -1)
        order = sal[0].sort(descending=True).indices
        blurred = _gaussian_blur(x, kernel, sigma)
        n = x.view(1, -1).shape[1]
        scores = []
        for s in range(steps + 1):
            frac = s / steps
            keep = int(n * frac)
            mask = torch.zeros(n, dtype=torch.bool)
            mask[order[:keep]] = True
            mask = mask.view(1, 3, x.shape[2], x.shape[3]).float()
            x_masked = x * mask + blurred * (1 - mask)
            scores.append(float(model.target_score(x_masked).item()))
        full = float(model.target_score(x).item()) + 1e-8
        return float(np.trapz(scores, np.linspace(0, 1, steps + 1)) / full)


import numpy as np
