"""RP1 Stage 2: temperature scaling (objects) + split-conformal sets (lanes) + ECE."""
import numpy as np
import torch
import torch.nn.functional as F


class TemperatureScaling:
    """Post-hoc scalar temperature T* minimizing NLL of softmax(z/T) on the validation split."""

    def __init__(self, max_iter: int = 500):
        self.T = 1.0
        self.max_iter = max_iter

    def fit(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """logits: (n, C); targets: (n,) class indices."""
        logits, targets = logits.detach().cpu(), targets.detach().cpu()
        t = torch.tensor(0.0, requires_grad=True)              # T = softplus(t) > 0
        opt = torch.optim.LBFGS([t], max_iter=self.max_iter, lr=0.1)

        def closure():
            opt.zero_grad()
            T = F.softplus(t)
            loss = F.cross_entropy(logits / T, targets)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(F.softplus(t))
        return self.T

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T


class ConformalBinaryCalibrator:
    """Split-conformal prediction sets for binary lane-existence per row.
    Under exchangeability the sets satisfy 1 - alpha <= Pr(y in C(x)) <= 1 - alpha + 1/(n+1)."""

    def __init__(self, alpha: float = 0.1):
        self.alpha, self.qhat = alpha, None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """scores = P(y=1) predictions (n,); labels in {0,1} (n,)."""
        scores, labels = np.asarray(scores, float), np.asarray(labels, int)
        n = len(scores)
        nonconf = np.abs(labels - scores)                      # s(x,y) = 1 - P(y)
        idx = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.qhat = float(np.sort(nonconf)[min(idx - 1, n - 1)])
        return self.qhat

    def predict_set(self, scores: np.ndarray):
        """Returns, per row: includes1 (P(y=1) >= 1 - qhat), includes0 (P(y=0) >= 1 - qhat)."""
        s = np.asarray(scores, float)
        includes1 = (1.0 - s) <= self.qhat
        includes0 = s <= self.qhat
        return includes1, includes0

    def coverage(self, scores: np.ndarray, labels: np.ndarray) -> float:
        inc1, inc0 = self.predict_set(scores)
        y = np.asarray(labels, int) == 1
        return float(np.mean(np.where(y, inc1, inc0)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with confidence binning."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        acc = (labels[m] == 1).mean()
        conf = probs[m].mean()
        total += (m.sum() / len(probs)) * abs(acc - conf)
    return float(total)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Data for reliability diagrams (accuracy and confidence per bin)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        confs.append(probs[m].mean()); accs.append((labels[m] == 1).mean())
    return np.array(confs), np.array(accs)
