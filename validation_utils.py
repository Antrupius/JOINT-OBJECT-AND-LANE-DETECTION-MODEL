"""Validation helpers for direct joint training."""
import math
import torch
from losses import total_loss


@torch.no_grad()
def validate_loss(model, loader, cfg):
    """Calculate mean validation loss without updating model weights."""
    was_training = model.training
    model.eval()
    if not hasattr(cfg, "_epoch"):
        cfg._epoch = 0
    total_value, batches, samples_seen = 0.0, 0, 0
    try:
        for images, batch in loader:
            images = images.to(cfg.device)
            batch = {key: value.to(cfg.device) if torch.is_tensor(value) else value for key, value in batch.items()}
            amp_enabled = str(cfg.device).startswith("cuda") and torch.cuda.is_available()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                outputs = model(images)
                loss = total_loss(outputs, batch, cfg)
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise ValueError(f"Validation produced a non-finite loss: {value}")
            total_value += value * images.size(0)
            batches += 1
            samples_seen += images.size(0)
        if batches == 0:
            raise ValueError("Validation loader produced no batches.")
        return total_value / samples_seen
    finally:
        model.train(was_training)
