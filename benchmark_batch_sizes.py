
import gc
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, DataConfig, TrainConfig
from datasets import make_dataset, collate_fn
from models.detector import JointDetector
from losses import total_loss
from train import _move_batch

DATA_DIR = r"D:\MinorProjectData\BDD100K"
CHECKPOINT = r"experiments\class_aware\resume.pt"
BATCH_SIZES = [4, 5, 6, 8]
WARMUP = 3
STEPS = 10


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    state = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )

    mc, dc, tc = ModelConfig(), DataConfig(), TrainConfig()
    dc.dataset_name = "bdd100k"
    dc.bdd_dir = DATA_DIR
    dc.num_workers = 0
    tc.device = "cuda"
    tc.image_h, tc.image_w = mc.image_h, mc.image_w
    tc.num_classes = mc.num_classes
    tc._epoch = int(state.get("epoch_idx", 7))

    print("Loading BDD100K training data...")
    dataset = make_dataset("bdd100k", mc, dc, split="train")
    print("Images:", len(dataset))

    for bs in BATCH_SIZES:
        model = optimizer = scaler = loader = None
        gc.collect()
        torch.cuda.empty_cache()

        try:
            print(f"\nTesting batch size {bs}...", flush=True)

            model = JointDetector(mc).to(device)
            model.load_state_dict(state["model_state_dict"])
            model.train()

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-4,
                weight_decay=tc.weight_decay
            )
            scaler = torch.amp.GradScaler("cuda", enabled=True)

            loader = DataLoader(
                dataset, batch_size=bs, shuffle=False,
                drop_last=True, collate_fn=collate_fn,
                num_workers=0, pin_memory=True
            )
            iterator = iter(loader)

            torch.cuda.reset_peak_memory_stats()
            elapsed = 0.0
            processed = 0

            for step in range(WARMUP + STEPS):
                try:
                    images, batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    images, batch = next(iterator)

                images = images.to(device, non_blocking=True)
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)

                torch.cuda.synchronize()
                start = time.perf_counter()

                with torch.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    outputs = model(images)
                    loss = total_loss(outputs, batch, tc)

                if not torch.isfinite(loss).all():
                    raise FloatingPointError("Non-finite loss")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                torch.cuda.synchronize()

                if step >= WARMUP:
                    elapsed += time.perf_counter() - start
                    processed += images.shape[0]

            allocated = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3

            print(f"Images/sec: {processed / elapsed:.2f}")
            print(f"Seconds/step: {elapsed / STEPS:.3f}")
            print(f"Peak allocated VRAM: {allocated:.2f} GB")
            print(f"Peak reserved VRAM: {reserved:.2f} GB")

        except torch.cuda.OutOfMemoryError:
            print("CUDA OUT OF MEMORY")
        finally:
            del model, optimizer, scaler, loader
            gc.collect()
            torch.cuda.empty_cache()

    print("\nBenchmark finished. No checkpoints were modified.")


if __name__ == "__main__":
    main()
