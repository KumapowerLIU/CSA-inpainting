"""Compare stage-1 GPU throughput on cached real samples; exclude input I/O."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from csa_remote_train import RecordDataset
from torch.utils.data import DataLoader, Subset

from csa_inpainting.config import Config
from csa_inpainting.engine import seed_everything
from csa_inpainting.stage1 import Stage1Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    args = parser.parse_args()
    runtime = Path(args.runtime)
    torch.set_num_threads(8)
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(48 * 1024**3 / capacity)
    config = Config()
    config.train.device, config.train.amp = "cuda:0", "bf16"
    data = RecordDataset(runtime / "data/train.jsonl", True, config.data)
    pieces = list(
        DataLoader(Subset(data, range(64)), batch_size=16, num_workers=4, prefetch_factor=1)
    )
    base_image = torch.cat([part["image"] for part in pieces])
    base_mask = torch.cat([part["mask"] for part in pieces])
    del pieces
    results = []
    for size in [32, 128, 256, 512, 1024]:
        trainer = images = masks = None
        try:
            seed_everything(2019)
            config.data.batch_size = size
            trainer = Stage1Trainer(config)
            indices = torch.arange(size) % len(base_image)
            images, masks = base_image[indices].pin_memory(), base_mask[indices].pin_memory()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for step in range(7):
                torch.cuda.synchronize()
                start = time.monotonic()
                losses = trainer.step(images, masks)
                torch.cuda.synchronize()
                if step >= 2:
                    times.append(time.monotonic() - start)
            mean = sum(times) / len(times)
            row = dict(
                batch_size=size,
                status="PASS",
                seconds=times,
                mean_seconds=mean,
                images_per_second=size / mean,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                losses=losses,
            )
        except torch.cuda.OutOfMemoryError as error:
            row = dict(batch_size=size, status="OOM", error=str(error))
        results.append(row)
        print(json.dumps(row), flush=True)
        (runtime / "batch-benchmarks.json").write_text(
            json.dumps(
                {
                    "scope": (
                        "GPU updates on 64 cached real images repeated as needed; "
                        "input I/O excluded"
                    ),
                    "stage": 1,
                    "amp": "bf16",
                    "allocator_limit_gib": 48,
                    "results": results,
                },
                indent=2,
            )
        )
        del trainer, images, masks
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
