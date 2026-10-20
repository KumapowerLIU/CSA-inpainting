"""Offline default-size D/G integration check, NOT a reproduction experiment.

Uses random VGG weights solely to check the complete architecture without a
download. The public `csa train` command always requires pretrained VGG weights.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from csa_inpainting.config import Config
from csa_inpainting.engine import Trainer, seed_everything
from csa_inpainting.masks import center_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default="runs/full-check/report.json")
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("Use batch-size >= 2 to exercise independent masks")
    torch.set_num_threads(2)
    seed_everything(2019)
    config = Config()
    config.train.device = args.device
    start = time.monotonic()
    trainer = Trainer(config, diagnostic="random_vgg")
    image = torch.rand(args.batch_size, 3, 256, 256) * 2 - 1
    mask = torch.stack(
        [torch.roll(center_mask(256, 256, 4), i * 23, -1) for i in range(args.batch_size)]
    )
    losses = trainer.step(image, mask)
    report = {
        "purpose": (
            "Full-size software integration only; random VGG, not training-quality validation"
        ),
        "torch": str(torch.__version__),
        "feature_mode": trainer.feature_mode,
        "device": str(trainer.device),
        "batch_size": args.batch_size,
        "image_size": 256,
        "generator_parameters": sum(p.numel() for p in trainer.generator.parameters()),
        "seconds": time.monotonic() - start,
        "losses": losses,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
