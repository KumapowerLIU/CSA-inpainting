"""Recursive image loading and deterministic per-sample/per-epoch augmentation."""

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

from .config import DataConfig
from .masks import center_mask, random_mask

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())
    if not paths:
        raise ValueError(f"No supported images found in {root}")
    return paths


def rgb_tensor(image: Image.Image) -> Tensor:
    array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1


def read_mask(path: str | Path, size: tuple[int, int], white_is_hole: bool) -> Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize(size, Image.Resampling.NEAREST)
        mask = torch.from_numpy(np.array(image, copy=True)) >= 128
    return (mask if white_is_hole else ~mask).unsqueeze(0)


def brush_mask(size: int, rng: random.Random) -> Tensor:
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    for _ in range(rng.randint(2, 6)):
        x, y = rng.randrange(size), rng.randrange(size)
        for _ in range(rng.randint(2, 5)):
            angle, length = rng.uniform(0, 2 * math.pi), rng.uniform(size / 16, size / 3)
            nx = max(0, min(size - 1, round(x + length * math.cos(angle))))
            ny = max(0, min(size - 1, round(y + length * math.sin(angle))))
            width = rng.randint(max(1, size // 32), max(2, size // 10))
            draw.line((x, y, nx, ny), fill=255, width=width)
            radius = width // 2
            draw.ellipse((nx - radius, ny - radius, nx + radius, ny + radius), fill=255)
            x, y = nx, ny
    return torch.from_numpy(np.array(canvas, copy=True)).unsqueeze(0) > 0


class InpaintingDataset(Dataset):
    def __init__(self, config: DataConfig, training: bool, seed: int = 2019):
        self.config, self.training, self.seed, self.epoch = config, training, seed, 0
        root = config.train_images if training else config.val_images
        if not root:
            raise ValueError("Set train_images/val_images to a nonempty image directory")
        self.paths = image_paths(root)
        self.masks = []
        if config.mask_mode == "files":
            mask_root = config.train_masks if training else config.val_masks
            if not mask_root:
                raise ValueError("mask_mode=files requires train_masks/val_masks")
            self.masks = image_paths(mask_root)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch if self.training else 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        # Independent of worker count, ordering, prefetching and global RNG.
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index * 97)
        size = self.config.image_size
        with Image.open(self.paths[index]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image = image.resize((size, size), Image.Resampling.BICUBIC)
            if self.training and self.config.horizontal_flip and rng.random() < 0.5:
                image = ImageOps.mirror(image)
            image = rgb_tensor(image)
        if self.config.mask_mode == "center":
            mask = center_mask(size, size, self.config.center_overlap)
        elif self.config.mask_mode == "random":
            mask = random_mask(
                size, size, rng, self.config.random_min_fraction, self.config.random_max_fraction
            )
        elif self.config.mask_mode == "brush":
            mask = brush_mask(size, rng)
        else:
            # Validation has a fixed sorted pairing (cycling if counts differ).
            mask_index = (
                rng.randrange(len(self.masks)) if self.training else index % len(self.masks)
            )
            mask = read_mask(self.masks[mask_index], (size, size), self.config.mask_white_is_hole)
        if bool(mask.all()) or not bool(mask.any()):
            raise ValueError(f"Empty or fully missing mask for {self.paths[index]}")
        return {"image": image, "mask": mask, "path": str(self.paths[index])}
