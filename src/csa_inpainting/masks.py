"""Binary masks: generators return 1 x H x W; batched masks are B x 1 x H x W."""

import math
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.nn import functional as F


def validate_mask(mask: Tensor, image: Tensor) -> Tensor:
    expected = (image.shape[0], 1, *image.shape[-2:])
    if tuple(mask.shape) != expected:
        raise ValueError(f"Expected one mask per sample {expected}, got {tuple(mask.shape)}")
    if mask.device != image.device:
        raise ValueError("Image and mask must be on the same device")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("Masks must be binary, with 1 meaning missing")
    return mask.bool()


def feature_mask(mask: Tensor, levels: int = 3, threshold: float = 5 / 16) -> Tensor:
    """Appendix A: repeated 4x4, stride-2, padding-1 averaging convolutions."""
    x = mask.float()
    for _ in range(levels):
        x = F.avg_pool2d(x, 4, stride=2, padding=1, count_include_pad=True)
    return x > threshold


def center_mask(height: int, width: int, overlap: int = 0) -> Tensor:
    if overlap < 0 or overlap * 2 >= min(height // 2, width // 2):
        raise ValueError("overlap must leave a nonempty center hole")
    mask = torch.zeros(1, height, width, dtype=torch.bool)
    mask[
        :,
        height // 4 + overlap : 3 * height // 4 - overlap,
        width // 4 + overlap : 3 * width // 4 - overlap,
    ] = True
    return mask


def random_mask(
    height: int,
    width: int,
    rng: random.Random,
    min_fraction: float = 0.1,
    max_fraction: float = 0.4,
) -> Tensor:
    """Sample rectangles, free-form strokes, or their union at random locations.

    The supplied RNG makes masks independent of global/worker RNG state. Rejection
    sampling bounds the missing area; this is a synthetic training distribution,
    not the irregular-mask benchmark released with the original paper.
    """
    if height < 2 or width < 2:
        raise ValueError("Random mask dimensions must be at least 2")
    if not 0 < min_fraction < max_fraction < 1:
        raise ValueError("Require 0 < min_fraction < max_fraction < 1")
    pixels = height * width
    minimum, maximum = math.ceil(pixels * min_fraction), math.floor(pixels * max_fraction)
    if minimum > maximum:
        raise ValueError("Mask fraction range contains no whole-pixel area")
    short_side = min(height, width)
    for _ in range(32):
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        kind = rng.choice(("rectangles", "strokes", "mixed"))
        if kind in {"rectangles", "mixed"}:
            for _ in range(rng.randint(1, 3)):
                box_h = rng.randint(max(1, height // 8), max(1, height * 2 // 3))
                box_w = rng.randint(max(1, width // 8), max(1, width * 2 // 3))
                top, left = rng.randrange(height - box_h + 1), rng.randrange(width - box_w + 1)
                draw.rectangle((left, top, left + box_w - 1, top + box_h - 1), fill=255)
        if kind in {"strokes", "mixed"}:
            for _ in range(rng.randint(2, 6)):
                x, y = rng.randrange(width), rng.randrange(height)
                for _ in range(rng.randint(1, 4)):
                    angle = rng.uniform(0, 2 * math.pi)
                    length = rng.uniform(short_side / 16, short_side / 2)
                    nx = max(0, min(width - 1, round(x + length * math.cos(angle))))
                    ny = max(0, min(height - 1, round(y + length * math.sin(angle))))
                    stroke = rng.randint(max(1, short_side // 25), max(2, short_side // 7))
                    draw.line((x, y, nx, ny), fill=255, width=stroke)
                    radius = stroke // 2
                    for cx, cy in ((x, y), (nx, ny)):
                        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
                    x, y = nx, ny
        array = np.array(canvas, copy=True) > 0
        if minimum <= int(array.sum()) <= maximum:
            return torch.from_numpy(array).unsqueeze(0)

    # Very narrow/custom bounds may defeat rejection sampling. Use a randomly
    # placed rectangle with a partial final row to guarantee an admissible area.
    count = rng.randint(minimum, maximum)
    box_w = rng.randint(math.ceil(count / height), min(width, count))
    full_rows, remainder = divmod(count, box_w)
    box_h = full_rows + bool(remainder)
    top, left = rng.randrange(height - box_h + 1), rng.randrange(width - box_w + 1)
    array = np.zeros((height, width), dtype=np.bool_)
    array[top : top + full_rows, left : left + box_w] = True
    if remainder:
        array[top + full_rows, left : left + remainder] = True
    return torch.from_numpy(array).unsqueeze(0)
