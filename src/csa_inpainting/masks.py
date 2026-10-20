"""All public masks use B x 1 x H x W, with True/1 meaning missing."""

import torch
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
