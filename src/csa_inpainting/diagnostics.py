"""Small deterministic fixtures for offline software checks, not image restoration."""

from torch import Tensor, nn
from torch.nn import functional as F


class TinyFeatures(nn.Module):
    """Frozen synthetic features; never substituted for VGG in ordinary training."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        x = F.avg_pool2d(image, 8)
        x = x.repeat(1, (self.channels + 2) // 3, 1, 1)[:, : self.channels]
        return {"pool3": x, "relu4_3": x}
