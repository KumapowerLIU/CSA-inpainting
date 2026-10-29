"""Explicit CSA encoder/decoders; released-code differences are in MIGRATION.md."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .attention import CoherentSemanticAttention
from .masks import feature_mask, validate_mask


def norm(channels: int) -> nn.Module:
    return nn.InstanceNorm2d(channels, affine=True)


def initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def channels_for(width: int, depth: int) -> list[int]:
    return [width * min(2**i, 8) for i in range(depth)]


class RoughNetwork(nn.Module):
    def __init__(self, width: int = 64, depth: int = 8):
        super().__init__()
        channels = [3, *channels_for(width, depth)]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth):
            layers = [] if i == 0 else [nn.LeakyReLU(0.2)]
            layers.append(nn.Conv2d(channels[i], channels[i + 1], 4, 2, 1))
            if 0 < i < depth - 1:
                layers.append(norm(channels[i + 1]))
            self.encoder.append(nn.Sequential(*layers))
        for i in range(depth - 1, -1, -1):
            in_channels = channels[i + 1] * (1 if i == depth - 1 else 2)
            layers = [nn.ReLU(), nn.ConvTranspose2d(in_channels, channels[i], 4, 2, 1)]
            layers.append(norm(channels[i]) if i else nn.Tanh())
            self.decoder.append(nn.Sequential(*layers))

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        for i, layer in enumerate(self.decoder):
            if i:
                x = torch.cat((x, skips[-1 - i]), dim=1)
            x = layer(x)
        return x


class RefineDown(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(incoming, incoming, 4, 2, 3, dilation=2),
            norm(incoming),
            nn.LeakyReLU(0.2),
        )
        self.channel = nn.Conv2d(incoming, outgoing, 3, 1, 1)
        self.normalization = norm(outgoing)

    def forward(self, x: Tensor) -> Tensor:
        return self.channel(self.spatial(x))


class RefineUp(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__(
            nn.ReLU(),
            nn.ConvTranspose2d(incoming * 2, outgoing, 3, 1, 1),
            norm(outgoing),
            nn.ReLU(),
            nn.ConvTranspose2d(outgoing, outgoing, 4, 2, 1),
            norm(outgoing),
        )


class RefinementNetwork(nn.Module):
    def __init__(self, width: int = 64, depth: int = 8, chunk_size: int = 256):
        super().__init__()
        channels = channels_for(width, depth)
        self.stem = nn.Conv2d(6, width, 3, 1, 1)
        self.encoder = nn.ModuleList(
            [RefineDown(channels[i - 1], channels[i]) for i in range(1, depth)]
        )
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2),
            # Modern checkpoints use a dense bottleneck. The released code used
            # dilation=2/padding=3 here; do not silently change existing weights'
            # sampling geometry (identical parameter shapes hide that change).
            nn.Conv2d(channels[-1], channels[-1], 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels[-1], channels[-1], 4, 2, 1),
            norm(channels[-1]),
        )
        self.decoder = nn.ModuleList(
            [RefineUp(channels[i], channels[i - 1]) for i in range(depth - 1, 0, -1)]
        )
        # Original refinement output is linear (the rough network uses tanh).
        self.head = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(width * 2, 3, 3, 1, 1))
        self.attention = CoherentSemanticAttention(chunk_size)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        skips = [x]
        encoded = None
        for level, layer in enumerate(self.encoder, start=1):
            x = layer(x)
            if level == 3:
                x = self.attention(x, mask)
                encoded = x
            x = layer.normalization(x)
            skips.append(x)
        x = self.bottleneck(x)
        decoded = None
        for level, layer in zip(range(len(skips) - 1, 0, -1), self.decoder, strict=True):
            if level == 3:
                decoded = x
            x = layer(torch.cat((x, skips[level]), dim=1))
        prediction = self.head(torch.cat((x, skips[0]), dim=1))
        assert encoded is not None and decoded is not None
        return prediction, encoded, decoded


@dataclass
class InpaintingOutput:
    masked: Tensor
    rough: Tensor
    refined: Tensor
    completed: Tensor
    encoder_features: Tensor
    decoder_features: Tensor
    feature_mask: Tensor


class InpaintingModel(nn.Module):
    def __init__(
        self,
        width: int = 64,
        depth: int = 8,
        attention_chunk_size: int = 256,
        mask_threshold: float = 5 / 16,
    ):
        super().__init__()
        if depth < 4 or width < 1:
            raise ValueError("depth must be >= 4 and width must be positive")
        self.width, self.depth = width, depth
        self.mask_threshold = mask_threshold
        self.rough = RoughNetwork(width, depth)
        self.refine = RefinementNetwork(width, depth, attention_chunk_size)
        self.register_buffer("fill", torch.tensor([123, 104, 117]).view(1, 3, 1, 1) / 127.5 - 1)
        self.apply(initialize)

    def forward(self, image: Tensor, mask: Tensor) -> InpaintingOutput:
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Expected floating RGB images B x 3 x H x W in [-1, 1]")
        divisor = 2**self.depth
        if any(s < divisor or s % divisor for s in image.shape[-2:]):
            raise ValueError(f"Image dimensions must be positive multiples of {divisor}")
        mask = validate_mask(mask, image)
        latent_mask = feature_mask(mask, threshold=self.mask_threshold)
        masked = torch.where(mask, self.fill.to(image.dtype), image)
        rough = self.rough(masked)
        # Match the released training graph: rough is supervised by its own L1,
        # while refinement receives a fixed rough prediction for this batch.
        # Explicit detach preserves that boundary without legacy .data mutation.
        composite = torch.where(mask, rough.detach(), image)
        refined, encoder, decoder = self.refine(torch.cat((composite, masked), dim=1), latent_mask)
        return InpaintingOutput(
            masked,
            rough,
            refined,
            torch.where(mask, refined, image),
            encoder,
            decoder,
            latent_mask,
        )


class PatchDiscriminator(nn.Sequential):
    def __init__(self, width: int = 64):
        super().__init__(
            nn.Conv2d(3, width, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width, width * 2, 4, 2, 1),
            norm(width * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1),
            norm(width * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width * 4, width * 8, 4, 1, 1),
            norm(width * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width * 8, 1, 4, 1, 1),
        )
        self.apply(initialize)


class FeaturePatchDiscriminator(nn.Sequential):
    def __init__(self, in_channels: int = 256, width: int = 512):
        # Released implementation: pool3 input and three stride-2 convolutions.
        super().__init__(
            nn.Conv2d(in_channels, width, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width, width, 4, 2, 1),
            nn.InstanceNorm2d(width),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width, width, 4, 2, 1),
        )
        self.apply(initialize)


class VGGFeatures(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.layers = vgg16(weights=weights).features[:23]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def forward(self, image: Tensor, *, pool3_only: bool = False) -> dict[str, Tensor]:
        x = ((image + 1) / 2 - self.mean) / self.std
        output = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 16:
                output["pool3"] = x
                if pool3_only:
                    return output
            elif i == 22:
                output["relu4_3"] = x
        return output

    def pool3_features(self, image: Tensor) -> Tensor:
        """Feature-GAN fake inputs need no relu4_3 activations or computation."""
        return self(image, pool3_only=True)["pool3"]
