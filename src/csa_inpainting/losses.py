from torch import Tensor
from torch.nn import functional as F

from .networks import InpaintingOutput


def relativistic_discriminator(real: Tensor, fake: Tensor) -> Tensor:
    """RaLSGAN D objective; average over every batch/patch score, in FP32."""
    real, fake = real.float(), fake.float()
    real_relative = real - fake.mean()
    fake_relative = fake - real.mean()
    return ((real_relative - 1).square().mean() + (fake_relative + 1).square().mean()) / 2


def relativistic_generator(real: Tensor, fake: Tensor) -> Tensor:
    """RaLSGAN G objective; both terms must retain gradients through fake."""
    real, fake = real.float(), fake.float()
    real_relative = real - fake.mean()
    fake_relative = fake - real.mean()
    return ((real_relative + 1).square().mean() + (fake_relative - 1).square().mean()) / 2


def reconstruction(output: InpaintingOutput, target: Tensor) -> Tensor:
    return F.l1_loss(output.rough.float(), target.float()) + F.l1_loss(
        output.refined.float(), target.float()
    )


def consistency(output: InpaintingOutput, target: Tensor) -> Tensor:
    if (
        output.encoder_features.shape != target.shape
        or output.decoder_features.shape != target.shape
    ):
        raise ValueError("Consistency features must match VGG relu4_3 (use model width=64)")
    mask = output.feature_mask.float()
    # Mean over the whole feature grid, matching released MSE reduction; both
    # sides masked, so visible pixels do not create a constant loss offset.
    return sum(
        ((features.float() - target.float()).square() * mask).mean()
        for features in (output.encoder_features, output.decoder_features)
    )
