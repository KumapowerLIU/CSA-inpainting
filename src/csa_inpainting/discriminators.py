"""Real-ESRGAN's RGB U-Net discriminator and strict pretrained initialization.

Architecture adapted from xinntao/Real-ESRGAN, discriminator_arch.py,
Git blob 4b66ab1226d6793de846bc9828bbe427031a0e2d. The BasicSR registry
dependency is omitted; convolution names, operations and spectral-normalization
state keys are retained. Inputs use the upstream RGB [0, 1] convention.

BSD 3-Clause License
Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import hashlib
import io
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .perceptual import _weights_sha256

REALESRGAN_D_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth"
)
REALESRGAN_D_ARCHITECTURE = "realesrgan_unet_sn_v1"
REALESRGAN_D_SHA256 = "afc291719bf0a8bd25ba2b4afbbc5fc0634382dac111d3e1d3a7240d32dfd33a"


class UNetDiscriminatorSN(nn.Module):
    """Upstream architecture, returning B x 1 x H x W raw realness logits.

    The official pretrained network has num_in_ch=3, num_feat=64 and skip
    connections enabled. Smaller widths support isolated diagnostic tests.
    Normalization to [0, 1] belongs at the training caller, not inside this
    module, so loading upstream weights preserves the original forward graph.
    """

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, skip_connection: bool = True):
        super().__init__()
        if num_in_ch != 3 or num_feat < 1 or not isinstance(skip_connection, bool):
            raise ValueError("Real-ESRGAN D requires RGB, positive width and boolean skips")
        self.num_in_ch = num_in_ch
        self.num_feat = num_feat
        self.skip_connection = skip_connection
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Real-ESRGAN D expects floating B x 3 x H x W RGB images")
        if any(size < 8 or size % 8 for size in image.shape[-2:]):
            raise ValueError("Real-ESRGAN D image dimensions must be multiples of 8")
        x0 = F.leaky_relu(self.conv0(image), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        return self.conv9(out)


def discriminator_architecture(module: UNetDiscriminatorSN) -> dict:
    return {
        "architecture": REALESRGAN_D_ARCHITECTURE,
        "num_in_ch": module.num_in_ch,
        "num_feat": module.num_feat,
        "skip_connection": module.skip_connection,
        "input_range": [0, 1],
    }


def validate_discriminator_provenance(provenance: dict, module: UNetDiscriminatorSN) -> None:
    """Validate initialization metadata without comparing it to evolved weights."""
    expected = discriminator_architecture(module)
    keys = {*expected, "source", "checkpoint", "file_sha256", "initial_state_sha256", "param_key"}
    if (
        not isinstance(provenance, dict)
        or set(provenance) != keys
        or any(provenance[key] != value for key, value in expected.items())
        or provenance.get("param_key") != "params"
        or any(
            not isinstance(provenance.get(key), str) or not provenance[key]
            for key in ("source", "checkpoint")
        )
        or any(
            not isinstance(provenance.get(key), str)
            or re.fullmatch(r"[0-9a-f]{64}", provenance[key]) is None
            for key in ("file_sha256", "initial_state_sha256")
        )
    ):
        raise ValueError("Invalid Real-ESRGAN discriminator initialization provenance")


def load_pretrained_discriminator(
    module: UNetDiscriminatorSN,
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    source: str = REALESRGAN_D_URL,
) -> dict:
    """Strictly load an upstream ``params`` checkpoint and record its identity.

    A single byte snapshot is used for hashing and weights-only deserialization.
    This loader does not download assets or accept partial/mismatched tensors.
    Missing keys, shapes, dtypes or nonfinite weights fail before model mutation.
    """
    path = Path(path).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Real-ESRGAN pretrained discriminator SHA256 mismatch")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not isinstance(state.get("params"), Mapping):
        raise ValueError("Real-ESRGAN discriminator checkpoint requires a params mapping")
    weights = state["params"]
    current = module.state_dict()
    if set(weights) != set(current):
        raise ValueError("Real-ESRGAN discriminator parameter keys do not match")
    for name, target in current.items():
        value = weights[name]
        if (
            not isinstance(value, Tensor)
            or value.shape != target.shape
            or value.dtype != target.dtype
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Invalid Real-ESRGAN discriminator tensor: {name}")
    provenance = {
        **discriminator_architecture(module),
        "source": source,
        "checkpoint": str(path),
        "file_sha256": digest,
        "initial_state_sha256": _weights_sha256(weights),
        "param_key": "params",
    }
    validate_discriminator_provenance(provenance, module)
    module.load_state_dict(weights, strict=True)
    return provenance
