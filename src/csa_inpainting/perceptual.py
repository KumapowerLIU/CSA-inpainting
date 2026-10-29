"""Frozen, FP32 LPIPS v0.1 with bounded feature-activation memory."""

import hashlib
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def _weights_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class LPIPSLoss(nn.Module):
    """Average calibrated LPIPS over images already normalized to [-1, 1].

    Import and pretrained-weight loading occur only on construction. Chunking
    plus activation recomputation avoids retaining every image's VGG features
    until the generator backward pass. The network is frozen, but gradients to
    ``prediction`` remain enabled.
    """

    def __init__(self, net: str = "vgg", batch_size: int = 8, *, restoring: bool = False):
        super().__init__()
        if batch_size < 1:
            raise ValueError("LPIPS batch_size must be positive")
        try:
            import lpips
        except ImportError as error:
            raise ImportError(
                "LPIPS requires lpips==0.1.4; install csa-inpainting[perceptual]"
            ) from error
        self.network = (
            lpips.LPIPS(
                net=net,
                version="0.1",
                pretrained=not restoring,
                pnet_rand=restoring,
                pnet_tune=False,
                eval_mode=True,
                verbose=False,
            )
            .float()
            .eval()
            .requires_grad_(False)
        )
        self.batch_size = batch_size
        self.net = net
        self._restore_pending = restoring
        self.provenance = self._provenance(pretrained=not restoring)

    def _provenance(self, *, pretrained: bool) -> dict:
        try:
            package_version = version("lpips")
        except PackageNotFoundError:
            package_version = "unknown"
        weights = self.network.state_dict()
        return {
            "package": "lpips",
            "package_version": package_version,
            "net": self.net,
            "version": "0.1",
            "pretrained": pretrained,
            "pnet_rand": not pretrained,
            "pnet_tune": False,
            "normalize": False,
            "feature_weights_sha256": _weights_sha256(
                {key: value for key, value in weights.items() if key.startswith("net.")}
            ),
            "metric_weights_sha256": _weights_sha256(
                {key: value for key, value in weights.items() if not key.startswith("net.")}
            ),
        }

    def restore_checkpoint(self, state: dict, provenance: dict) -> None:
        """Restore a self-contained metric without fetching pretrained assets."""
        self._restore_pending = True
        # A different implementation may produce different features even with
        # identical tensors, so package/network metadata must also match.
        current = self._provenance(pretrained=True)
        hash_keys = {"feature_weights_sha256", "metric_weights_sha256"}
        if (
            not isinstance(provenance, dict)
            or set(provenance) != set(current)
            or any(provenance[key] != current[key] for key in current if key not in hash_keys)
        ):
            raise ValueError("LPIPS provenance differs from the checkpoint implementation")
        self.load_state_dict(state, strict=True)
        restored = self._provenance(pretrained=True)
        if restored != provenance:
            raise ValueError("LPIPS provenance hash does not match the saved metric weights")
        self.network.eval().requires_grad_(False)
        self.provenance = restored
        self._restore_pending = False

    def train(self, mode: bool = True):
        # LPIPS calibration layers contain dropout. Their inference behavior
        # is part of the perceptual metric, even while the generator trains.
        super().train(False)
        return self

    def _chunk(self, prediction: Tensor, target: Tensor) -> Tensor:
        with torch.autocast(prediction.device.type, enabled=False):
            return self.network(prediction.float(), target.float(), normalize=False).mean()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self._restore_pending:
            raise RuntimeError("LPIPS is awaiting checkpoint weights")
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[0] < 1:
            raise ValueError("LPIPS expects matching nonempty B x C x H x W tensors")
        self.eval()
        total = prediction.new_zeros((), dtype=torch.float32)
        count = prediction.shape[0]
        for start in range(0, count, self.batch_size):
            predicted_chunk = prediction[start : start + self.batch_size]
            target_chunk = target[start : start + self.batch_size]
            if torch.is_grad_enabled() and (
                predicted_chunk.requires_grad or target_chunk.requires_grad
            ):
                value = checkpoint(self._chunk, predicted_chunk, target_chunk, use_reentrant=False)
            else:
                value = self._chunk(predicted_chunk, target_chunk)
            total = total + value * (predicted_chunk.shape[0] / count)
        return total
