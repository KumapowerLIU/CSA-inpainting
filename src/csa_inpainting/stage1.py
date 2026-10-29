"""Standalone rough-network training: L1 with optional completed-image LPIPS."""

import copy
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import Config
from .engine import choose_device
from .masks import validate_mask
from .networks import RoughNetwork, initialize
from .perceptual import LPIPSLoss

TRAINING_SEMANTICS = "stage1_rough_l1_v1"
LPIPS_TRAINING_SEMANTICS = "stage1_rough_l1_completed_lpips_v1"
CHECKPOINT_FORMAT = "stage1_v1"


@dataclass
class RoughOutput:
    masked: Tensor
    rough: Tensor
    completed: Tensor


class RoughInpaintingModel(nn.Module):
    """Only the original rough U-Net; state keys retain the ``rough.`` prefix."""

    def __init__(self, width: int = 64, depth: int = 8):
        super().__init__()
        if depth < 4 or width < 1:
            raise ValueError("depth must be >= 4 and width must be positive")
        self.width, self.depth = width, depth
        self.rough = RoughNetwork(width, depth)
        self.register_buffer("fill", torch.tensor([123, 104, 117]).view(1, 3, 1, 1) / 127.5 - 1)
        self.apply(initialize)

    def forward(self, image: Tensor, mask: Tensor) -> RoughOutput:
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Expected floating RGB images B x 3 x H x W in [-1, 1]")
        divisor = 2**self.depth
        if any(size < divisor or size % divisor for size in image.shape[-2:]):
            raise ValueError(f"Image dimensions must be positive multiples of {divisor}")
        mask = validate_mask(mask, image)
        masked = torch.where(mask, self.fill.to(image.dtype), image)
        rough = self.rough(masked)
        return RoughOutput(masked, rough, torch.where(mask, rough, image))


def load_stage1_checkpoint(path: str | Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("format_version") != CHECKPOINT_FORMAT
        or state.get("kind") != "csa_stage1"
        or state.get("stage") != 1
        or state.get("training_semantics") not in {TRAINING_SEMANTICS, LPIPS_TRAINING_SEMANTICS}
    ):
        raise ValueError(
            "Expected a stage-1 rough-only checkpoint with matching training semantics"
        )
    config = Config.from_dict(state["config"])
    expected = LPIPS_TRAINING_SEMANTICS if config.train.lpips_weight > 0 else TRAINING_SEMANTICS
    if state["training_semantics"] != expected:
        raise ValueError("Stage-1 checkpoint loss config disagrees with its training semantics")
    return state


class Stage1Trainer:
    feature_mode = "none_stage1"
    training_semantics = TRAINING_SEMANTICS

    def __init__(self, config: Config, *, restoring: bool = False):
        config.validate()
        self.config = config
        self._restore_pending = restoring
        self.device = choose_device(config.train.device)
        self.amp = config.train.amp
        if self.amp != "off" and self.device.type != "cuda":
            raise ValueError("AMP is supported on CUDA only; use amp: 'off' on CPU/MPS")
        if self.amp == "bf16":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError(f"CUDA device {self.device} does not support bf16")
        self.amp_dtype = torch.bfloat16 if self.amp == "bf16" else torch.float16
        self.generator = RoughInpaintingModel(config.model.width, config.model.depth).to(
            self.device
        )
        train = config.train
        self.perceptual = None
        if train.lpips_weight > 0:
            self.training_semantics = LPIPS_TRAINING_SEMANTICS
            self.feature_mode = f"lpips_{train.lpips_net}_v0.1_stage1"
            self.perceptual = LPIPSLoss(train.lpips_net, train.lpips_batch_size).to(self.device)
        self.lpips_provenance = self.perceptual.provenance if self.perceptual is not None else None
        self.p_optimizer = torch.optim.Adam(
            self.generator.rough.parameters(), lr=train.lr, betas=(train.beta1, train.beta2)
        )

        def decay(epoch: int) -> float:
            return max(
                0.0,
                1 - max(0, epoch - train.decay_start) / (train.epochs - train.decay_start),
            )

        self.p_scheduler = torch.optim.lr_scheduler.LambdaLR(self.p_optimizer, decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp == "fp16")
        self.global_step = 0
        self.completed_epochs = 0
        # The data runner owns this cursor; optimizer steps alone cannot locate
        # a sample after a batch-size change.
        self.data_progress: dict | None = None
        self.loss_transitions: list[dict] = []

    def autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp != "off")

    def _require_ready(self) -> None:
        if self._restore_pending:
            raise RuntimeError("Stage1Trainer is awaiting a successful restore()")

    def step(self, image: Tensor, mask: Tensor) -> dict[str, float]:
        self._require_ready()
        image = image.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        self.generator.train()
        self.p_optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            output = self.generator(image, mask)
        reconstruction = F.l1_loss(output.rough.float(), image.float())
        loss = reconstruction * self.config.train.reconstruction_weight
        perceptual = None
        if self.perceptual is not None:
            perceptual = self.perceptual(output.completed, image)
            loss = loss + self.config.train.lpips_weight * perceptual
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite stage-1 loss at step {self.global_step}")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.p_optimizer)
        if not self.scaler.is_enabled():
            nn.utils.clip_grad_norm_(
                self.generator.parameters(), float("inf"), error_if_nonfinite=True
            )
        self.scaler.step(self.p_optimizer)
        self.scaler.update()
        self.global_step += 1
        result = {
            "generator": loss.detach().item(),
            "reconstruction": reconstruction.detach().item(),
        }
        if perceptual is not None:
            result["reconstruction_weighted"] = (
                self.config.train.reconstruction_weight * result["reconstruction"]
            )
            result["lpips"] = perceptual.detach().item()
            result["lpips_weighted"] = self.config.train.lpips_weight * result["lpips"]
        return result

    def finish_epoch(self) -> None:
        self._require_ready()
        self.p_scheduler.step()
        self.completed_epochs += 1

    def save(self, path: str | Path) -> None:
        self._require_ready()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": CHECKPOINT_FORMAT,
            "kind": "csa_stage1",
            "stage": 1,
            "training_semantics": self.training_semantics,
            "feature_mode": self.feature_mode,
            "config": self.config.as_dict(),
            "torch_version": str(torch.__version__),
            "generator": self.generator.state_dict(),
            "p_optimizer": self.p_optimizer.state_dict(),
            "p_scheduler": self.p_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "completed_epochs": self.completed_epochs,
            "data_progress": self.data_progress,
            "loss_transitions": self.loss_transitions,
            "lpips_provenance": self.lpips_provenance,
            "rng_cpu": torch.get_rng_state(),
            "rng_python": random.getstate(),
            "rng_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "rng_mps": torch.mps.get_rng_state() if self.device.type == "mps" else None,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def restore(
        self,
        path: str | Path,
        *,
        allow_batch_size_change: bool = False,
        allow_loss_change: bool = False,
    ) -> None:
        """Restore state; batch, LPIPS enabling and coefficient changes need opt-in.

        Changing batch size changes the optimization trajectory. The caller is
        responsible for continuing the saved sample cursor in ``data_progress``.
        """
        self._restore_pending = True
        state = load_stage1_checkpoint(path)
        saved = Config.from_dict(state["config"])
        ignored_train = {"device", "output", "log_every"}
        enable_lpips = (
            state["training_semantics"] == TRAINING_SEMANTICS
            and self.training_semantics == LPIPS_TRAINING_SEMANTICS
        )
        coefficient_change = state[
            "training_semantics"
        ] == self.training_semantics == LPIPS_TRAINING_SEMANTICS and (
            saved.train.reconstruction_weight != self.config.train.reconstruction_weight
            or saved.train.lpips_weight != self.config.train.lpips_weight
        )
        if allow_loss_change and enable_lpips:
            ignored_train.update({"lpips_weight", "lpips_net", "lpips_batch_size"})
        elif allow_loss_change and coefficient_change:
            ignored_train.update({"reconstruction_weight", "lpips_weight"})
        old_train = {k: v for k, v in asdict(saved.train).items() if k not in ignored_train}
        new_train = {k: v for k, v in asdict(self.config.train).items() if k not in ignored_train}
        old_data, new_data = asdict(saved.data), asdict(self.config.data)
        for key in ("train_images", "val_images", "train_masks", "val_masks", "workers"):
            old_data.pop(key)
            new_data.pop(key)
        if allow_batch_size_change:
            old_data.pop("batch_size")
            new_data.pop("batch_size")
        if (
            saved.model != self.config.model
            or old_train != new_train
            or old_data != new_data
            or (
                state.get("feature_mode") != self.feature_mode
                and not (
                    allow_loss_change
                    and enable_lpips
                    and state.get("feature_mode") == "none_stage1"
                )
            )
            or (
                state["training_semantics"] != self.training_semantics
                and not (allow_loss_change and enable_lpips)
            )
            or (
                state["training_semantics"] == LPIPS_TRAINING_SEMANTICS
                and state.get("lpips_provenance") != self.lpips_provenance
            )
        ):
            raise ValueError("Resume config differs from the saved stage-1 settings")
        if (
            self.device.type == "cuda"
            and state["rng_cuda"]
            and len(state["rng_cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Cannot restore stage-1 CUDA RNG state across different GPU counts")
        for name in ("generator", "p_optimizer", "p_scheduler", "scaler"):
            getattr(self, name).load_state_dict(state[name])
        self.global_step = state["global_step"]
        self.completed_epochs = state["completed_epochs"]
        self.data_progress = state.get("data_progress")
        self.loss_transitions = copy.deepcopy(state.get("loss_transitions", []))
        if enable_lpips or coefficient_change:
            loss_keys = ("reconstruction_weight", "lpips_weight", "lpips_net", "lpips_batch_size")
            self.loss_transitions.append(
                {
                    "change_kind": "enable_lpips" if enable_lpips else "loss_coefficients",
                    "parent_checkpoint": str(Path(path).resolve()),
                    "parent_global_step": self.global_step,
                    "parent_training_semantics": state["training_semantics"],
                    "training_semantics": self.training_semantics,
                    "parent_feature_mode": state["feature_mode"],
                    "feature_mode": self.feature_mode,
                    "parent_loss_config": {key: getattr(saved.train, key) for key in loss_keys},
                    "loss_config": {key: getattr(self.config.train, key) for key in loss_keys},
                }
            )
        torch.set_rng_state(state["rng_cpu"])
        random.setstate(state["rng_python"])
        if self.device.type == "cuda" and state["rng_cuda"]:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        if self.device.type == "mps" and state["rng_mps"] is not None:
            torch.mps.set_rng_state(state["rng_mps"])
        self._restore_pending = False


def load_stage1_generator(path: str | Path, device: torch.device):
    state = load_stage1_checkpoint(path)
    config = Config.from_dict(state["config"])
    model = RoughInpaintingModel(config.model.width, config.model.depth).to(device)
    model.load_state_dict(state["generator"])
    return model.eval(), config
