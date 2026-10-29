"""One discriminator update and one generator update, with resumable state."""

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from .config import Config
from .diagnostics import TinyFeatures
from .losses import (
    consistency,
    reconstruction,
    relativistic_discriminator,
    relativistic_generator,
)
from .networks import (
    FeaturePatchDiscriminator,
    InpaintingModel,
    PatchDiscriminator,
    VGGFeatures,
)

TRAINING_SEMANTICS = "rough_detached_separate_adam_v1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Supported devices: cpu, cuda[:index], mps")
    return device


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") not in {1, 2}:
        raise ValueError("Unsupported checkpoint; legacy weights are not interchangeable")
    return checkpoint


class Trainer:
    def __init__(
        self,
        config: Config,
        *,
        test_fixture: bool = False,
        restoring: bool = False,
        diagnostic: Literal["tiny", "random_vgg"] | None = None,
    ):
        config.validate()
        if config.train.lpips_weight > 0:
            raise ValueError(
                "LPIPS loss requires Stage1Trainer or Stage2Trainer; see docs/TRAINING.md"
            )
        if diagnostic not in {None, "tiny", "random_vgg"}:
            raise ValueError("diagnostic must be 'tiny', 'random_vgg', or None")
        if test_fixture:
            if diagnostic not in {None, "tiny"}:
                raise ValueError("test_fixture selects tiny diagnostic features")
            diagnostic = "tiny"
        if diagnostic != "tiny" and config.model.width != 64:
            raise ValueError("Real VGG consistency supervision requires model.width=64")
        self.config = config
        self.test_fixture = diagnostic is not None
        self.feature_mode = {
            None: "imagenet_vgg",
            "tiny": "tiny_diagnostic",
            "random_vgg": "random_vgg_diagnostic",
        }[diagnostic]
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
        self.generator = InpaintingModel(**asdict(config.model)).to(self.device)
        self.image_d = PatchDiscriminator(config.train.discriminator_width).to(self.device)
        feature_channels = config.model.width * 8 if diagnostic == "tiny" else 256
        self.feature_d = FeaturePatchDiscriminator(
            feature_channels, config.train.feature_discriminator_width
        ).to(self.device)
        self.features = (
            (
                TinyFeatures(feature_channels)
                if diagnostic == "tiny"
                else VGGFeatures(pretrained=not restoring and diagnostic is None)
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        t = config.train
        self.g_optimizer = torch.optim.Adam(
            self.generator.refine.parameters(), lr=t.lr, betas=(t.beta1, t.beta2)
        )
        self.p_optimizer = torch.optim.Adam(
            self.generator.rough.parameters(), lr=t.lr, betas=(t.beta1, t.beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            [*self.image_d.parameters(), *self.feature_d.parameters()],
            lr=t.lr,
            betas=(t.beta1, t.beta2),
        )

        def decay(epoch: int) -> float:
            return max(0.0, 1 - max(0, epoch - t.decay_start) / (t.epochs - t.decay_start))

        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, decay)
        self.p_scheduler = torch.optim.lr_scheduler.LambdaLR(self.p_optimizer, decay)
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp == "fp16")
        self.global_step = 0
        self.completed_epochs = 0

    def autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp != "off")

    def _require_ready(self) -> None:
        if self._restore_pending:
            raise RuntimeError(
                "Trainer is awaiting a successful restore(); training or saving now would "
                "use uninitialized checkpoint state"
            )

    def step(self, image: Tensor, mask: Tensor) -> dict[str, float]:
        self._require_ready()
        image = image.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        self.generator.train()
        self.image_d.train().requires_grad_(True)
        self.feature_d.train().requires_grad_(True)
        self.g_optimizer.zero_grad(set_to_none=True)
        self.p_optimizer.zero_grad(set_to_none=True)
        self.d_optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            output = self.generator(image, mask)
        # Frozen VGG weights do not imply no_grad for generated images.
        with torch.no_grad():
            real_features = self.features(image.float())
        fake_features = self.features(output.refined.float())
        with self.autocast():
            d_image = relativistic_discriminator(
                self.image_d(image), self.image_d(output.refined.detach())
            )
            d_feature = relativistic_discriminator(
                self.feature_d(real_features["pool3"]),
                self.feature_d(fake_features["pool3"].detach()),
            )
            d_loss = (d_image + d_feature) / 2
        self._backward_step(d_loss, self.d_optimizer)
        self.d_optimizer.zero_grad(set_to_none=True)
        self.image_d.requires_grad_(False)
        self.feature_d.requires_grad_(False)
        with self.autocast():
            with torch.no_grad():
                real_logits = self.image_d(image)
                real_feature_logits = self.feature_d(real_features["pool3"])
            g_image = relativistic_generator(real_logits, self.image_d(output.refined))
            g_feature = relativistic_generator(
                real_feature_logits, self.feature_d(fake_features["pool3"])
            )
        rec = reconstruction(output, image)
        con = consistency(output, real_features["relu4_3"])
        t = self.config.train
        g_loss = t.reconstruction_weight * rec + t.consistency_weight * con
        g_loss = g_loss + t.adversarial_weight * (g_image + g_feature)
        # The detached rough-to-refine input makes these two loss graphs
        # independent. Backpropagate once, then update both networks using the
        # same batch's forward pass, as in the released master implementation.
        self._backward_step(g_loss, self.g_optimizer, self.p_optimizer)
        self.scaler.update()
        self.image_d.requires_grad_(True)
        self.feature_d.requires_grad_(True)
        self.global_step += 1
        losses = {
            "generator": g_loss,
            "discriminator": d_loss,
            "reconstruction": rec,
            "consistency": con,
            "adversarial_image": g_image,
            "adversarial_feature": g_feature,
        }
        return {name: value.detach().item() for name, value in losses.items()}

    def _backward_step(self, loss: Tensor, *optimizers: torch.optim.Optimizer) -> None:
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss at step {self.global_step}")
        self.scaler.scale(loss).backward()
        for optimizer in optimizers:
            self.scaler.unscale_(optimizer)
        # In fp16 GradScaler handles overflow by skipping the update. Otherwise
        # fail before corrupting weights or optimizer moments.
        if not self.scaler.is_enabled():
            for optimizer in optimizers:
                nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    float("inf"),
                    error_if_nonfinite=True,
                )
        for optimizer in optimizers:
            self.scaler.step(optimizer)

    def finish_epoch(self) -> None:
        self._require_ready()
        self.g_scheduler.step()
        self.p_scheduler.step()
        self.d_scheduler.step()
        self.completed_epochs += 1

    def save(self, path: str | Path) -> None:
        self._require_ready()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 2,
            "training_semantics": TRAINING_SEMANTICS,
            "config": self.config.as_dict(),
            "test_fixture": self.test_fixture,
            "feature_mode": self.feature_mode,
            "torch_version": str(torch.__version__),
            "generator": self.generator.state_dict(),
            "image_d": self.image_d.state_dict(),
            "feature_d": self.feature_d.state_dict(),
            "features": self.features.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "p_optimizer": self.p_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_scheduler": self.g_scheduler.state_dict(),
            "p_scheduler": self.p_scheduler.state_dict(),
            "d_scheduler": self.d_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "completed_epochs": self.completed_epochs,
            "rng_cpu": torch.get_rng_state(),
            "rng_python": random.getstate(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "rng_mps": torch.mps.get_rng_state() if self.device.type == "mps" else None,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def restore(self, path: str | Path) -> None:
        # Any failure leaves this instance unusable until a full restore succeeds;
        # otherwise partially loaded optimizer/model/RNG state could escape.
        self._restore_pending = True
        state = load_checkpoint(path)
        if state["format_version"] != 2 or state.get("training_semantics") != TRAINING_SEMANTICS:
            raise ValueError(
                "Checkpoint training semantics differ: exact resume requires detached "
                "rough-to-refine gradients and separate generator optimizers. "
                "Start a new training run; older modern checkpoints remain inference-compatible."
            )
        # Changes to placement/logging/data paths are okay; optimization and
        # architecture changes are not an exact resume and should be explicit.
        saved = Config.from_dict(state["config"])
        ignored = {"device", "output", "log_every"}
        old_train = {k: v for k, v in asdict(saved.train).items() if k not in ignored}
        new_train = {k: v for k, v in asdict(self.config.train).items() if k not in ignored}
        old_data, new_data = asdict(saved.data), asdict(self.config.data)
        for key in ("train_images", "val_images", "train_masks", "val_masks", "workers"):
            old_data.pop(key)
            new_data.pop(key)
        if (
            saved.model != self.config.model
            or old_train != new_train
            or old_data != new_data
            or state.get(
                "feature_mode",
                "tiny_diagnostic" if state["test_fixture"] else "imagenet_vgg",
            )
            != self.feature_mode
        ):
            raise ValueError("Resume config differs from the saved training/model/data settings")
        if (
            self.device.type == "cuda"
            and state["rng_cuda"]
            and len(state["rng_cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError(
                "Cannot restore CUDA RNG state across different GPU counts: "
                f"checkpoint has {len(state['rng_cuda'])}, "
                f"current host has {torch.cuda.device_count()}"
            )
        for name in (
            "generator",
            "image_d",
            "feature_d",
            "features",
            "g_optimizer",
            "p_optimizer",
            "d_optimizer",
            "g_scheduler",
            "p_scheduler",
            "d_scheduler",
            "scaler",
        ):
            getattr(self, name).load_state_dict(state[name])
        self.global_step = state["global_step"]
        self.completed_epochs = state["completed_epochs"]
        torch.set_rng_state(state["rng_cpu"])
        random.setstate(state["rng_python"])
        if self.device.type == "cuda" and state["rng_cuda"]:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        if self.device.type == "mps" and state["rng_mps"] is not None:
            torch.mps.set_rng_state(state["rng_mps"])
        self._restore_pending = False


def load_generator(path: str | Path, device: torch.device) -> tuple[InpaintingModel, Config]:
    from .inference import load_inference_generator

    return load_inference_generator(path, device)
