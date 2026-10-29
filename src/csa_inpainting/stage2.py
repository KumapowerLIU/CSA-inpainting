"""Refinement training initialized from a frozen, separately trained rough model."""

import copy
import hashlib
import os
import random
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torch.utils.checkpoint import checkpoint

from .config import Config
from .diagnostics import TinyFeatures
from .discriminators import (
    REALESRGAN_D_SHA256,
    REALESRGAN_D_URL,
    UNetDiscriminatorSN,
    load_pretrained_discriminator,
    validate_discriminator_provenance,
)
from .engine import choose_device
from .losses import consistency, relativistic_discriminator, relativistic_generator
from .masks import feature_mask, validate_mask
from .networks import (
    FeaturePatchDiscriminator,
    InpaintingModel,
    InpaintingOutput,
    PatchDiscriminator,
    VGGFeatures,
)
from .perceptual import LPIPSLoss, _weights_sha256
from .stage1 import load_stage1_checkpoint

CHECKPOINT_FORMAT = "stage2_v1"
TRAINING_SEMANTICS = "stage2_frozen_rough_tanh_refine_l1_completed_lpips_csa_ralsgan_v1"
REALESRGAN_TRAINING_SEMANTICS = (
    "stage2_frozen_rough_tanh_refine_l1_completed_lpips_csa_ralsgan_realesrgan_unet_sn_v1"
)
SPECTRAL_NORM_POLICY = "fp32_once_before_each_d_and_g_phase_frozen_chunk_replay_v1"


def stage2_training_semantics(config: Config) -> str:
    return (
        REALESRGAN_TRAINING_SEMANTICS
        if config.train.image_discriminator == "realesrgan_unet_sn"
        else TRAINING_SEMANTICS
    )


class Stage2InpaintingModel(InpaintingModel):
    """Frozen rough U-Net and a bounded refinement output; the bottleneck stays dense."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.refine.head.add_module("tanh", nn.Tanh())
        self.rough.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Parent train()/eval() calls must never unfreeze rough behavior.
        self.rough.requires_grad_(False).eval()
        return self

    def forward(self, image: Tensor, mask: Tensor) -> InpaintingOutput:
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Expected floating RGB images B x 3 x H x W in [-1, 1]")
        divisor = 2**self.depth
        if any(size < divisor or size % divisor for size in image.shape[-2:]):
            raise ValueError(f"Image dimensions must be positive multiples of {divisor}")
        mask = validate_mask(mask, image)
        latent_mask = feature_mask(mask, threshold=self.mask_threshold)
        masked = torch.where(mask, self.fill.to(image.dtype), image)
        with torch.no_grad():
            rough = self.rough(masked)
        composite = torch.where(mask, rough, image)
        refined, encoder, decoder = self.refine(torch.cat((composite, masked), 1), latent_mask)
        return InpaintingOutput(
            masked,
            rough,
            refined,
            torch.where(mask, refined, image),
            encoder,
            decoder,
            latent_mask,
        )


def load_stage2_checkpoint(path: str | Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("format_version") != CHECKPOINT_FORMAT
        or state.get("kind") != "csa_stage2"
        or state.get("stage") != 2
        or state.get("output_activation") != "tanh"
        or state.get("rough_frozen") is not True
        or state.get("bottleneck") != "dense"
    ):
        raise ValueError("Expected a stage-2 checkpoint with matching training semantics")
    config = Config.from_dict(state["config"])
    if state.get("training_semantics") != stage2_training_semantics(config):
        raise ValueError("Stage-2 discriminator and training semantics do not match")
    if config.train.image_discriminator == "realesrgan_unet_sn":
        provenance = state.get("image_d_provenance")
        if (
            not isinstance(provenance, dict)
            or set(provenance)
            != {
                "architecture",
                "num_in_ch",
                "num_feat",
                "skip_connection",
                "input_range",
                "source",
                "checkpoint",
                "file_sha256",
                "initial_state_sha256",
                "param_key",
            }
            or provenance.get("architecture") != "realesrgan_unet_sn_v1"
            or provenance.get("num_in_ch") != 3
            or provenance.get("num_feat") != config.train.discriminator_width
            or provenance.get("skip_connection") is not True
            or provenance.get("input_range") != [0, 1]
            or provenance.get("param_key") != "params"
            or state.get("image_d_spectral_norm_policy") != SPECTRAL_NORM_POLICY
            or any(
                not isinstance(provenance.get(key), str) or not provenance[key]
                for key in ("source", "checkpoint")
            )
        ):
            raise ValueError("Invalid pretrained image discriminator provenance or SN policy")
        for key in ("file_sha256", "initial_state_sha256"):
            value = provenance.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("Invalid pretrained image discriminator hash provenance")
        if not state.get("test_fixture") and (
            provenance["file_sha256"] != REALESRGAN_D_SHA256
            or provenance["source"] != REALESRGAN_D_URL
            or provenance["num_feat"] != 64
        ):
            raise ValueError("Production stage 2 requires the verified official discriminator")
        if _weights_sha256(state["image_d"]) != state.get("image_d_state_sha256"):
            raise ValueError("Image discriminator state hash mismatch")
    parent = state.get("parent_provenance")
    if (
        not isinstance(parent, dict)
        or parent.get("stage") != 1
        or parent.get("format_version") != "stage1_v1"
        or not isinstance(parent.get("checkpoint"), str)
        or not parent["checkpoint"]
        or not isinstance(parent.get("sha256"), str)
        or len(parent["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in parent["sha256"])
        or not isinstance(parent.get("global_step"), int)
        or parent["global_step"] < 0
        or not isinstance(parent.get("completed_epochs"), int)
        or parent["completed_epochs"] < 0
    ):
        raise ValueError("Stage-2 checkpoint is missing valid stage-1 parent provenance")
    return state


@contextmanager
def _frozen_parameters(*modules: nn.Module):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, flags, strict=True):
            parameter.requires_grad_(enabled)


class Stage2Trainer:
    training_semantics = TRAINING_SEMANTICS

    def __init__(self, config: Config, *, restoring: bool = False, test_fixture: bool = False):
        config.validate()
        if not test_fixture and config.model.width != 64:
            raise ValueError("Real VGG consistency supervision requires model.width=64")
        self.config = config
        self.training_semantics = stage2_training_semantics(config)
        self.test_fixture = test_fixture
        self.feature_mode = "tiny_diagnostic_stage2" if test_fixture else "imagenet_vgg_stage2"
        self._ready = False
        self._restoring_requested = restoring
        self._restore_attempted = False
        self.device = choose_device(config.train.device)
        self.amp = config.train.amp
        if self.amp != "off" and self.device.type != "cuda":
            raise ValueError("AMP is supported on CUDA only; use amp: 'off' on CPU/MPS")
        if self.amp == "bf16":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError(f"CUDA device {self.device} does not support bf16")
        self.amp_dtype = torch.bfloat16 if self.amp == "bf16" else torch.float16
        self.generator = Stage2InpaintingModel(**asdict(config.model)).to(self.device)
        t = config.train
        self.image_d_provenance = None
        if t.image_discriminator == "realesrgan_unet_sn":
            if not test_fixture and t.discriminator_width != 64:
                raise ValueError("Official Real-ESRGAN discriminator requires width=64")
            self.image_d = UNetDiscriminatorSN(
                num_in_ch=3, num_feat=t.discriminator_width, skip_connection=True
            )
            if not restoring:
                if not t.image_discriminator_pretrained:
                    raise ValueError(
                        "Fresh Real-ESRGAN stage 2 requires pretrained image D weights"
                    )
                self.image_d_provenance = load_pretrained_discriminator(
                    self.image_d,
                    t.image_discriminator_pretrained,
                    expected_sha256=None if test_fixture else REALESRGAN_D_SHA256,
                    source="diagnostic_fixture" if test_fixture else REALESRGAN_D_URL,
                )
            self.image_d.to(self.device).eval()
        else:
            self.image_d = PatchDiscriminator(t.discriminator_width).to(self.device)
        channels = config.model.width * 8 if test_fixture else 256
        self.feature_d = FeaturePatchDiscriminator(channels, t.feature_discriminator_width).to(
            self.device
        )
        self.features = (
            (TinyFeatures(channels) if test_fixture else VGGFeatures(pretrained=not restoring))
            .to(self.device)
            .float()
            .eval()
            .requires_grad_(False)
        )
        self.features_provenance = self._features_provenance()
        self.perceptual = (
            LPIPSLoss(t.lpips_net, t.lpips_batch_size, restoring=restoring).to(self.device)
            if t.lpips_weight > 0
            else None
        )
        self.lpips_provenance = (
            copy.deepcopy(self.perceptual.provenance) if self.perceptual is not None else None
        )
        self.g_optimizer = torch.optim.Adam(
            self.generator.refine.parameters(), lr=t.lr, betas=(t.beta1, t.beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            [*self.image_d.parameters(), *self.feature_d.parameters()],
            lr=t.lr,
            betas=(t.beta1, t.beta2),
        )

        def decay(epoch: int) -> float:
            return max(0.0, 1 - max(0, epoch - t.decay_start) / (t.epochs - t.decay_start))

        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, decay)
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp == "fp16")
        self.global_step = 0
        self.completed_epochs = 0
        self.data_progress: dict | None = None
        self.parent_provenance: dict | None = None

    def autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp != "off")

    def _features_provenance(self) -> dict:
        return {
            "feature_mode": self.feature_mode,
            "weights": "tiny_diagnostic" if self.test_fixture else "VGG16_IMAGENET1K_V1",
            "state_sha256": _weights_sha256(self.features.state_dict()),
        }

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "Stage2Trainer is awaiting a successful init_from_stage1() or restore()"
            )

    def init_from_stage1(self, path: str | Path) -> None:
        if self._ready or self._restore_attempted or self._restoring_requested:
            raise RuntimeError("init_from_stage1() requires a fresh, uninitialized Stage2Trainer")
        path = Path(path).resolve()
        # Keep one inode open across hashing and loading: latest.pt may be
        # atomically replaced by a running stage-1 trainer at any time.
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
            stream.seek(0)
            state = load_stage1_checkpoint(stream)
        parent_config = Config.from_dict(state["config"])
        if (
            parent_config.model.width != self.config.model.width
            or parent_config.model.depth != self.config.model.depth
        ):
            raise ValueError("Stage-1 rough architecture width/depth does not match stage 2")
        source = state["generator"]
        expected_keys = {"fill", *(f"rough.{key}" for key in self.generator.rough.state_dict())}
        if set(source) != expected_keys:
            raise ValueError("Stage-1 generator must contain exactly rough weights and fill")
        fill = source["fill"]
        if (
            fill.shape != self.generator.fill.shape
            or not fill.is_floating_point()
            or not bool(torch.isfinite(fill).all())
            or not bool(((fill >= -1) & (fill <= 1)).all())
            or not torch.equal(fill.cpu(), self.generator.fill.detach().cpu())
        ):
            raise ValueError("Stage-1 fill buffer differs from the stage-2 input contract")
        rough_state = {
            key.removeprefix("rough."): value
            for key, value in source.items()
            if key.startswith("rough.")
        }
        self.generator.rough.load_state_dict(rough_state, strict=True)
        with torch.no_grad():
            self.generator.fill.copy_(fill)
        self.generator.rough.requires_grad_(False).eval()
        self.parent_provenance = {
            "checkpoint": str(path),
            "path": str(path),
            "sha256": digest.hexdigest(),
            "format_version": state["format_version"],
            "stage": state["stage"],
            "global_step": state["global_step"],
            "completed_epochs": state["completed_epochs"],
            "training_semantics": state["training_semantics"],
            "loss_config": {
                key: getattr(parent_config.train, key)
                for key in ("reconstruction_weight", "lpips_weight", "lpips_net")
            },
        }
        # Only rough/fill are inherited. G/D, optimizer moments, stage clocks,
        # RNG and the sample cursor belong to the new refinement experiment.
        self._ready = True

    def _real_features(self, image: Tensor) -> dict[str, Tensor]:
        chunks = {"pool3": [], "relu4_3": []}
        with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
            for part in image.split(self.config.train.lpips_batch_size):
                values = self.features(part.float())
                for name in chunks:
                    chunks[name].append(values[name])
        return {name: torch.cat(values, 0) for name, values in chunks.items()}

    def _prepare_image_discriminator(self) -> None:
        """Advance SN once per optimizer phase, independently of chunk count.

        All subsequent real/fake forwards and activation-checkpoint replays use
        these same immutable buffers. Updating u/v inside a chunked forward
        would change the objective with batch partitioning and corrupt replay.
        """
        if self.config.train.image_discriminator != "realesrgan_unet_sn":
            return
        self.image_d.eval()
        with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
            for module in self.image_d.modules():
                for hook in module._forward_pre_hooks.values():
                    if isinstance(hook, SpectralNorm):
                        hook.compute_weight(module, do_power_iteration=True)

    def _image_logits(self, image: Tensor, *, checkpoint_activations: bool = True) -> Tensor:
        if self.config.train.image_discriminator == "patch":
            return self.image_d(image)
        if any(module.training for module in self.image_d.modules()):
            raise RuntimeError("Real-ESRGAN spectral-normalization state must be frozen per phase")
        # Both real and fake RGB inputs follow the pretrained [0, 1] contract.
        # Do not clamp or detach: generator-side image gradients must survive.
        image = (image.float() + 1) * 0.5
        needs_grad = torch.is_grad_enabled() and (
            image.requires_grad
            or any(parameter.requires_grad for parameter in self.image_d.parameters())
        )
        chunks = []
        for part in image.split(self.config.train.image_discriminator_chunk_size):
            if needs_grad and checkpoint_activations:
                value = checkpoint(self.image_d, part, use_reentrant=False)
            else:
                value = self.image_d(part)
            chunks.append(value)
        # Keep RaLSGAN's batch/patch averages global. These are activation chunks,
        # not independent discriminator losses or gradient-accumulation steps.
        return torch.cat(chunks, 0)

    def _pool3(self, image: Tensor) -> Tensor:
        with torch.autocast(self.device.type, enabled=False):
            if hasattr(self.features, "pool3_features"):
                return self.features.pool3_features(image.float())
            return self.features(image.float())["pool3"]

    def _fake_pool3(self, image: Tensor) -> Tensor:
        # Chunk only feature extraction. Concatenate before either D loss so
        # RaLSGAN means still include every sample/patch in the optimizer batch.
        chunks = []
        for part in image.split(self.config.train.lpips_batch_size):
            if torch.is_grad_enabled() and part.requires_grad:
                chunks.append(checkpoint(self._pool3, part, use_reentrant=False))
            else:
                chunks.append(self._pool3(part))
        return torch.cat(chunks, 0)

    def _backward_step(self, loss: Tensor, optimizer: torch.optim.Optimizer) -> None:
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite stage-2 loss at step {self.global_step}")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        if not self.scaler.is_enabled():
            nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                float("inf"),
                error_if_nonfinite=True,
            )
        self.scaler.step(optimizer)

    def step(self, image: Tensor, mask: Tensor) -> dict[str, float]:
        self._require_ready()
        try:
            return self._step(image, mask)
        except Exception:
            # D may already have updated when G fails. Do not permit a save or
            # another step pretending this partial iteration is a valid state.
            self._ready = False
            self._restore_attempted = True
            raise

    def _step(self, image: Tensor, mask: Tensor) -> dict[str, float]:
        image = image.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        self.generator.train()
        self.image_d.train()
        self.feature_d.train()
        self.features.eval()
        self.g_optimizer.zero_grad(set_to_none=True)
        self.d_optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            output = self.generator(image, mask)
        real_features = self._real_features(image)
        fake_pool3 = self._fake_pool3(output.refined)
        self._prepare_image_discriminator()
        with self.autocast():
            d_image = relativistic_discriminator(
                self._image_logits(image), self._image_logits(output.refined.detach())
            )
            d_feature = relativistic_discriminator(
                self.feature_d(real_features["pool3"]), self.feature_d(fake_pool3.detach())
            )
            d_loss = (d_image + d_feature) / 2
        self._backward_step(d_loss, self.d_optimizer)
        self.d_optimizer.zero_grad(set_to_none=True)
        self._prepare_image_discriminator()
        with _frozen_parameters(self.image_d, self.feature_d):
            with self.autocast():
                with torch.no_grad():
                    real_logits = self._image_logits(image)
                    real_feature_logits = self.feature_d(real_features["pool3"])
                g_image = relativistic_generator(real_logits, self._image_logits(output.refined))
                g_feature = relativistic_generator(real_feature_logits, self.feature_d(fake_pool3))
            rec = F.l1_loss(output.refined.float(), image.float())
            con = consistency(output, real_features["relu4_3"])
            perceptual = (
                self.perceptual(output.completed, image)
                if self.perceptual is not None
                else rec.new_zeros(())
            )
            t = self.config.train
            weighted = {
                "reconstruction_weighted": t.reconstruction_weight * rec,
                "lpips_weighted": t.lpips_weight * perceptual,
                "consistency_weighted": t.consistency_weight * con,
                "adversarial_image_weighted": t.adversarial_weight * g_image,
                "adversarial_feature_weighted": t.adversarial_weight * g_feature,
            }
            g_loss = sum(weighted.values())
            self._backward_step(g_loss, self.g_optimizer)
        self.scaler.update()
        self.global_step += 1
        values = {
            "generator": g_loss,
            "discriminator": d_loss,
            "discriminator_image": d_image,
            "discriminator_feature": d_feature,
            "discriminator_image_weighted": d_image / 2,
            "discriminator_feature_weighted": d_feature / 2,
            "reconstruction": rec,
            "lpips": perceptual,
            "consistency": con,
            "adversarial_image": g_image,
            "adversarial_feature": g_feature,
            "refined_min": output.refined.detach().amin(),
            "refined_max": output.refined.detach().amax(),
            **weighted,
        }
        return {name: value.detach().item() for name, value in values.items()}

    def finish_epoch(self) -> None:
        self._require_ready()
        self.g_scheduler.step()
        self.d_scheduler.step()
        self.completed_epochs += 1

    def save(self, path: str | Path) -> None:
        self._require_ready()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": CHECKPOINT_FORMAT,
            "kind": "csa_stage2",
            "stage": 2,
            "training_semantics": self.training_semantics,
            "output_activation": "tanh",
            "rough_frozen": True,
            "bottleneck": "dense",
            "config": self.config.as_dict(),
            "test_fixture": self.test_fixture,
            "feature_mode": self.feature_mode,
            "torch_version": str(torch.__version__),
            "generator": self.generator.state_dict(),
            "image_d": self.image_d.state_dict(),
            "image_d_provenance": copy.deepcopy(self.image_d_provenance),
            "image_d_state_sha256": _weights_sha256(self.image_d.state_dict()),
            "image_d_spectral_norm_policy": (
                SPECTRAL_NORM_POLICY
                if self.config.train.image_discriminator == "realesrgan_unet_sn"
                else None
            ),
            "feature_d": self.feature_d.state_dict(),
            "features": self.features.state_dict(),
            "features_provenance": self.features_provenance,
            "perceptual": self.perceptual.state_dict() if self.perceptual is not None else None,
            "lpips_provenance": self.lpips_provenance,
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_scheduler": self.g_scheduler.state_dict(),
            "d_scheduler": self.d_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "completed_epochs": self.completed_epochs,
            "data_progress": copy.deepcopy(self.data_progress),
            "parent_provenance": copy.deepcopy(self.parent_provenance),
            "rng_cpu": torch.get_rng_state(),
            "rng_python": random.getstate(),
            "rng_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "rng_mps": torch.mps.get_rng_state() if self.device.type == "mps" else None,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def restore(self, path: str | Path) -> None:
        self._ready = False
        self._restore_attempted = True
        state = load_stage2_checkpoint(path)
        saved = Config.from_dict(state["config"])
        ignored = {"device", "output", "log_every", "image_discriminator_pretrained"}
        old_train = {key: value for key, value in asdict(saved.train).items() if key not in ignored}
        new_train = {
            key: value for key, value in asdict(self.config.train).items() if key not in ignored
        }
        old_data, new_data = asdict(saved.data), asdict(self.config.data)
        for key in ("train_images", "val_images", "train_masks", "val_masks", "workers"):
            old_data.pop(key)
            new_data.pop(key)
        if (
            saved.model != self.config.model
            or old_train != new_train
            or old_data != new_data
            or state["test_fixture"] != self.test_fixture
            or state["feature_mode"] != self.feature_mode
        ):
            raise ValueError("Resume config differs from the saved stage-2 settings")
        if state["training_semantics"] != self.training_semantics:
            raise ValueError("Resume discriminator semantics differ")
        if self.config.train.image_discriminator == "realesrgan_unet_sn":
            validate_discriminator_provenance(state["image_d_provenance"], self.image_d)
        if (
            self.device.type == "cuda"
            and state["rng_cuda"]
            and len(state["rng_cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Cannot restore stage-2 CUDA RNG state across different GPU counts")
        for name in (
            "generator",
            "image_d",
            "feature_d",
            "features",
            "g_optimizer",
            "d_optimizer",
            "g_scheduler",
            "d_scheduler",
            "scaler",
        ):
            getattr(self, name).load_state_dict(state[name])
        restored_features = self._features_provenance()
        if restored_features != state.get("features_provenance"):
            raise ValueError("VGG feature provenance does not match the saved weights")
        self.features_provenance = restored_features
        if self.perceptual is not None:
            self.perceptual.restore_checkpoint(state["perceptual"], state["lpips_provenance"])
            self.lpips_provenance = copy.deepcopy(self.perceptual.provenance)
        elif state.get("perceptual") is not None or state.get("lpips_provenance") is not None:
            raise ValueError("LPIPS checkpoint state disagrees with the configured loss")
        self.generator.rough.requires_grad_(False).eval()
        self.image_d_provenance = copy.deepcopy(state.get("image_d_provenance"))
        if self.config.train.image_discriminator == "realesrgan_unet_sn":
            self.image_d.eval()
        self.features.eval().requires_grad_(False)
        self.global_step = state["global_step"]
        self.completed_epochs = state["completed_epochs"]
        self.data_progress = copy.deepcopy(state["data_progress"])
        self.parent_provenance = copy.deepcopy(state["parent_provenance"])
        torch.set_rng_state(state["rng_cpu"])
        random.setstate(state["rng_python"])
        if self.device.type == "cuda" and state["rng_cuda"]:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        if self.device.type == "mps" and state["rng_mps"] is not None:
            torch.mps.set_rng_state(state["rng_mps"])
        self._ready = True


def load_stage2_generator(path: str | Path, device: torch.device):
    state = load_stage2_checkpoint(path)
    config = Config.from_dict(state["config"])
    model = Stage2InpaintingModel(**asdict(config.model)).to(device)
    model.load_state_dict(state["generator"], strict=True)
    return model.eval(), config
