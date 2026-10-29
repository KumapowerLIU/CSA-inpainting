"""Portable stage-2 inference bundles, with no training-network dependencies at runtime."""

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import Config
from .networks import InpaintingModel
from .perceptual import _weights_sha256

INFERENCE_FORMAT = "csa_inference_v1"
_EVALUATION_KEYS = {
    "mask_mode",
    "random_min_fraction",
    "random_max_fraction",
    "mask_white_is_hole",
    "image_size",
    "center_overlap",
}
_INPUT_CONTRACT = {
    "image_range": [-1, 1],
    "mask": "true_is_hole",
    "known_pixels": "copied_from_input",
}


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stage2_metadata(state: dict, config: Config) -> None:
    from .stage2 import stage2_training_semantics

    if (
        state.get("stage") != 2
        or state.get("output_activation") != "tanh"
        or state.get("rough_frozen") is not True
        or state.get("bottleneck") != "dense"
        or state.get("training_semantics") != stage2_training_semantics(config)
    ):
        raise ValueError("Incompatible stage-2 inference semantics")


def _generator(state: dict, config: Config, device: torch.device) -> InpaintingModel:
    from .stage2 import Stage2InpaintingModel

    weights = state.get("generator")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Missing generator state")
    if any(
        not isinstance(key, str)
        or not isinstance(value, torch.Tensor)
        or not bool(torch.isfinite(value).all())
        for key, value in weights.items()
    ):
        raise ValueError("Generator state must contain only finite tensors")
    model = Stage2InpaintingModel(**asdict(config.model))
    # Tanh has no tensors: selecting the stage-2 class is as important as strict loading.
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def _bundle_config(state: dict) -> Config:
    required = {
        "format_version",
        "kind",
        "stage",
        "training_semantics",
        "output_activation",
        "rough_frozen",
        "bottleneck",
        "model_config",
        "evaluation_config",
        "seed",
        "image_discriminator",
        "input_contract",
        "provenance",
        "generator_sha256",
        "generator",
    }
    if set(state) != required or state.get("kind") != "csa_inference":
        raise ValueError("Invalid inference bundle fields")
    evaluation = state["evaluation_config"]
    if not isinstance(evaluation, dict) or set(evaluation) != _EVALUATION_KEYS:
        raise ValueError("Invalid inference evaluation settings")
    if state["input_contract"] != _INPUT_CONTRACT:
        raise ValueError("Unsupported inference input contract")
    provenance = state["provenance"]
    if (
        not isinstance(provenance, dict)
        or set(provenance)
        != {
            "source_format",
            "checkpoint_sha256",
            "global_step",
            "completed_epochs",
            "parent",
        }
        or provenance.get("source_format") != "stage2_v1"
        or not _sha256(provenance.get("checkpoint_sha256"))
        or not _nonnegative_int(provenance.get("global_step"))
        or not _nonnegative_int(provenance.get("completed_epochs"))
    ):
        raise ValueError("Invalid inference checkpoint provenance")
    parent = provenance["parent"]
    if (
        not isinstance(parent, dict)
        or set(parent) != {"stage", "format_version", "sha256", "global_step", "completed_epochs"}
        or parent.get("stage") != 1
        or parent.get("format_version") != "stage1_v1"
        or not _sha256(parent.get("sha256"))
        or not _nonnegative_int(parent.get("global_step"))
        or not _nonnegative_int(parent.get("completed_epochs"))
    ):
        raise ValueError("Invalid inference parent provenance")
    config = Config.from_dict(
        {
            "model": state["model_config"],
            "data": evaluation,
            "train": {"seed": state["seed"], "image_discriminator": state["image_discriminator"]},
        }
    )
    _stage2_metadata(state, config)
    weights = state.get("generator")
    if (
        not isinstance(weights, dict)
        or any(
            not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in weights.items()
        )
        or not _sha256(state["generator_sha256"])
        or _weights_sha256(weights) != state["generator_sha256"]
    ):
        raise ValueError("Inference generator checksum mismatch")
    return config


def load_inference_generator(
    path: str | Path,
    device: torch.device,
) -> tuple[InpaintingModel, Config]:
    """Load portable, stage-2, or historical joint checkpoints without training models.

    Historical format 1/2 keeps its original, unbounded refinement head. Stage-2
    inference validates the generator contract; resuming training should continue
    to use the stronger trainer checkpoint validator, including discriminator state.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected a CSA checkpoint mapping")
    version = state.get("format_version")
    if version == INFERENCE_FORMAT:
        config = _bundle_config(state)
        return _generator(state, config, device), config
    if version == "stage2_v1" and state.get("kind") == "csa_stage2":
        config = Config.from_dict(state["config"])
        _stage2_metadata(state, config)
        return _generator(state, config, device), config
    if type(version) is int and version in {1, 2}:
        config = Config.from_dict(state["config"])
        model = InpaintingModel(**asdict(config.model)).to(device)
        model.load_state_dict(state["generator"], strict=True)
        return model.eval(), config
    raise ValueError("Unsupported checkpoint; legacy weights are not interchangeable")


def export_checkpoint(checkpoint: str | Path, output: str | Path) -> dict:
    """Export a production stage-2 generator on CPU; never overwrite an existing file.

    Paths, data manifests, discriminator/perceptual networks and optimizer state
    are deliberately excluded. Source and parent hashes identify the training state.
    """
    from .stage2 import load_stage2_checkpoint

    source, destination = Path(checkpoint), Path(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    before = source.stat()
    state = load_stage2_checkpoint(source)
    if state.get("test_fixture") is not False or state.get("feature_mode") != "imagenet_vgg_stage2":
        raise ValueError("Only production stage-2 checkpoints can be exported for release")
    if not all(_nonnegative_int(state.get(key)) for key in ("global_step", "completed_epochs")):
        raise ValueError("Invalid stage-2 training progress")
    config = Config.from_dict(state["config"])
    _stage2_metadata(state, config)
    model = _generator(state, config, torch.device("cpu"))
    generator_hash = _weights_sha256(state["generator"])
    if _weights_sha256(model.state_dict()) != generator_hash:
        raise ValueError(
            "Export would change generator tensors or dtypes; expected native FP32 state"
        )
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    source_hash = digest.hexdigest()
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("Source checkpoint changed during export; retry with a stable snapshot")
    parent = state["parent_provenance"]
    weights = model.state_dict()
    bundle = {
        "format_version": INFERENCE_FORMAT,
        "kind": "csa_inference",
        **{
            key: state[key]
            for key in (
                "stage",
                "training_semantics",
                "output_activation",
                "rough_frozen",
                "bottleneck",
            )
        },
        "model_config": asdict(config.model),
        "evaluation_config": {key: getattr(config.data, key) for key in sorted(_EVALUATION_KEYS)},
        "seed": config.train.seed,
        "image_discriminator": config.train.image_discriminator,
        "input_contract": dict(_INPUT_CONTRACT),
        "provenance": {
            "source_format": state["format_version"],
            "checkpoint_sha256": source_hash,
            "global_step": state["global_step"],
            "completed_epochs": state["completed_epochs"],
            "parent": {
                key: parent[key]
                for key in (
                    "stage",
                    "format_version",
                    "sha256",
                    "global_step",
                    "completed_epochs",
                )
            },
        },
        "generator_sha256": generator_hash,
        "generator": weights,
    }
    _bundle_config(bundle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against an output appearing after validation.
    # Clean up a failed serialization/close without replacing an earlier release.
    created = False
    try:
        with destination.open("xb") as stream:
            created = True
            torch.save(bundle, stream)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return {key: value for key, value in bundle.items() if key != "generator"}
