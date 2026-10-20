import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    width: int = 64
    depth: int = 8
    attention_chunk_size: int = 256
    mask_threshold: float = 5 / 16


@dataclass
class DataConfig:
    train_images: str = ""
    val_images: str = ""
    train_masks: str = ""
    val_masks: str = ""
    mask_mode: str = "center"
    mask_white_is_hole: bool = True
    image_size: int = 256
    center_overlap: int = 4
    horizontal_flip: bool = True
    batch_size: int = 1
    workers: int = 4


@dataclass
class TrainConfig:
    output: str = "runs/csa"
    device: str = "auto"
    seed: int = 2019
    epochs: int = 120
    decay_start: int = 20
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    reconstruction_weight: float = 1.0
    consistency_weight: float = 0.01
    adversarial_weight: float = 0.002
    discriminator_width: int = 64
    feature_discriminator_width: int = 512
    amp: str = "off"
    log_every: int = 50


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Config":
        if not isinstance(value, dict) or set(value) - {"model", "data", "train"}:
            raise ValueError("Config must contain only model, data and train sections")
        parts = {}
        for name, section_type in (
            ("model", ModelConfig),
            ("data", DataConfig),
            ("train", TrainConfig),
        ):
            section = value.get(name, {})
            if not isinstance(section, dict):
                raise ValueError(f"{name} must be a mapping")
            unknown = set(section) - {f.name for f in fields(section_type)}
            if unknown:
                raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
            parts[name] = section_type(**section)
        config = cls(**parts)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path, overrides: Sequence[str] = ()) -> "Config":
        with Path(path).open() as stream:
            config = cls.from_dict(yaml.safe_load(stream))
        return config.with_overrides(overrides)

    def with_overrides(self, overrides: Sequence[str]) -> "Config":
        """Return a validated copy with explicit ``section.key=value`` changes."""
        value = self.as_dict()
        for expression in overrides:
            path, separator, raw = expression.partition("=")
            parts = path.split(".")
            if not separator or len(parts) != 2:
                raise ValueError(f"Expected section.key=value, got {expression!r}")
            section, name = parts
            if section not in value or name not in value[section]:
                raise ValueError(f"Unknown config setting: {path}")
            try:
                parsed = yaml.safe_load(raw)
            except yaml.YAMLError as error:
                raise ValueError(f"Invalid value for {path}: {raw!r}") from error
            expected = next(f.type for f in fields(getattr(self, section)) if f.name == name)
            if expected is str:
                # YAML 1.1 interprets "off"/"on" as booleans; typed strings keep
                # their literal meaning for options such as train.amp=off.
                parsed = parsed if isinstance(parsed, str) else raw
            elif expected is float and isinstance(parsed, str):
                try:
                    parsed = float(parsed)  # Also accept scientific notation, e.g. 2e-4.
                except ValueError:
                    pass
            value[section][name] = parsed
        return type(self).from_dict(value)

    def validate(self) -> None:
        for section in ("model", "data", "train"):
            config = getattr(self, section)
            for definition in fields(config):
                value = getattr(config, definition.name)
                expected = definition.type
                valid = (
                    type(value) in (int, float) if expected is float else type(value) is expected
                )
                key = f"{section}.{definition.name}"
                if not valid:
                    raise ValueError(f"{key} must be {expected.__name__}, got {value!r}")
                if expected is float and not math.isfinite(value):
                    raise ValueError(f"{key} must be finite")
        m, d, t = self.model, self.data, self.train
        if m.depth < 4 or m.width < 1 or m.attention_chunk_size < 1:
            raise ValueError("Invalid model dimensions")
        if not 0 < m.mask_threshold < 1:
            raise ValueError("mask_threshold must be between 0 and 1")
        if d.image_size < 64 or d.image_size % 2**m.depth:
            raise ValueError("image_size must be >=64 and divisible by 2**depth")
        if d.batch_size < 1 or d.workers < 0:
            raise ValueError("batch_size must be positive and workers nonnegative")
        if d.mask_mode not in {"center", "files", "brush"}:
            raise ValueError("mask_mode must be center, files or brush")
        if d.center_overlap < 0 or d.center_overlap * 2 >= d.image_size // 2:
            raise ValueError("center_overlap removes the entire hole")
        if t.epochs < 1 or not 0 <= t.decay_start < t.epochs:
            raise ValueError("Require 0 <= decay_start < epochs")
        if t.lr <= 0 or not (0 <= t.beta1 < 1 and 0 <= t.beta2 < 1):
            raise ValueError("Invalid Adam settings")
        if t.amp not in {"off", "fp16", "bf16"}:
            raise ValueError("amp must be off, fp16 or bf16")
        if t.discriminator_width < 1 or t.feature_discriminator_width < 1 or t.log_every < 1:
            raise ValueError("Discriminator widths and log_every must be positive")
        if min(t.reconstruction_weight, t.consistency_weight, t.adversarial_weight) < 0:
            raise ValueError("Loss weights cannot be negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
