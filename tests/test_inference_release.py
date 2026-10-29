import hashlib
import json
import sys
from dataclasses import asdict

import pytest
import torch
from PIL import Image
from torch import nn

from csa_inpainting.cli import main
from csa_inpainting.engine import load_generator
from csa_inpainting.inference import INFERENCE_FORMAT, export_checkpoint
from csa_inpainting.networks import InpaintingModel
from csa_inpainting.stage2 import TRAINING_SEMANTICS, Stage2InpaintingModel


@pytest.fixture
def checkpoint(tmp_path, tiny_config):
    # Synthetic small-width weights exercise the production serialization contract
    # without constructing VGG, LPIPS, or a training process.
    tiny_config.data.train_images = "/private/training/images.txt"
    tiny_config.data.val_images = "/private/validation/images.txt"
    tiny_config.data.mask_mode = "brush"
    tiny_config.train.output = "/private/run"
    model = Stage2InpaintingModel(**asdict(tiny_config.model)).eval()
    with torch.no_grad():
        final = [layer for layer in model.refine.head if isinstance(layer, nn.ConvTranspose2d)][-1]
        final.weight.zero_()
        final.bias.fill_(50)
    state = {
        "format_version": "stage2_v1",
        "kind": "csa_stage2",
        "stage": 2,
        "training_semantics": TRAINING_SEMANTICS,
        "output_activation": "tanh",
        "rough_frozen": True,
        "bottleneck": "dense",
        "test_fixture": False,
        "feature_mode": "imagenet_vgg_stage2",
        "config": tiny_config.as_dict(),
        "global_step": 97397,
        "completed_epochs": 5,
        "generator": model.state_dict(),
        "parent_provenance": {
            "stage": 1,
            "format_version": "stage1_v1",
            "checkpoint": "/private/parent.pt",
            "sha256": "1" * 64,
            "global_step": 156181,
            "completed_epochs": 9,
        },
        "g_optimizer": {"private": "/private/optimizer"},
        "image_d": {"unused": torch.ones(4)},
        "features": {"unused": torch.ones(4)},
        "perceptual": {"unused": torch.ones(4)},
    }
    path = tmp_path / "training.pt"
    torch.save(state, path)
    return path, state, model


def test_export_preserves_both_stages_tanh_and_known_pixels(
    checkpoint,
    tmp_path,
    batch,
    monkeypatch,
):
    path, state, original = checkpoint

    def no_download(*args, **kwargs):
        pytest.fail("Inference/export must not download weights")

    monkeypatch.setattr(torch.hub, "download_url_to_file", no_download)
    output = tmp_path / "portable.pt"
    metadata = export_checkpoint(path, output)
    bundle = torch.load(output, weights_only=True)
    restored, config = load_generator(output, torch.device("cpu"))
    from_training, _ = load_generator(path, torch.device("cpu"))
    assert isinstance(restored, Stage2InpaintingModel)
    assert isinstance(from_training, Stage2InpaintingModel)
    assert not restored.training and not restored.rough.training
    assert not any(p.requires_grad for p in restored.rough.parameters())
    assert config.model == original_config(state).model
    assert config.data.mask_mode == "brush" and not config.data.val_images
    assert metadata["provenance"]["global_step"] == 97397
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["provenance"]["checkpoint_sha256"] == source_hash
    assert "/private" not in json.dumps(metadata)
    assert "generator" not in metadata
    assert not {"g_optimizer", "image_d", "features", "perceptual", "config"} & bundle.keys()
    for key, expected in state["generator"].items():
        assert torch.equal(bundle["generator"][key], expected)
    with torch.inference_mode():
        expected = original(*batch)
        for model in (restored, from_training):
            actual = model(*batch)
            for name in ("rough", "refined", "completed"):
                torch.testing.assert_close(
                    getattr(actual, name),
                    getattr(expected, name),
                    rtol=0,
                    atol=0,
                )
            assert actual.refined.min() == actual.refined.max() == 1
            image, mask = batch
            valid = ~mask.expand_as(image)
            assert torch.equal(actual.completed[valid], image[valid])


def original_config(state):
    from csa_inpainting.config import Config

    return Config.from_dict(state["config"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("format_version", "stage1_v1"),
        ("training_semantics", "joint"),
        ("output_activation", "identity"),
        ("rough_frozen", False),
        ("test_fixture", True),
        ("feature_mode", "tiny_diagnostic_stage2"),
        ("global_step", -1),
    ],
)
def test_export_rejects_incompatible_or_diagnostic(checkpoint, tmp_path, key, value):
    path, state, _ = checkpoint
    state[key] = value
    torch.save(state, path)
    output = tmp_path / "portable.pt"
    with pytest.raises(ValueError):
        export_checkpoint(path, output)
    assert not output.exists()


@pytest.mark.parametrize("corruption", ["missing", "nonfinite", "extra"])
def test_export_strict_generator_validation(checkpoint, tmp_path, corruption):
    path, state, _ = checkpoint
    key = next(k for k in state["generator"] if k.startswith("rough."))
    if corruption == "missing":
        del state["generator"][key]
    elif corruption == "nonfinite":
        state["generator"][key].flatten()[0] = float("nan")
    else:
        state["generator"]["unknown.weight"] = torch.ones(1)
    torch.save(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        export_checkpoint(path, tmp_path / "portable.pt")


def test_export_rejects_silent_dtype_conversion(checkpoint, tmp_path):
    path, state, _ = checkpoint
    state["generator"] = {key: value.half() for key, value in state["generator"].items()}
    torch.save(state, path)
    output = tmp_path / "portable.pt"
    with pytest.raises(ValueError, match="would change generator tensors or dtypes"):
        export_checkpoint(path, output)
    assert not output.exists()


@pytest.mark.parametrize("corruption", ["weights", "activation", "private_path", "parent"])
def test_bundle_rejects_corruption(checkpoint, tmp_path, corruption):
    path, _, _ = checkpoint
    output = tmp_path / "portable.pt"
    export_checkpoint(path, output)
    state = torch.load(output, weights_only=True)
    if corruption == "weights":
        state["generator"]["fill"].add_(1)
    elif corruption == "activation":
        state["output_activation"] = "identity"
    elif corruption == "private_path":
        state["evaluation_config"]["val_images"] = "/private/images"
    else:
        state["provenance"]["parent"]["sha256"] = "invalid"
    torch.save(state, output)
    with pytest.raises(ValueError):
        load_generator(output, torch.device("cpu"))


def test_export_never_overwrites(checkpoint, tmp_path):
    path, _, _ = checkpoint
    output = tmp_path / "existing.pt"
    output.write_bytes(b"existing release")
    with pytest.raises(FileExistsError):
        export_checkpoint(path, output)
    assert output.read_bytes() == b"existing release"
    source = path.read_bytes()
    with pytest.raises(FileExistsError):
        export_checkpoint(path, path)
    assert path.read_bytes() == source


def test_export_removes_failed_write(checkpoint, tmp_path, monkeypatch):
    path, _, _ = checkpoint
    output = tmp_path / "portable.pt"

    def fail_save(value, stream):
        stream.write(b"incomplete")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        export_checkpoint(path, output)
    assert not output.exists()


def test_cli_export_and_infer(checkpoint, tmp_path, monkeypatch, capsys):
    path, _, _ = checkpoint
    output = tmp_path / "portable.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "csa",
            "export",
            "--checkpoint",
            str(path),
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["format_version"] == INFERENCE_FORMAT
    image_path, mask_path, result_path = (tmp_path / n for n in ("rgb.png", "mask.png", "out.png"))
    Image.new("RGB", (71, 67), (40, 80, 120)).save(image_path)
    mask = Image.new("L", (71, 67), 0)
    mask.paste(255, (20, 20, 40, 40))
    mask.save(mask_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "csa",
            "infer",
            "--checkpoint",
            str(output),
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output",
            str(result_path),
            "--device",
            "cpu",
        ],
    )
    main()
    with Image.open(result_path) as result:
        assert result.size == (71, 67)
        assert result.getpixel((0, 0)) == (40, 80, 120)
        assert result.getpixel((25, 25)) == (255, 255, 255)
    capsys.readouterr()
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (64, 64), (40, 80, 120)).save(images / "rgb.png")
    mask.save(masks / "mask.png")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "csa",
            "evaluate",
            "--checkpoint",
            str(output),
            "--images",
            str(images),
            "--masks",
            str(masks),
            "--output",
            str(tmp_path / "evaluation"),
            "--device",
            "cpu",
        ],
    )
    main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["images"] == 1 and summary["hole_l1"] > 0
    assert (tmp_path / "evaluation" / "metrics.json").is_file()


@pytest.mark.parametrize("version", [1, 2])
def test_joint_checkpoint_keeps_original_head(checkpoint, tmp_path, batch, version):
    _, state, _ = checkpoint
    config = original_config(state)
    model = InpaintingModel(**asdict(config.model)).eval()
    model.load_state_dict(state["generator"], strict=True)
    path = tmp_path / "historical.pt"
    torch.save(
        {"format_version": version, "config": config.as_dict(), "generator": model.state_dict()},
        path,
    )
    restored, _ = load_generator(path, torch.device("cpu"))
    assert type(restored) is InpaintingModel
    with torch.inference_mode():
        expected, actual = model(*batch), restored(*batch)
        torch.testing.assert_close(actual.completed, expected.completed, rtol=0, atol=0)
        assert actual.refined.min() == actual.refined.max() == 50
