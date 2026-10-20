import json
from dataclasses import asdict

import numpy as np
import pytest
import yaml
from PIL import Image

from csa_inpainting import cli
from csa_inpainting.config import Config
from csa_inpainting.engine import Trainer, load_checkpoint
from csa_inpainting.experiment import check_output_directory


def test_train_resume_evaluate_infer_commands(tmp_path, monkeypatch, tiny_config):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(3):
        pixels = np.random.default_rng(i).integers(0, 256, (67, 71, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images / f"{i}.png")
    tiny_config.data.train_images = str(images)
    tiny_config.data.val_images = str(images)
    tiny_config.train.epochs = 2
    tiny_config.train.decay_start = 1
    tiny_config.train.output = str(tmp_path / "training")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(tiny_config)))
    # Test all command wiring without downloading VGG or training a large model.
    monkeypatch.setattr(
        cli,
        "Trainer",
        lambda config, restoring=False: Trainer(config, test_fixture=True, restoring=restoring),
    )

    def run(*arguments):
        monkeypatch.setattr("sys.argv", ["csa", *map(str, arguments)])
        cli.main()

    run("train", "--config", config_path)
    checkpoint = tmp_path / "training/latest.pt"
    state = load_checkpoint(checkpoint)
    assert state["completed_epochs"] == 2
    assert state["global_step"] == 4  # 3 images, batch 2, tail kept, 2 epochs
    run("train", "--config", config_path, "--resume", checkpoint)
    assert load_checkpoint(checkpoint)["global_step"] == 4
    records = [
        json.loads(line)
        for line in (tmp_path / "training/invocations.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]["source_sha256"]["engine.py"]
    assert records[0]["config"] == asdict(tiny_config)
    assert records[1]["resume"] == str(checkpoint.resolve())
    with pytest.raises(ValueError, match="not empty"):
        run("train", "--config", config_path)
    evaluation = tmp_path / "evaluation"
    run(
        "evaluate",
        "--checkpoint",
        checkpoint,
        "--images",
        images,
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--output",
        evaluation,
    )
    assert json.loads((evaluation / "metrics.json").read_text())["summary"]["images"] == 3

    mask = np.zeros((67, 71), dtype=np.uint8)
    mask[15:40, 20:50] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)
    output_path = tmp_path / "result.png"
    run(
        "infer",
        "--checkpoint",
        checkpoint,
        "--image",
        images / "0.png",
        "--mask",
        mask_path,
        "--device",
        "cpu",
        "--output",
        output_path,
    )
    with Image.open(output_path) as result, Image.open(images / "0.png") as original:
        assert result.size == original.size == (71, 67)
        np.testing.assert_array_equal(np.array(result)[mask == 0], np.array(original)[mask == 0])


def test_dry_run_overrides_without_data_weights_or_files(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "base.yaml"
    config_path.write_text("{}")
    destination = tmp_path / "should-not-exist"

    def unexpected(*args, **kwargs):
        pytest.fail("Dry run must not initialize data or models")

    monkeypatch.setattr(cli, "Trainer", unexpected)
    monkeypatch.setattr(cli, "InpaintingDataset", unexpected)
    monkeypatch.setattr(
        "sys.argv",
        [
            "csa",
            "train",
            "--config",
            str(config_path),
            "--dry-run",
            "--set",
            "data.batch_size=4",
            "train.lr=2e-4",
            "train.amp=off",
            "data.horizontal_flip=false",
            f"train.output={destination}",
        ],
    )
    cli.main()
    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["data"]["batch_size"] == 4
    assert resolved["data"]["horizontal_flip"] is False
    assert resolved["train"]["lr"] == 0.0002
    assert resolved["train"]["amp"] == "off"
    assert not destination.exists()
    assert config_path.read_text() == "{}"


@pytest.mark.parametrize(
    "override",
    [
        "data.batch_size=0",
        "data.batch_size=true",
        "data.batch_size=1.5",
        "data.batch_size=four",
        "data.horizontal_flip=1",
        "train.lr=.nan",
        "train.lr=.inf",
        "train.learning_rate=0.1",
        "train.lr",
        "model.width.extra=2",
        "train.amp=bogus",
    ],
)
def test_invalid_experiment_overrides(override):
    with pytest.raises(ValueError):
        Config().with_overrides([override])


def test_override_does_not_mutate_original_config():
    original = Config()
    overridden = original.with_overrides(["data.batch_size=4"])
    assert original.data.batch_size == 1
    assert overridden.data.batch_size == 4


def test_resume_does_not_overwrite_a_different_experiment(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "latest.pt").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="not empty"):
        check_output_directory(output, tmp_path / "other/latest.pt")
    assert (output / "latest.pt").read_bytes() == b"preserve"
    assert check_output_directory(output, output / "latest.pt") == output
