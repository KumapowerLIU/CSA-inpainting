import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from csa_inpainting.config import Config, DataConfig
from csa_inpainting.data import InpaintingDataset, read_mask
from csa_inpainting.evaluation import evaluate, metrics
from csa_inpainting.networks import InpaintingModel


def make_images(root):
    root.mkdir()
    (root / "nested").mkdir()
    for i in range(3):
        pixels = np.random.default_rng(i).integers(0, 256, (80, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "nested" / f"{i}.png")


def test_recursive_loader_masks_and_determinism(tmp_path):
    images = tmp_path / "images"
    make_images(images)
    config = DataConfig(
        train_images=str(images), val_images=str(images), mask_mode="brush", image_size=64
    )
    dataset = InpaintingDataset(config, True)
    assert len(dataset) == 3
    first = dataset[0]
    assert first["image"].shape == (3, 64, 64) and first["mask"].shape == (1, 64, 64)
    assert first["image"].min() >= -1 and first["image"].max() <= 1
    assert torch.equal(first["mask"], dataset[0]["mask"])
    dataset.set_epoch(1)
    assert not torch.equal(first["mask"], dataset[0]["mask"])
    validation = InpaintingDataset(config, False)
    before = validation[1]
    validation.set_epoch(7)
    assert torch.equal(before["mask"], validation[1]["mask"])
    single_worker = list(DataLoader(dataset, batch_size=2, num_workers=0))
    multi_worker = list(DataLoader(dataset, batch_size=2, num_workers=2))
    for a, b in zip(single_worker, multi_worker, strict=True):
        assert torch.equal(a["image"], b["image"])
        assert torch.equal(a["mask"], b["mask"])


def test_mask_polarity_and_validation_pairing(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    make_images(images)
    masks.mkdir()
    pixels = np.zeros((64, 64), dtype=np.uint8)
    pixels[10:30, 20:40] = 255
    Image.fromarray(pixels).save(masks / "0.png")
    Image.fromarray(255 - pixels).save(masks / "1.png")
    white = read_mask(masks / "0.png", (64, 64), True)
    black = read_mask(masks / "0.png", (64, 64), False)
    assert torch.equal(white, ~black)
    config = DataConfig(
        val_images=str(images), val_masks=str(masks), mask_mode="files", image_size=64
    )
    dataset = InpaintingDataset(config, False)
    assert torch.equal(dataset[0]["mask"], dataset[2]["mask"])
    assert not torch.equal(dataset[0]["mask"], dataset[1]["mask"])


def test_metrics_known_values():
    target = torch.zeros(2, 3, 16, 16)
    prediction = target.clone()
    mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    mask[:, :, :8, :8] = True
    prediction[0, :, :8, :8] = 1
    values = metrics(prediction, target, mask)
    assert values[0]["l1"] == pytest.approx(0.125)
    assert values[0]["l2"] == pytest.approx(0.0625)
    assert values[0]["hole_l1"] == pytest.approx(0.5)
    assert values[0]["hole_l2"] == pytest.approx(0.25)
    assert values[0]["psnr"] == pytest.approx(12.0411998)
    assert values[1]["ssim"] == pytest.approx(1)
    assert values[1]["psnr"] == 120


def test_evaluation_weights_tail_batch_correctly(tmp_path):
    images = tmp_path / "images"
    make_images(images)
    config = DataConfig(val_images=str(images), mask_mode="center", image_size=64)
    dataset = InpaintingDataset(config, False)
    model = InpaintingModel(width=4, depth=6)
    output = tmp_path / "evaluation"
    batched = evaluate(model, DataLoader(dataset, batch_size=2), torch.device("cpu"), output)
    single = evaluate(model, DataLoader(dataset, batch_size=1), torch.device("cpu"))
    assert batched["images"] == 3
    assert model.training
    for key in batched:
        assert batched[key] == pytest.approx(single[key], abs=1e-5)
    assert len(json.loads((output / "metrics.json").read_text())["per_image"]) == 3
    with Image.open(output / "00000.png") as preview:
        assert preview.size == (256, 64)


def test_invalid_config_and_empty_data(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        Config.from_dict({"train": {"learning_rate_typo": 1}})
    with pytest.raises(ValueError, match="image_size"):
        Config.from_dict({"data": {"image_size": 123}})
    with pytest.raises(ValueError, match="No supported images"):
        InpaintingDataset(DataConfig(train_images=str(tmp_path)), True)
