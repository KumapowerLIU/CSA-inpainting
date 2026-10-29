"""CPU integration check on real indexed images; does not run a quality evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from csa_remote_train import RecordDataset
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from csa_inpainting.config import Config
from csa_inpainting.engine import Trainer, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    args = parser.parse_args()
    runtime = Path(args.runtime)
    torch.set_num_threads(2)
    seed_everything(2019)
    config = Config()
    train = RecordDataset(runtime / "data/train.jsonl", True, config.data)
    validation = RecordDataset(runtime / "data/validation.jsonl", False, config.data)
    samples = [train[i] for i in range(16)]
    masks = [sample["mask"] for sample in samples]
    fractions = [mask.float().mean().item() for mask in masks]
    assert all(0.1 <= fraction <= 0.4 for fraction in fractions)
    assert len({hashlib.sha256(mask.numpy().tobytes()).hexdigest() for mask in masks}) == 16
    train.set_epoch(1)
    assert all(not torch.equal(mask, train[i]["mask"]) for i, mask in enumerate(masks))
    train.set_epoch(0)
    assert all(torch.equal(mask, train[i]["mask"]) for i, mask in enumerate(masks))
    validation_mask = validation[0]["mask"]
    validation.set_epoch(11)
    assert torch.equal(validation_mask, validation[0]["mask"])

    subset = Subset(train, range(8))
    single = list(DataLoader(subset, batch_size=4, num_workers=0))
    multi = list(DataLoader(subset, batch_size=4, num_workers=2))
    for a, b in zip(single, multi, strict=True):
        assert torch.equal(a["image"], b["image"])
        assert torch.equal(a["mask"], b["mask"])

    # Validate a complete optimizer update using actual TFRecord images and
    # random masks, with a tiny diagnostic network and no VGG download or GPU.
    tiny = Config.from_dict(
        {
            "model": {"width": 4, "depth": 6},
            "data": {"image_size": 64, "batch_size": 2, "workers": 0},
            "train": {
                "device": "cpu",
                "epochs": 2,
                "decay_start": 1,
                "discriminator_width": 4,
                "feature_discriminator_width": 16,
            },
        }
    )
    tiny_data = RecordDataset(runtime / "data/train.jsonl", True, tiny.data)
    batch = next(iter(DataLoader(Subset(tiny_data, range(2)), batch_size=2)))
    trainer = Trainer(tiny, test_fixture=True)
    losses = trainer.step(batch["image"], batch["mask"])
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())

    output = runtime / "validation"
    output.mkdir(exist_ok=True)
    canvas = Image.new("RGB", (4 * 264 + 8, 2 * 288 + 36), "#20242a")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "Random mask preview - inputs only, not predictions", fill="white")
    fill = torch.tensor([123, 104, 117], dtype=torch.float32).view(3, 1, 1) / 127.5 - 1
    for i, index in enumerate([582, 605, 1659, 477, 97, 1001, 1492, 1850]):
        sample = validation[index]
        masked = torch.where(sample["mask"], fill, sample["image"])
        pixels = ((masked + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0)
        x, y = 8 + (i % 4) * 264, 36 + (i // 4) * 288
        canvas.paste(Image.fromarray(pixels.numpy()), (x, y))
        fraction = sample["mask"].float().mean().item()
        draw.text((x, y + 260), f"sample {index} | hole {fraction:.1%}", fill="white")
    canvas.save(output / "random-mask-preview.png")
    report = {
        "status": "PASS",
        "device": "cpu",
        "mask_mode": config.data.mask_mode,
        "samples_checked": len(samples),
        "hole_fraction_min": min(fractions),
        "hole_fraction_max": max(fractions),
        "worker_count_independent": True,
        "training_changes_each_epoch": True,
        "validation_masks_fixed": True,
        "diagnostic_update_losses": losses,
        "note": "Input/mask and software check; not a trained model quality result",
    }
    (output / "runtime-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
