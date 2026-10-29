"""Freeze the latest stage-1 checkpoint and evaluate a fixed held-out subset."""

import argparse
import hashlib
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from csa_remote_train import RecordDataset
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from csa_inpainting.config import Config
from csa_inpainting.evaluation import metrics, save_image
from csa_inpainting.stage1 import RoughInpaintingModel, load_stage1_checkpoint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument(
        "--no-checkpoint-copy",
        action="store_true",
        help="Load the pinned latest checkpoint directly without retaining a copy.",
    )
    args = parser.parse_args()
    runtime = Path(args.runtime)
    output = runtime / ("quick-eval-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    digest = hashlib.sha256()
    # Opening once pins one complete inode while the trainer may replace latest.pt.
    with (runtime / "run/latest.pt").open("rb") as source:
        if args.no_checkpoint_copy:
            for block in iter(lambda: source.read(8 * 1024**2), b""):
                digest.update(block)
            source.seek(0)
            state = load_stage1_checkpoint(source)
        else:
            with (output / "checkpoint.pt").open("wb") as target:
                for block in iter(lambda: source.read(8 * 1024**2), b""):
                    target.write(block)
                    digest.update(block)
            state = load_stage1_checkpoint(output / "checkpoint.pt")
    config = Config.from_dict(state["config"])
    checkpoint_step = state["global_step"]
    data_progress = state.get("data_progress")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(
        4 * 1024**3 / torch.cuda.get_device_properties(device).total_memory
    )
    model = RoughInpaintingModel(config.model.width, config.model.depth)
    model.load_state_dict(state["generator"])
    del state
    model = model.to(device).eval()
    validation = RecordDataset(
        runtime / "data/validation.jsonl", False, config.data, config.train.seed
    )
    shown = [582, 605, 1659, 477, 97, 1001, 1492, 1850]
    others = [i for i in range(len(validation)) if i not in shown]
    indices = shown + random.Random(20261023).sample(others, 56)
    loader = DataLoader(Subset(validation, indices), batch_size=4, num_workers=4)
    rows, visible_rows, count = [], [], 0
    for batch in loader:
        image, mask = batch["image"].to(device), batch["mask"].to(device)
        prediction = model(image, mask)
        assert torch.equal(
            prediction.completed[~mask.expand_as(image)], image[~mask.expand_as(image)]
        )
        batch_metrics = metrics(prediction.completed, image, mask)
        for i, row in enumerate(batch_metrics):
            index = indices[count]
            row.update(validation_index=index, path=batch["path"][i])
            rows.append(row)
            if count < len(shown):
                strip = torch.cat((prediction.masked[i], prediction.completed[i], image[i]), dim=-1)
                path = output / f"sample-{index:04d}.png"
                save_image(strip, path)
                visible_rows.append((path, row))
                Image.fromarray((mask[i, 0].cpu().numpy() * 255).astype(np.uint8)).save(
                    output / f"mask-{index:04d}.png"
                )
            count += 1
    keys = [key for key in rows[0] if key not in {"validation_index", "path"}]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["images"] = len(rows)
    (output / "metrics.json").write_text(
        json.dumps({"summary": summary, "per_image": rows}, indent=2)
    )

    for part in range(2):
        canvas = Image.new("RGB", (792, 4 * 286 + 64), "#20242a")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 10), f"Stage 1 only | frozen checkpoint step {checkpoint_step}", fill="white"
        )
        for column, label in enumerate(["Masked input", "Stage 1 completion", "Ground truth"]):
            draw.text((12 + column * 256, 34), label, fill="white")
        for offset, (path, row) in enumerate(visible_rows[part * 4 : (part + 1) * 4]):
            y = 60 + offset * 286
            with Image.open(path) as strip:
                canvas.paste(strip, (12, y))
            draw.text(
                (12, y + 260),
                f"sample {row['validation_index']} | hole {row['hole_fraction']:.1%} | "
                f"hole PSNR {row['hole_psnr']:.2f} dB",
                fill="white",
            )
        canvas.save(output / f"comparison-{part + 1}.png")
    report = {
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_retained": not args.no_checkpoint_copy,
        "stage": 1,
        "batch_size_at_save": config.data.batch_size,
        "data_progress": data_progress,
        "validation_indices": indices,
        "shown_indices": shown,
        "mask_mode": config.data.mask_mode,
        "precision": "FP32",
        "device": torch.cuda.get_device_name(device),
        "inference_batch_size": 4,
        "summary": summary,
        "elapsed_seconds": time.monotonic() - start,
        "note": (
            "Quick 64-image held-out evaluation; known regions copied from ground truth; "
            "use hole metrics. Record-disjoint, not content-deduplicated."
        ),
    }
    (output / "test.json").write_text(json.dumps(report, indent=2))
    shutil.copy2(__file__, output / "evaluate.py")
    files = sorted(
        [
            *output.glob("*.png"),
            output / "metrics.json",
            output / "test.json",
            output / "evaluate.py",
        ]
    )
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (output / "COMPLETE.json").write_text(
        json.dumps({"status": "COMPLETE", "files": hashes}, indent=2)
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "checkpoint_step": checkpoint_step,
                "data_progress": data_progress,
                "summary": summary,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
