import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torch import Tensor
from torch.utils.data import DataLoader

from .networks import InpaintingModel


def save_image(image: Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = ((image.detach().cpu().float().clamp(-1, 1) + 1) * 127.5).round()
    Image.fromarray(pixels.to(torch.uint8).permute(1, 2, 0).numpy()).save(path)


def metrics(prediction: Tensor, target: Tensor, mask: Tensor) -> list[dict[str, float]]:
    """Per-image metrics in [0,1]; L1/L2 are fractions, PSNR is dB.

    Full-image values are computed on the completed composite. Hole metrics
    normalize by missing pixels * channels. SSIM is RGB, Gaussian, sigma=1.5,
    11x11, population covariance, averaged across channels (scikit-image).
    """
    prediction = (prediction.detach().cpu().float().clamp(-1, 1) + 1) / 2
    target = (target.detach().cpu().float() + 1) / 2
    mask = mask.detach().cpu().bool()
    rows = []
    for p, t, m in zip(prediction, target, mask, strict=True):
        error = p - t
        l2 = error.square().mean().item()
        hole_error = error[m.expand_as(error)]
        if hole_error.numel() == 0:
            raise ValueError("Hole metrics require nonempty masks")
        hole_l2 = hole_error.square().mean().item()
        rows.append(
            {
                "l1": error.abs().mean().item(),
                "l2": l2,
                "psnr": float(-10 * np.log10(max(l2, 1e-12))),
                "ssim": float(
                    structural_similarity(
                        t.permute(1, 2, 0).numpy(),
                        p.permute(1, 2, 0).numpy(),
                        data_range=1,
                        channel_axis=2,
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                    )
                ),
                "hole_l1": hole_error.abs().mean().item(),
                "hole_l2": hole_l2,
                "hole_psnr": float(-10 * np.log10(max(hole_l2, 1e-12))),
                "hole_fraction": m.float().mean().item(),
            }
        )
    return rows


@torch.inference_mode()
def evaluate(
    model: InpaintingModel,
    loader: DataLoader,
    device: torch.device,
    output_dir: str | Path | None = None,
    save_limit: int = 8,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows = []
    try:
        for batch in loader:
            image, mask = batch["image"].to(device), batch["mask"].to(device)
            output = model(image, mask)
            batch_metrics = metrics(output.completed, image, mask)
            for i, row in enumerate(batch_metrics):
                row["path"] = batch["path"][i]
                if output_dir and len(rows) < save_limit:
                    preview = torch.cat(
                        (
                            output.masked[i],
                            output.rough[i],
                            output.completed[i],
                            image[i],
                        ),
                        dim=-1,
                    )
                    save_image(preview, Path(output_dir) / f"{len(rows):05d}.png")
                rows.append(row)
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("Validation dataset is empty")
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "path"}
    summary["images"] = len(rows)
    if output_dir:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metrics.json").write_text(
            json.dumps({"summary": summary, "per_image": rows}, indent=2) + "\n"
        )
    return summary
