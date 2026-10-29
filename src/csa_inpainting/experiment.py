"""Small, explicit helpers for keeping experiments identifiable and separate."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch


def check_output_directory(output: str | Path, resume: str | Path | None) -> Path:
    destination = Path(output)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Experiment output is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()):
        latest = destination / "latest.pt"
        if resume is None or Path(resume).resolve() != latest.resolve() or not latest.is_file():
            raise ValueError(
                f"Experiment directory is not empty: {destination}. "
                "Choose another train.output, or resume this directory's latest.pt."
            )
    return destination


def record_invocation(
    destination: Path,
    config_path: str,
    overrides: list[str],
    resume: str | None,
    device: torch.device,
    resolved_config: dict[str, Any],
) -> None:
    """Append provenance, including source hashes for uncommitted research code."""
    source = Path(__file__).parent
    files = sorted(source.rglob("*.py"))
    hashes = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    git: dict[str, str | bool] = {}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git = {"revision": revision.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        pass  # Installed wheels need not live inside a Git checkout.
    entry = {
        # The remote training environment currently uses Python 3.10.
        "time_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchvision": version("torchvision"),
        "device": str(device),
        "config_source": str(Path(config_path).resolve()),
        "overrides": overrides,
        "config": resolved_config,
        "resume": str(Path(resume).resolve()) if resume else None,
        "git": git,
        "source_sha256": hashes,
    }
    if device.type == "cuda":
        entry["gpu"] = torch.cuda.get_device_name(device)
        entry["cuda"] = torch.version.cuda
    with (destination / "invocations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
