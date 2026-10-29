"""Exercise process lifetime and signal handling without CUDA or training data."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).parents[1] / "scripts/remote/launch_stage2.sh"


def wait_for(predicate, process, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if process.poll() is not None:
            raise AssertionError(f"Launcher exited before expected event: {process.returncode}")
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for fake trainer")
        time.sleep(0.01)


def lock_is_held(path):
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False


@pytest.fixture
def launch(tmp_path):
    runtime = tmp_path / "experiment/runtime"
    runtime.mkdir(parents=True)
    parent = tmp_path / "parent.pt"
    parent.write_text("fake checkpoint")
    commands = tmp_path / "bin"
    commands.mkdir()
    for name, source in {
        "nvidia-smi": "#!/bin/sh\nprintf '81920\\n'\n",
        "flock": (
            f"#!{sys.executable}\nimport fcntl, sys\n"
            "try:\n    fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "except BlockingIOError:\n    sys.exit(1)\n"
        ),
    }.items():
        path = commands / name
        path.write_text(source)
        path.chmod(0o755)
    (runtime / "csa_remote_train.py").write_text(
        "import json, os, signal, sys, time\n"
        "from pathlib import Path\n"
        "runtime = Path(sys.argv[sys.argv.index('--runtime') + 1])\n"
        "mode = os.environ['CSA_TEST_MODE']\n"
        "received = 0\n"
        "def stop(signum, frame):\n"
        "    global received\n"
        "    received += 1\n"
        "    (runtime/'signals').write_text(str(received))\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "signal.signal(signal.SIGINT, stop)\n"
        "(runtime/'started').write_text(str(os.getpid()))\n"
        "if mode == 'stop':\n"
        "    while not received or not (runtime/'allow-save').exists():\n"
        "        time.sleep(0.01)\n"
        "    (runtime/'saved.pt').write_text('complete checkpoint')\n"
        "    (runtime/'training-state.json').write_text(json.dumps({'state': 'STOPPED'}))\n"
        "elif mode == 'complete':\n"
        "    (runtime/'run').mkdir()\n"
        "    (runtime/'run/COMPLETE').write_text('done')\n"
        "elif mode == 'fail':\n"
        "    sys.exit(7)\n"
    )
    processes = []

    def start(mode):
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("CSA_")
        }
        environment.update(
            PATH=str(commands) + os.pathsep + environment["PATH"],
            CSA_GPU_UUID="GPU-fake",
            CSA_TEST_MODE=mode,
            TORCH_HOME=str(tmp_path),
        )
        process = subprocess.Popen(
            [
                "bash",
                str(LAUNCHER),
                str(runtime),
                sys.executable,
                "4",
                "12",
                "--init-stage1",
                str(parent),
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        processes.append(process)
        return process, runtime, tmp_path / ".gpu-GPU-fake-csa.lock"

    yield start
    for process in processes:
        # Always release a fake child held at the save barrier, including after
        # an assertion failure where its launcher may already have exited.
        (runtime / "allow-save").touch()
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=3)


@pytest.mark.parametrize(
    "mode,code,state",
    [("complete", 0, "COMPLETED"), ("fail", 7, "FAILED"), ("early", 0, "EXITED_EARLY")],
)
def test_launcher_records_actual_child_exit_and_releases_lock(launch, mode, code, state):
    process, runtime, lock = launch(mode)
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == code, stdout + stderr
    assert int((runtime / "exit_code").read_text()) == code
    metadata = json.loads((runtime / "training-state.json").read_text())
    assert metadata["state"] == state and metadata["exit_code"] == code
    assert not lock_is_held(lock)


@pytest.mark.parametrize(
    "sent_signals",
    [(signal.SIGTERM,), (signal.SIGTERM, signal.SIGTERM), (signal.SIGTERM, signal.SIGINT)],
)
def test_signals_wait_for_checkpoint_before_launcher_exit_or_lock_release(launch, sent_signals):
    process, runtime, lock = launch("stop")
    wait_for(lambda: (runtime / "started").exists(), process)
    assert lock_is_held(lock)

    def signal_received(count):
        try:
            return int((runtime / "signals").read_text()) >= count
        except (FileNotFoundError, ValueError):
            return False

    for count, signum in enumerate(sent_signals, start=1):
        process.send_signal(signum)
        wait_for(lambda: signal_received(count), process)
    # The child has acknowledged TERM but cannot finish its checkpoint yet.
    time.sleep(0.1)
    assert process.poll() is None
    assert not (runtime / "exit_code").exists()
    assert not (runtime / "saved.pt").exists()
    assert lock_is_held(lock)
    (runtime / "allow-save").touch()
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stdout + stderr
    assert (runtime / "saved.pt").read_text() == "complete checkpoint"
    assert json.loads((runtime / "training-state.json").read_text())["state"] == "STOPPED"
    assert int((runtime / "exit_code").read_text()) == 0
    assert not lock_is_held(lock)


def test_gpu_lock_prevents_a_second_trainer_from_starting(launch):
    first, runtime, lock = launch("stop")
    wait_for(lambda: (runtime / "started").exists(), first)
    first_pid = (runtime / "train.pid").read_text()
    second, _, _ = launch("complete")
    second.communicate(timeout=5)
    assert second.returncode != 0
    assert first.poll() is None and lock_is_held(lock)
    assert (runtime / "train.pid").read_text() == first_pid
    assert not (runtime / "run/COMPLETE").exists()
    first.send_signal(signal.SIGTERM)
    (runtime / "allow-save").touch()
    first.communicate(timeout=5)
    assert first.returncode == 0
    assert not lock_is_held(lock)
