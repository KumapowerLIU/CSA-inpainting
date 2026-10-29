"""Wait for one idle H100 on this host, then launch the prepared CSA run once."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

runtime = Path(sys.argv[1]).resolve()
lock = (runtime / "queue.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state_path = runtime / "queue-state.json"


def write_state(state, **kwargs):
    result = dict(state=state, updated_unix=time.time(), queue_pid=os.getpid(), **kwargs)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(state_path)
    print(json.dumps(result), flush=True)


def idle_gpus():
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True,
        timeout=15,
    ).splitlines()
    occupied = {row.strip() for row in processes}
    candidates = []
    for row in rows.splitlines():
        index, uuid, name, used, utilization = [x.strip() for x in row.split(",")]
        if "H100" in name and uuid not in occupied and int(used) < 100 and int(utilization) < 5:
            candidates.append((int(index), uuid))
    return sorted(
        candidates,
        key=lambda item: [1, 2, 3, 0].index(item[0]) if item[0] in [1, 2, 3, 0] else item[0] + 4,
    )


previous = set()
write_state("WAITING_FOR_IDLE_GPU")
while True:
    try:
        candidates = idle_gpus()
    except (subprocess.SubprocessError, ValueError) as error:
        write_state("WAITING_FOR_GPU_QUERY", error=str(error))
        previous = set()
        time.sleep(30)
        continue
    stable = [item for item in candidates if item[1] in previous]
    if stable:
        index, uuid = stable[0]
        env = os.environ.copy()
        env["CSA_GPU_UUID"] = uuid
        with (runtime / "train.log").open("a") as log:
            child = subprocess.Popen(
                ["bash", str(runtime / "launch.sh")],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            write_state("RUNNING", gpu_index=index, gpu_uuid=uuid, launcher_pid=child.pid)
            result = child.wait()
        write_state(
            "COMPLETED" if result == 0 else "FAILED",
            exit_code=result,
            gpu_index=index,
            gpu_uuid=uuid,
            launcher_pid=child.pid,
        )
        sys.exit(result)
    previous = {uuid for _, uuid in candidates}
    time.sleep(30)
