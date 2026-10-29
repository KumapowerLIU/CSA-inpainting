#!/bin/bash
# Usage: CSA_GPU_UUID=... bash launch_stage1.sh RUNTIME PYTHON [BATCH] [GiB] [CHECKPOINT]
set -euo pipefail
runtime=$(realpath "${1:?Runtime directory required}")
python=${2:?Python executable required}
batch_size=${3:-32}
memory_budget=${4:-12}
resume=${5:-}
workers=${CSA_DATA_WORKERS:-8}
save_every=${CSA_SAVE_EVERY:-100}
reconstruction_weight=${CSA_RECONSTRUCTION_WEIGHT:-1}
lpips_weight=${CSA_LPIPS_WEIGHT:-0}
lpips_net=${CSA_LPIPS_NET:-vgg}
lpips_batch_size=${CSA_LPIPS_BATCH_SIZE:-8}
allow_loss_change=${CSA_ALLOW_LOSS_CHANGE:-0}
case "$allow_loss_change" in
    0|1) ;;
    *) printf '%s\n' 'CSA_ALLOW_LOSS_CHANGE must be 0 or 1' >&2; exit 2 ;;
esac
resume_args=()
if [[ -n "$resume" ]]; then
    resume_args=(--resume "$resume" --allow-batch-size-change)
fi
if [[ "${CSA_ALLOW_DATASET_CHANGE:-0}" == 1 ]]; then
    if [[ -z "$resume" ]]; then
        printf '%s\n' 'CSA_ALLOW_DATASET_CHANGE requires a resume checkpoint' >&2
        exit 2
    fi
    resume_args+=(--allow-dataset-change)
fi
if [[ "$allow_loss_change" == 1 ]]; then
    if [[ -z "$resume" ]]; then
        printf '%s\n' 'CSA_ALLOW_LOSS_CHANGE requires a resume checkpoint' >&2
        exit 2
    fi
    resume_args+=(--allow-loss-change)
fi
loss_args=(--reconstruction-weight "$reconstruction_weight"
           --lpips-weight "$lpips_weight" --lpips-net "$lpips_net"
           --lpips-batch-size "$lpips_batch_size")
export CUDA_VISIBLE_DEVICES="${CSA_GPU_UUID:?GPU UUID required}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$runtime/source"
exec 9>"$runtime/../../.gpu-${CUDA_VISIBLE_DEVICES}-csa.lock"
flock -n 9

finish() {
    task_exit=$?
    trap - EXIT
    printf '%s\n' "$task_exit" > "$runtime/exit_code"
    "$python" - "$runtime" "$task_exit" <<'PY'
import json,sys,time
from pathlib import Path
runtime,code=Path(sys.argv[1]),int(sys.argv[2])
path=runtime/'training-state.json'
state=json.loads(path.read_text()) if path.exists() else {}
if code:
    if state.get('state') != 'STOPPED':
        state['state']='FAILED'
else:
    state['state']='COMPLETED' if (runtime/'run/COMPLETE').exists() else 'EXITED_EARLY'
state.update(exit_code=code,updated_unix=time.time(),stage='stage1')
temporary=path.with_suffix('.tmp')
temporary.write_text(json.dumps(state,indent=2))
temporary.replace(path)
PY
    exit "$task_exit"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Require headroom beyond the allocator limit before starting a new process.
"$python" - "$CUDA_VISIBLE_DEVICES" "$memory_budget" <<'PY'
import subprocess,sys
free=int(subprocess.check_output(['nvidia-smi','--id='+sys.argv[1],
    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
required=float(sys.argv[2])+3
if free < required*1024:
    raise SystemExit(f'Need at least {required:g} GiB free before launching; currently {free/1024:.2f} GiB')
print(f'Shared GPU free before launch: {free/1024:.2f} GiB',flush=True)
PY
"$python" -u "$runtime/csa_remote_train.py" probe --runtime "$runtime" \
    --stage stage1 --batch-size "$batch_size" --amp bf16 --memory-budget-gib "$memory_budget" \
    --mask-mode random --workers "$workers" --steps 2 "${loss_args[@]}" \
    > "$runtime/probe.log" 2>&1
"$python" -u "$runtime/csa_remote_train.py" train --runtime "$runtime" \
    --stage stage1 --batch-size "$batch_size" --amp bf16 --memory-budget-gib "$memory_budget" \
    --mask-mode random --workers "$workers" --save-every "$save_every" --preview-every 500 \
    "${loss_args[@]}" "${resume_args[@]}"
