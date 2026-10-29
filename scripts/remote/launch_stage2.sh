#!/bin/bash
# Usage: CSA_GPU_UUID=... TORCH_HOME=... bash launch_stage2.sh RUNTIME PYTHON BATCH GiB --init-stage1|--resume CHECKPOINT
set -euo pipefail
runtime=$(realpath "${1:?Runtime required}")
python=${2:?Python required}
batch_size=${3:?Batch size required}
memory_budget=${4:?Allocator budget in GiB required}
initialization=${5:?Use --init-stage1 or --resume}
checkpoint=${6:?Checkpoint required}
case "$initialization" in
    --init-stage1|--resume) ;;
    *) printf '%s\n' 'Use --init-stage1 or --resume' >&2; exit 2 ;;
esac
test -f "$checkpoint"
export CUDA_VISIBLE_DEVICES="${CSA_GPU_UUID:?GPU UUID required}"
export TORCH_HOME="${TORCH_HOME:?Existing pretrained-weight cache required}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$runtime/source:$runtime"
exec 9>"$runtime/../../.gpu-${CUDA_VISIBLE_DEVICES}-csa.lock"
flock -n 9

"$python" - "$CUDA_VISIBLE_DEVICES" "$memory_budget" <<'PY'
import subprocess, sys
free = int(subprocess.check_output([
    'nvidia-smi', '--id='+sys.argv[1], '--query-gpu=memory.free',
    '--format=csv,noheader,nounits',
], text=True).strip()) / 1024
required = float(sys.argv[2]) + 3
if free < required:
    raise SystemExit(f'Need {required:g} GiB free; currently {free:.2f} GiB')
print(f'GPU free before stage 2: {free:.2f} GiB', flush=True)
PY

child=''
request_stop() {
    if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi
}
trap request_stop TERM INT
d_args=(--image-discriminator "${CSA_IMAGE_DISCRIMINATOR:-patch}"
        --image-discriminator-chunk-size "${CSA_IMAGE_D_CHUNK_SIZE:-8}")
if [[ -n "${CSA_IMAGE_D_PRETRAINED:-}" ]]; then
    d_args+=(--image-discriminator-pretrained "$CSA_IMAGE_D_PRETRAINED")
fi
"$python" -u "$runtime/csa_remote_train.py" train --runtime "$runtime" \
    --stage stage2 --batch-size "$batch_size" --amp bf16 \
    --memory-budget-gib "$memory_budget" --mask-mode random \
    --workers "${CSA_DATA_WORKERS:-32}" --save-every "${CSA_SAVE_EVERY:-25}" \
    --preview-every "${CSA_PREVIEW_EVERY:-500}" --log-every "${CSA_LOG_EVERY:-1}" \
    --epochs "${CSA_EPOCHS:-120}" --decay-start "${CSA_DECAY_START:-20}" \
    --reconstruction-weight 0.1 --lpips-weight 1 --lpips-net vgg \
    --lpips-batch-size "${CSA_LPIPS_BATCH_SIZE:-8}" \
    "${d_args[@]}" \
    "$initialization" "$checkpoint" &
child=$!
printf '%s\n' "$child" > "$runtime/train.pid"
set +e
# A signal may interrupt wait while the trainer is still finishing its batch
# and saving. Repeated signals must not publish a premature terminal state.
while true; do
    wait "$child"
    task_exit=$?
    if ! kill -0 "$child" 2>/dev/null; then break; fi
done
printf '%s\n' "$task_exit" > "$runtime/exit_code"
"$python" - "$runtime" "$task_exit" <<'PY'
import json, sys, time
from pathlib import Path
runtime, code = Path(sys.argv[1]), int(sys.argv[2])
path = runtime/'training-state.json'
state = json.loads(path.read_text()) if path.exists() else {}
if state.get('state') != 'STOPPED':
    state['state'] = ('COMPLETED' if (runtime/'run/COMPLETE').exists()
                      else 'FAILED' if code else 'EXITED_EARLY')
state.update(exit_code=code, updated_unix=time.time(), stage='stage2')
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(state, indent=2))
temporary.replace(path)
PY
exit "$task_exit"
