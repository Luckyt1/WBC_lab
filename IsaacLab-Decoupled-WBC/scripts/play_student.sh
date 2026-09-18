#!/usr/bin/env bash
# Play a trained ELF3 student (JIT), loading its teacher's saved gains:
#   HTD_RUN=<run_folder> bash scripts/play_student.sh
# Activate the env first:  conda activate isaac_wbc
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

WBC_TASK="${WBC_TASK:-elf3_flat}"
HTD_RUN="${HTD_RUN:-}"
if [ -z "$HTD_RUN" ]; then
    echo "[ERROR] Set HTD_RUN=<ELF3 teacher run containing student_checkpoints/>. G1 example weights cannot control ELF3." >&2
    exit 1
fi

STUDENT_CKPT="logs/$WBC_TASK/$HTD_RUN/student_checkpoints/student_policy_jit.pt"
if [ ! -f "$STUDENT_CKPT" ]; then
    echo "[ERROR] Student checkpoint not found: $STUDENT_CKPT" >&2
    echo "        Use HTD_RUN=<run_folder> to select a different run." >&2
    exit 1
fi

echo "[INFO] Playing student checkpoint: $STUDENT_CKPT"
python legged_lab/scripts/play_student.py \
    --task="$WBC_TASK" \
    --load_run="$HTD_RUN" \
    --student_checkpoint="$STUDENT_CKPT" \
    --use_jit \
    --num_envs=1 \
    --max_steps=20000 \
    "$@"
