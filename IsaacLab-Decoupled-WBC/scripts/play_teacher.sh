#!/usr/bin/env bash
# Play a trained ELF3 teacher, loading its saved gains. Select the checkpoint:
#   HTD_RUN=<run_folder> HTD_CKPT=<model_xxx.pt> bash scripts/play_teacher.sh
# Activate the env first:  conda activate isaac_wbc
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

WBC_TASK="${WBC_TASK:-elf3_flat}"
HTD_RUN="${HTD_RUN:-}"
HTD_CKPT="${HTD_CKPT:-}"
if [ -z "$HTD_RUN" ] || [ -z "$HTD_CKPT" ]; then
    echo "[ERROR] Set HTD_RUN=<ELF3 run> HTD_CKPT=<model_N.pt>. Train an ELF3 teacher first." >&2
    exit 1
fi

if [ ! -f "logs/$WBC_TASK/$HTD_RUN/$HTD_CKPT" ]; then
    echo "[ERROR] Teacher checkpoint not found: logs/$WBC_TASK/$HTD_RUN/$HTD_CKPT" >&2
    echo "        Use HTD_RUN=<run_folder> HTD_CKPT=<model_xxx.pt> to select a different checkpoint." >&2
    exit 1
fi

echo "[INFO] Playing teacher checkpoint: logs/$WBC_TASK/$HTD_RUN/$HTD_CKPT"
python legged_lab/scripts/play_teacher.py \
    --task="$WBC_TASK" \
    --num_envs=1 \
    --load_run="$HTD_RUN" \
    --checkpoint="$HTD_CKPT" \
    "$@"
