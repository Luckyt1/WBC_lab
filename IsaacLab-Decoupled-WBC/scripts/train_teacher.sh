#!/usr/bin/env bash
# Train the ELF3 teacher. WBC_TASK=g1_flat selects the legacy G1 task.
# Activate the env first:  conda activate isaac_wbc
# Extra flags pass through, e.g.:  bash scripts/train_teacher.sh --logger=tensorboard
# For training and resuming, see ../README.md.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

python legged_lab/scripts/train_teacher.py \
    --task="${WBC_TASK:-elf3_flat}" \
    --run_name=elf3_dof31 \
    --logger=tensorboard \
    --num_envs=1024 \
    --max_iterations=60000 \
    --save_interval=1000 \
    --headless \
    "$@"
