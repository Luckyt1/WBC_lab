#!/usr/bin/env bash
# Distill a student from a teacher checkpoint via the one-command BC -> DAgger pipeline
# (Behavior Cloning for the first --bc_steps, then DAgger up to --max_steps). The
# student matches the teacher checkpoint's actuator gains and saved command
# sampling/ranges automatically. Select a trained ELF3 teacher via env vars:
#   HTD_RUN=<teacher_run> HTD_CKPT=<model_xxx.pt> bash scripts/train_student.sh
# Re-running the same command resumes the newest compatible student checkpoint.
# A teacher/config hash marker prevents accidentally mixing student lineages.
# Activate the env first:  conda activate isaac_wbc
# For installation and usage, see ../README.md.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

WBC_TASK="${WBC_TASK:-elf3_flat}"
HTD_RUN="${HTD_RUN:-}"
HTD_CKPT="${HTD_CKPT:-}"
if [ -z "$HTD_RUN" ] || [ -z "$HTD_CKPT" ]; then
    echo "[ERROR] Set HTD_RUN=<teacher_run> HTD_CKPT=<model_N.pt>. Train an ELF3 teacher first; G1 weights are incompatible." >&2
    exit 1
fi
STUDENT_SEED="${STUDENT_SEED:-42}"
WANDB_RUN_VERSION="${WANDB_RUN_VERSION:-v1}"

if [ ! -f "logs/$WBC_TASK/$HTD_RUN/$HTD_CKPT" ]; then
    echo "[ERROR] Teacher checkpoint not found: logs/$WBC_TASK/$HTD_RUN/$HTD_CKPT" >&2
    echo "        Use HTD_RUN=<run_folder> HTD_CKPT=<model_xxx.pt> to select a different checkpoint." >&2
    exit 1
fi
if [ ! -f "logs/$WBC_TASK/$HTD_RUN/params/env.yaml" ]; then
    echo "[ERROR] Teacher environment config not found: logs/$WBC_TASK/$HTD_RUN/params/env.yaml" >&2
    echo "        Refusing to train with mismatched fallback gains or command sampling." >&2
    exit 1
fi

echo "[INFO] Training student from teacher checkpoint: logs/$WBC_TASK/$HTD_RUN/$HTD_CKPT"
python legged_lab/scripts/train_student.py \
    --task="$WBC_TASK" \
    --load_run="$HTD_RUN" \
    --checkpoint="$HTD_CKPT" \
    --num_envs=1024 \
    --bc_steps=250000 \
    --max_steps=600000 \
    --student_lr=5e-4 \
    --seed="$STUDENT_SEED" \
    --save_interval=5000 \
    --keep_checkpoints=3 \
    --wandb_run_version="$WANDB_RUN_VERSION" \
    --auto_resume \
    --logger=tensorboard \
    --headless \
    "$@"
