#!/usr/bin/env bash
# Start a fresh ELF3 WBC teacher run. Extra arguments go to train_teacher.py.
set -euo pipefail
WBC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WBC_TASK="${WBC_TASK:-elf3_wbc}"
if [[ "${1:-}" == --dry-run ]]; then
    shift
    printf 'WBC_TASK=%q bash %q ' "$WBC_TASK" "$WBC_ROOT/IsaacLab-Decoupled-WBC/scripts/train_teacher.sh"
    if (( $# )); then printf '%q ' "$@"; fi
    printf '\n'
    exit 0
fi
source "$WBC_ROOT/activate_wbc.sh"
export PYTHONUNBUFFERED=1
exec bash scripts/train_teacher.sh "$@"
