#!/usr/bin/env bash
# Usage: source ./activate_wbc.sh
# Override WBC_CONDA_ROOT or WBC_CONDA_ENV for a different Conda installation.
_wbc_activate() {
    local wbc_root wbc_conda_root
    wbc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return
    if [[ -n "${WBC_CONDA_ROOT:-}" ]]; then
        wbc_conda_root="$WBC_CONDA_ROOT"
    elif command -v conda >/dev/null 2>&1; then
        wbc_conda_root="$(conda info --base)" || return
    else
        wbc_conda_root="$HOME/miniconda3"
    fi
    if [[ ! -f "$wbc_conda_root/etc/profile.d/conda.sh" ]]; then
        echo "Conda not found at $wbc_conda_root. Run setup_wbc_conda.sh or set WBC_CONDA_ROOT." >&2
        return 1
    fi
    source "$wbc_conda_root/etc/profile.d/conda.sh" || return
    conda activate "${WBC_CONDA_ENV:-isaac_wbc}" || return
    cd "$wbc_root/IsaacLab-Decoupled-WBC" || return
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
}
_wbc_activate
