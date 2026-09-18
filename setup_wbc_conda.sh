#!/usr/bin/env bash
# Install Conda and an isolated Python environment for this workspace.
set -euo pipefail

CONDA_PREFIX_WBC="${WBC_CONDA_ROOT:-$HOME/miniconda3}"
WBC_CONDA_ENV="${WBC_CONDA_ENV:-isaac_wbc}"
INSTALLER=/tmp/Miniconda3-py311_26.7.1-1-Linux-x86_64.sh
INSTALLER_SHA256=a6f98e6e19d5b7897ae887cd6af931eb863459f86ffd1a09cc370124cab0993e

if [ ! -x "$CONDA_PREFIX_WBC/bin/conda" ]; then
    wget --progress=dot:giga --timeout=60 --tries=3 -O "$INSTALLER" \
        https://repo.anaconda.com/miniconda/Miniconda3-py311_26.7.1-1-Linux-x86_64.sh
    printf '%s  %s\n' "$INSTALLER_SHA256" "$INSTALLER" | sha256sum -c -
    bash "$INSTALLER" -b -p "$CONDA_PREFIX_WBC"
fi

source "$CONDA_PREFIX_WBC/etc/profile.d/conda.sh"
if ! conda run -n "$WBC_CONDA_ENV" python --version >/dev/null 2>&1; then
    conda create -y -n "$WBC_CONDA_ENV" --override-channels -c conda-forge python=3.11 pip git
fi
conda run -n "$WBC_CONDA_ENV" python --version
