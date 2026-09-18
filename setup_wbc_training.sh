#!/usr/bin/env bash
# Complete the existing isaac_wbc environment for ELF3 WBC training.
set -euo pipefail
WBC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$WBC_ROOT/activate_wbc.sh"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=120
export PIP_BUILD_CONSTRAINT="$WBC_ROOT/wbc_build_constraints.txt"
export PIP_CONSTRAINT="$WBC_ROOT/wbc_training_constraints.txt"
exec > >(tee -a "$WBC_ROOT/setup_wbc_training.log") 2>&1
python -m pip uninstall -y wandb
python -m pip install setuptools==80.9.0 wheel==0.45.1 'click==8.1.7' 'sentry-sdk==1.43.0'
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaaclab[isaacsim,all]==2.2.0' 'isaacsim[all,extscache]==5.0.0' 'rsl-rl-lib==2.3.3' --extra-index-url https://pypi.nvidia.com
python -m pip install -e "$WBC_ROOT/IsaacLab-Decoupled-WBC" 'rsl-rl-lib==2.3.3' pytest
python -m pip check
python -m pip freeze --exclude isaaclab-decoupled-wbc > "$WBC_ROOT/isaac_wbc_requirements.lock.txt"
