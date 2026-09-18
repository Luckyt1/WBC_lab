#!/usr/bin/env bash
# Verify the GPU, ELF3 simulation, PPO, and student distillation on this host.
set -euo pipefail
WBC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$WBC_ROOT/activate_wbc.sh"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
cd "$WBC_ROOT/IsaacLab-Decoupled-WBC"
mkdir -p "$WBC_ROOT/environment_checks"
python -m pip check
python - <<'PY'
import importlib.metadata as m
import torch
for name in ('torch', 'torchvision', 'isaacsim', 'isaaclab', 'rsl-rl-lib', 'isaaclab-decoupled-wbc'):
    print(f'{name}: {m.version(name)}')
assert torch.cuda.is_available(), 'CUDA is unavailable'
x = torch.randn(256, 256, device='cuda')
assert torch.isfinite(x @ x).all()
print('GPU:', torch.cuda.get_device_name(0), 'CUDA:', torch.version.cuda)
PY
python -m pytest tests -q > "$WBC_ROOT/environment_checks/unit_tests.log" 2>&1
timeout 900 python tools/check_elf3_env.py --headless > "$WBC_ROOT/environment_checks/elf3_env.log" 2>&1
WBC_CHECK_RUN="host_setup_$(date +%Y%m%d_%H%M%S)"
timeout 900 bash scripts/train_teacher.sh --num_envs=8 --max_iterations=5 --save_interval=1 --run_name="$WBC_CHECK_RUN" > "$WBC_ROOT/environment_checks/teacher.log" 2>&1
HTD_RUN="$(find logs/elf3_flat -maxdepth 1 -type d -name "*_$WBC_CHECK_RUN" -printf '%f\n')"
test -n "$HTD_RUN"
export HTD_RUN HTD_CKPT=model_4.pt
timeout 900 bash scripts/train_student.sh --num_envs=8 --bc_steps=4 --max_steps=8 --save_interval=4 > "$WBC_ROOT/environment_checks/student.log" 2>&1
printf 'Verified teacher/student run: %s\n' "$HTD_RUN" | tee "$WBC_ROOT/environment_checks/result.txt"
