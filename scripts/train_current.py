"""Launch the documented recipe; --dry-run prints the command without training."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_command(config, output, deployment_python, python=sys.executable):
    required = {'task', 'max_iterations', 'num_envs', 'evaluation_interval', 'save_interval',
                'auxiliary_supervision', 'teleop_standing', 'teleop_standing_fraction',
                'pelvis_height', 'elf3_symmetry'}
    if set(config) - required - {'description'} or not required.issubset(config):
        raise ValueError('Recipe has unknown or missing fields')
    command = [str(python), '-u', 'tools/run_walk_robust_training.py', '--output', str(output),
               '--deployment_python', str(deployment_python)]
    for key in sorted(required):
        value = config[key]
        if isinstance(value, bool):
            if value:
                command.append('--' + key)
        else:
            command.extend(['--' + key, str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ppo_symmetry_dynamics.json')
    parser.add_argument('--output', type=Path, required=True, help='New run directory; an existing run is never overwritten')
    parser.add_argument('--deployment-python', default=os.environ.get('ELF3_DEPLOYMENT_PYTHON', sys.executable))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    # Preserve venv symlinks; resolving a Python executable can bypass the venv.
    deployment_python = Path(args.deployment_python).expanduser().absolute()
    output = args.output.expanduser().absolute()
    command = build_command(config, output, deployment_python)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if (output/'launch.lock').exists():
        parser.error('This output already contains a run; choose a new --output')
    from scripts.check_resources import check
    errors = check(deployment=True)
    if errors:
        parser.error('\n'.join(errors))
    subprocess.run([str(deployment_python), '-c',
                    'import mujoco, onnx, onnxruntime, torch, yaml'], check=True, cwd=ROOT)
    os.chdir(ROOT)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
