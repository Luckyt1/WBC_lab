"""Launch the IK v3 recipe; --dry-run does not import the simulator."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legged_lab.paths import ik_bank_dir


def build_command(config, output, deployment_python, ik_python, bank, python=sys.executable):
    required = {'task', 'max_iterations', 'num_envs', 'evaluation_interval', 'save_interval',
                'auxiliary_supervision', 'teleop_standing', 'teleop_standing_fraction',
                'pelvis_height', 'elf3_symmetry', 'liftoff_rewards'}
    if set(config) - required - {'description'} or not required.issubset(config):
        raise ValueError('Recipe has unknown or missing fields')
    command = [str(python), '-u', str(ROOT/'tools/run_walk_robust_training.py'),
               '--output', str(output), '--deployment_python', str(deployment_python),
               '--ik_python', str(ik_python), '--ik_arm_bank', str(bank)]
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
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ik_v3.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', type=Path, default=ik_bank_dir())
    parser.add_argument('--deployment-python', default=os.environ.get('ELF3_DEPLOYMENT_PYTHON', sys.executable))
    parser.add_argument('--ik-python', default=os.environ.get('ELF3_IK_PYTHON', sys.executable))
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--max-iterations', type=int, help='New PPO updates; also applies when resuming')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.max_iterations is not None:
        if args.max_iterations <= 0:
            parser.error('--max-iterations must be positive')
        config['max_iterations'] = args.max_iterations
    # absolute() preserves virtual-environment interpreter symlinks.
    deployment_python = Path(args.deployment_python).expanduser().absolute()
    ik_python = Path(args.ik_python).expanduser().absolute()
    output, bank = args.output.expanduser().absolute(), args.bank.expanduser().absolute()
    command = build_command(config, output, deployment_python, ik_python, bank)
    if args.resume_checkpoint:
        command += ['--resume_checkpoint', str(args.resume_checkpoint.expanduser().absolute())]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new or empty --output directory')
    from scripts.check_resources import check
    errors = check(deployment=True, bank=bank, amass=True)
    if errors:
        parser.error('\n'.join(errors))
    native_dir = ROOT/'legged_lab/envs/elf3_tang/wbc'
    for interpreter, imports in [(sys.executable, 'import torch'),
                                  (deployment_python, 'import mujoco, onnx, onnxruntime, torch, yaml'),
                                  (ik_python, 'import pinocchio, mujoco, onnxruntime, scipy, torch, yaml')]:
        subprocess.run([str(interpreter), '-c', imports], check=True, cwd=ROOT)
    for interpreter in (sys.executable, str(ik_python)):
        subprocess.run([str(interpreter), '-c',
                        'import sys; sys.path.insert(0, sys.argv[1]); import _arm_ruckig',
                        str(native_dir)], check=True, cwd=ROOT)
    os.chdir(ROOT)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
