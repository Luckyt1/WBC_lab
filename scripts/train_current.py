"""Launch the current no-waist-pose, 5/15 rad/s ankle recipe."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runtime_env import environment, load_env, project_path, python_path
from legged_lab.paths import ik_bank_dir

COMMON = {'task', 'max_iterations', 'num_envs', 'evaluation_interval', 'save_interval',
          'auxiliary_supervision', 'teleop_standing', 'teleop_standing_fraction',
          'elf3_symmetry', 'liftoff_rewards'}
HEIGHT = {'pelvis_height', 'pelvis_height_range', 'height_curriculum_start', 'height_curriculum_updates'}


def validate_config(config):
    if set(config) - COMMON - HEIGHT - {'description'} or not COMMON.issubset(config):
        raise ValueError('Recipe has unknown or missing fields')
    if ('pelvis_height' in config) == ('pelvis_height_range' in config):
        raise ValueError('Specify exactly one fixed height or height range')
    for key in ('max_iterations', 'num_envs', 'evaluation_interval', 'save_interval'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 <= config['teleop_standing_fraction'] <= 1:
        raise ValueError('standing fraction must be between 0 and 1')
    if 'pelvis_height_range' in config:
        low, high = config['pelvis_height_range']
        if not .4 <= low < high <= .8235 or config.get('height_curriculum_updates', 0) <= 0:
            raise ValueError('Invalid height range or curriculum length')


def recipe_arguments(config):
    validate_config(config)
    result = []
    for key in sorted(set(config) - {'description'}):
        value = config[key]
        if isinstance(value, bool):
            if value:
                result.append('--' + key)
        elif isinstance(value, list):
            result += ['--' + key, *map(str, value)]
        else:
            result += ['--' + key, str(value)]
    return result


def build_command(config, output, deployment_python, ik_python, bank, python=sys.executable):
    return [str(python), '-u', str(ROOT/'tools/run_walk_robust_training.py'),
            '--output', str(output), '--deployment_python', str(deployment_python),
            '--ik_python', str(ik_python), '--ik_arm_bank', str(bank), *recipe_arguments(config)]


def main():
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/current.json')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'logs'/datetime.datetime.now().strftime('no_waist_ankle_%Y%m%d_%H%M%S'))
    parser.add_argument('--bank', type=Path, default=ik_bank_dir())
    parser.add_argument('--deployment-python', default=str(python_path('eval')))
    parser.add_argument('--ik-python', default=str(python_path('ik')))
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--num-envs', type=int, help='Override parallel environments (current run: 12556)')
    parser.add_argument('--max-iterations', type=int, help='New PPO updates; also applies when resuming')
    parser.add_argument('--background', action='store_true', help='Detach supervisor and write supervisor.log')
    parser.add_argument('--dry-run', action='store_true', help='Print command; do not start any process')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ('max_iterations', 'num_envs'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.resume_checkpoint:
        config.pop('height_curriculum_start', None)
    try:
        validate_config(config)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    output, bank = args.output.expanduser().absolute(), args.bank.expanduser().absolute()
    command = build_command(config, output, project_path(args.deployment_python),
                            project_path(args.ik_python), bank, python_path('train'))
    if args.resume_checkpoint:
        checkpoint = args.resume_checkpoint.expanduser().absolute()
        command += ['--resume_checkpoint', str(checkpoint)]
        if not args.dry_run and (not checkpoint.is_file() or not (checkpoint.parent/'params/env.yaml').is_file()):
            parser.error('Resume needs a checkpoint with its original params/env.yaml and params/agent.yaml')
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new or empty --output directory')
    from scripts.doctor import check_all
    errors = check_all(bank=bank, deployment_python=project_path(args.deployment_python),
                       ik_python=project_path(args.ik_python))['errors']
    if errors:
        parser.error('\n'.join(errors))
    env = environment()
    if args.background:
        output.mkdir(parents=True, exist_ok=True)
        with (output/'supervisor.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (output/'launch.json').write_text(json.dumps({'supervisor_pid': child.pid, 'command': command}, indent=2)+'\n')
        time.sleep(1)
        if child.poll() is not None:
            parser.error(f'Supervisor exited {child.returncode}; see {output / "supervisor.log"}')
        print(f'Supervisor launched, PID {child.pid}; status: {output / "state.json"}')
    else:
        os.chdir(ROOT)
        os.execve(command[0], command, env)


if __name__ == '__main__':
    main()
