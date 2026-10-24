"""Exercise two real PPO updates and check PhysX ankle limits; no full evaluation."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.runtime_env import ROOT, environment, load_env, python_path
sys.path.insert(0, str(ROOT))
from legged_lab.paths import ik_bank_dir
from scripts.check_resources import check
from scripts.train_current import recipe_arguments


def main():
    load_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--output', type=Path, default=ROOT/'logs'/datetime.datetime.now().strftime('smoke_%Y%m%d_%H%M%S'))
    p.add_argument('--timeout', type=int, default=300)
    a = p.parse_args()
    errors = check(bank=ik_bank_dir())
    if errors:
        p.error('\n'.join(errors))
    out = a.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'configs/current.json').read_text())
    config.update(num_envs=a.num_envs, max_iterations=2, save_interval=1)
    args = recipe_arguments(config)
    index = args.index('--evaluation_interval')
    del args[index:index+2]
    name = out.name
    command = [str(python_path('train')), '-u', 'legged_lab/scripts/train.py', *args,
               '--seed', '42', '--headless', '--device', 'cuda:0', '--logger', 'tensorboard',
               '--run_name', name, '--experiment_name', 'deployment_smoke', '--from_scratch',
               '--ik_arm_bank', str(ik_bank_dir())]
    with (out/'train.log').open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment(), stdout=stream,
                                stderr=subprocess.STDOUT, timeout=a.timeout)
    log = (out/'train.log').read_text()
    runs = sorted((ROOT/'logs/deployment_smoke').glob('*_'+name))
    passed = result.returncode == 0 and '[ANKLE_VELOCITY_LIMITS_RAD_S]' in log and 'Learning iteration 1/' in log
    report = {'passed': passed, 'returncode': result.returncode, 'command': command,
              'log': str(out/'train.log'), 'run_dir': str(runs[-1]) if runs else None}
    if passed and runs:
        sys.path.insert(0, str(ROOT/'tools'))
        from run_scratch_training import health
        report['checkpoint_health'] = health(runs[-1]/'model_1.pt', std_bounds=(.1,.6))
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not passed:
        print(log[-7000:])
    raise SystemExit(not passed)


if __name__ == '__main__':
    main()
