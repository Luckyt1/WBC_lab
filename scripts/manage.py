"""Small, simulator-free dispatcher for deployment and training commands."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.runtime_env import ROOT, environment, load_env, project_path, python_path

HELP = """Usage: ./deploy.sh COMMAND [options]

  setup     Prepare Python environments and compile Ruckig (see --help)
  check     Validate resources, Python dependencies, CUDA and native extension
  train     Start the current recipe (see --help; supports --dry-run/--background)
  smoke     Run 2 PPO updates with 16 environments, without full evaluation
  export    Export an actor to ONNX and compare 32 samples against PyTorch
  status    Read a run: --output logs/RUN
  test      Run the CPU regression suite (extra pytest arguments accepted)
  pack      Create a relocatable archive (source only by default; --with-resources)

.env is loaded automatically. Resource paths may be relative to this checkout.
"""


def main():
    load_env()
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help', 'help'):
        print(HELP)
        return
    name, rest = args[0], args[1:]
    scripts = {'setup': 'setup_env.py', 'check': 'doctor.py', 'train': 'train_current.py',
               'smoke': 'smoke_train.py', 'pack': 'pack_release.py'}
    if name == 'status':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', type=Path, required=True)
        a = parser.parse_args(rest)
        path = project_path(a.output) / 'state.json'
        state = json.loads(path.read_text())
        keys = ('status', 'updated_at', 'completed_updates', 'target_total_iterations',
                'latest_checkpoint', 'latest_health', 'child_pid', 'eval_pid', 'stop_reason')
        print(json.dumps({k: state.get(k) for k in keys}, ensure_ascii=False, indent=2))
        if state.get('updated_at'):
            age = datetime.datetime.now().astimezone() - datetime.datetime.fromisoformat(state['updated_at'])
            print(f'Heartbeat age: {age.total_seconds():.0f}s')
        return
    if name not in scripts and name not in ('test', 'export'):
        raise SystemExit(f'Unknown command: {name}\n{HELP}')
    python = python_path('train') if name in ('train', 'smoke', 'test') else Path(sys.executable)
    if name == 'export':
        python = python_path('eval')
    # CLI help and dry-run work before installing the simulator.
    if name == 'train' and any(x in rest for x in ('--dry-run', '--help', '-h')):
        python = Path(sys.executable)
    if not python.is_file():
        raise SystemExit(f'Missing {python}; first run ./deploy.sh setup (or configure .env)')
    if name == 'test':
        command = [str(python), '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests', *rest]
    elif name == 'export':
        command = [str(python), str(ROOT/'tools/export_walk_actor_onnx.py'), *rest]
    else:
        command = [str(python), str(ROOT/'scripts'/scripts[name]), *rest]
    result = subprocess.run(command, cwd=ROOT, env=environment())
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
