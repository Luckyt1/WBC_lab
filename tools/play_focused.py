"""Play a guarded focused-run checkpoint from its live manifest."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', choices=('best', 'latest'), default='best',
                        help='best is regression-checked; latest may still be under training')
    parser.add_argument('--manifest', default=str(ROOT/'logs/training_launches/latest_focused_bxi31.json'))
    args, extra = parser.parse_known_args()
    state = json.loads(Path(args.manifest).read_text())
    path = Path(state.get(args.selection + '_checkpoint', state['baseline_checkpoint']))
    if not path.is_file():
        raise FileNotFoundError(path)
    print('Checkpoint:', path, flush=True)
    print('Selection:', args.selection, '(latest may be unevaluated)', flush=True)
    argv = [sys.executable, str(ROOT/'legged_lab/scripts/play.py'),
            '--task', 'walk_elf3_tang_bxi_focus', '--num_envs', '1',
            '--experiment_name', path.parent.parent.name,
            '--load_run', re.escape(path.parent.name) + '$',
            '--checkpoint', re.escape(path.name) + '$', *extra]
    os.chdir(ROOT)
    os.execv(sys.executable, argv)


if __name__ == '__main__':
    main()
