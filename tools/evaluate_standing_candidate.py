"""Run Isaac locomotion gates and independent deployment standing gates."""
import argparse
import os
from pathlib import Path
from legged_lab.paths import deployment_framework, deployment_mod, mujoco_xml
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deployment_python', type=Path, required=True)
    parser.add_argument('--task', choices=('walk_elf3_tang_bxi_still', 'walk_elf3_tang_bxi_robust'),
                        default='walk_elf3_tang_bxi_still')
    parser.add_argument('--include-mixed-yaw', action='store_true')
    parser.add_argument('--gait-diagnostics', action='store_true')
    parser.add_argument('--auxiliary-probe', action='store_true')
    parser.add_argument('--reuse-isaac', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reuse_isaac:
        from evaluation_runtime import validate_saved_isaac
        validate_saved_isaac(args.checkpoint, args.output, args.task,
                             args.include_mixed_yaw, args.gait_diagnostics, args.auxiliary_probe)
    else:
        subprocess.run([sys.executable, '-u', 'tools/evaluate_walk_tang.py',
                        '--task', args.task, '--checkpoint', str(args.checkpoint),
                        '--output', str(args.output), '--arm-guard', '--headless']
                       + (['--include-mixed-yaw'] if args.include_mixed_yaw else [])
                       + (['--gait-diagnostics'] if args.gait_diagnostics else [])
                       + (['--auxiliary-probe'] if args.auxiliary_probe else []), cwd=root, check=True)
    model = args.output/'policy.onnx'
    subprocess.run([str(args.deployment_python), 'tools/export_walk_actor_onnx.py',
                    '--checkpoint', str(args.checkpoint), '--output', str(model)], cwd=root, check=True)
    framework = deployment_framework()
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(framework) + os.pathsep + environment.get('PYTHONPATH', '')
    result = subprocess.run([str(args.deployment_python), 'tools/evaluate_walk_mujoco_standing.py',
                             '--mod', str(deployment_mod()),
                             '--xml', str(mujoco_xml()),
                             '--onnx', str(model), '--output', str(args.output/'mujoco_standing.json')],
                            cwd=root, env=environment)
    # A failed acceptance gate is a valid evaluation; a missing report is an error.
    if result.returncode not in (0, 1) or not (args.output/'mujoco_standing.json').is_file():
        raise RuntimeError(f'MuJoCo evaluator failed: {result.returncode}')


if __name__ == '__main__':
    main()
