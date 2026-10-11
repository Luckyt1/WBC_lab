"""Sequential independent stage evaluation, ONNX export and deployment-adapter check."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_lab.paths import deployment_framework, deployment_mod, mujoco_xml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_focused_validation import now, read, write, sha
from tools.teleop_curriculum import checkpoint_stage
from tools.teleop_curriculum_gates import gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deployment-python', type=Path, required=True)
    args = parser.parse_args()
    from tools.evaluation_runtime import deployment_python
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    python = deployment_python(args.deployment_python, out)
    checkpoint = args.checkpoint.resolve(strict=True)
    stage = checkpoint_stage(checkpoint)
    subprocess.run([sys.executable, '-u', 'tools/evaluate_walk_tang.py',
                    '--task', 'walk_elf3_tang_bxi_robust', '--checkpoint', str(checkpoint),
                    '--output', str(out), '--teleop-stage', stage,
                    '--warmup_s', '4', '--measure_s', '56', '--headless'], cwd=ROOT, check=True)
    model = out/'policy.onnx'
    subprocess.run([str(python), 'tools/export_walk_actor_onnx.py', '--checkpoint', str(checkpoint),
                    '--output', str(model)], cwd=ROOT, check=True)
    framework = deployment_framework()
    env = dict(os.environ, PYTHONPATH=str(framework)+os.pathsep+os.environ.get('PYTHONPATH', ''))
    result = subprocess.run([str(python), 'tools/evaluate_walk_mujoco_standing.py',
        '--mod', str(deployment_mod()),
        '--xml', str(mujoco_xml()),
        '--onnx', str(model), '--output', str(out/'mujoco_standing.json')], cwd=ROOT, env=env)
    if result.returncode not in (0, 1) or not (out/'mujoco_standing.json').is_file():
        raise RuntimeError('MuJoCo evaluation did not produce a valid report')
    evidence = {key: dict(path=str(out/name), sha256=sha(out/name)) for key, name in
                [('isaac', 'report.json'), ('mujoco', 'mujoco_standing.json'), ('onnx', 'policy.onnx')]}
    checked = gate(read(out/'report.json'), read(out/'mujoco_standing.json'), stage,
                   sha(checkpoint), evidence['onnx']['sha256'])
    write(out/'curriculum_gate.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
          stage=stage, completed_at=now(), gate=checked, evidence=evidence))


if __name__ == '__main__':
    main()
