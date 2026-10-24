"""Check the deployable checkout without starting an Isaac simulation."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.runtime_env import environment, load_env, python_path
from scripts.check_resources import check


def check_all(bank=None, resources_only=False, deployment_python=None, ik_python=None, verify_bundle=False):
    load_env()
    report = {'checked_at': datetime.datetime.now().astimezone().isoformat(), 'root': str(ROOT),
              'checks': [], 'errors': []}
    try:
        errors = check(deployment=True, bank=bank, amass=True)
    except Exception as exc:
        errors = [f'Resource validation failed: {exc}']
    report['errors'].extend(errors)
    report['checks'].append({'name': 'resources', 'passed': not errors})
    if verify_bundle:
        manifest = ROOT/'BUNDLE_MANIFEST.json'
        if not manifest.is_file():
            report['errors'].append('BUNDLE_MANIFEST.json missing; use an extracted deployment archive')
        else:
            for name, expected in json.loads(manifest.read_text())['sha256'].items():
                path = ROOT/name
                if not path.is_file() or not path.resolve().is_relative_to(ROOT):
                    report['errors'].append(f'Missing/unsafe bundle file: {name}')
                elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    report['errors'].append(f'Bundle checksum mismatch: {name}')
    if not resources_only:
        native = str(ROOT/'legged_lab/envs/elf3_tang/wbc')
        common = f'import sys; assert sys.version_info[:2] == (3,11); sys.path.insert(0, {native!r}); import _arm_ruckig; '
        programs = [
            ('train', python_path('train'), common +
             "import torch, numpy, yaml, scipy, tensorboard, onnx, git, gymnasium, rsl_rl, importlib.util, importlib.metadata; "
             "assert torch.cuda.is_available(), 'CUDA unavailable'; "
             "assert importlib.util.find_spec('isaaclab'), 'isaaclab missing'; "
             "assert importlib.util.find_spec('isaacsim'), 'isaacsim missing'; "
             "assert importlib.metadata.version('isaaclab') == '2.2.0', 'expected Isaac Lab 2.2.0'; "
             "assert importlib.metadata.version('isaacsim').startswith('5.0.0'), 'expected Isaac Sim 5.0.0'; "
             f"assert rsl_rl.__file__.startswith({str(ROOT/'rsl_rl')!r}), 'wrong RSL-RL source'; "
             "print('CUDA:', torch.cuda.get_device_name(0), '| torch:', torch.__version__, '| RSL:', rsl_rl.__file__)"),
            ('eval', deployment_python or python_path('eval'), common +
             "import torch, numpy, mujoco, onnx, onnxruntime, yaml; "
             "from bxi_example_py_elf3.framework.inference import JointPolicy, ModelSpec, PolicyJointContract; "
             "from legged_lab.paths import mujoco_xml; model=mujoco.MjModel.from_xml_path(str(mujoco_xml())); "
             "print('MuJoCo:', mujoco.__version__, '| model joints:', model.njnt)"),
            ('ik', ik_python or python_path('ik'), common +
             "import torch, numpy, scipy, pinocchio, mujoco, onnxruntime, yaml; "
             "from tools.elf3_ik_core import new_solver; _, solver, _, _ = new_solver(); "
             "print('Pinocchio:', pinocchio.__version__, '| IK nq:', solver.model.nq)")]
        for name, python, program in programs:
            try:
                result = subprocess.run([str(python), '-c', program], cwd=ROOT, env=environment(),
                                        text=True, capture_output=True, timeout=90)
                detail = (result.stdout + result.stderr).strip()
                passed = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired) as exc:
                detail, passed = str(exc), False
            report['checks'].append({'name': name, 'passed': passed, 'detail': detail})
            if not passed:
                report['errors'].append(f'{name} environment: {detail}')
    report['passed'] = not report['errors']
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resources-only', action='store_true')
    p.add_argument('--verify-bundle', action='store_true')
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    report = check_all(resources_only=a.resources_only, verify_bundle=a.verify_bundle)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    raise SystemExit(not report['passed'])


if __name__ == '__main__':
    main()
