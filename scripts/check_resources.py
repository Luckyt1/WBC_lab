"""Check external inputs without initializing Isaac Sim or touching the GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legged_lab.paths import robot_asset_dir, amass_root, deployment_framework, mujoco_xml


def check(deployment=False):
    errors = []
    asset = robot_asset_dir()
    expected = json.loads((ROOT/'legged_lab/assets/elf3_bxi_31/control_parameters.json').read_text())
    for relative in ['usd/elf3.usd', 'control_parameters.json', 'source/' + expected['urdf_relative_path']]:
        if not (asset/relative).is_file():
            errors.append(f'Missing robot resource: {asset/relative}; see docs/installation.md')
    if (asset/'control_parameters.json').is_file():
        actual = json.loads((asset/'control_parameters.json').read_text())
        for key in ['joints', 'policy_joint_names', 'arm_joint_names', 'deployment_joint_names', 'control_period_s']:
            if actual.get(key) != expected[key]:
                errors.append(f'Robot control contract differs in {key}; do not mix asset versions')
    urdf = asset/'source'/expected['urdf_relative_path']
    if urdf.is_file() and hashlib.sha256(urdf.read_bytes()).hexdigest() != expected['source_hashes'][expected['urdf_relative_path']]:
        errors.append('Robot URDF does not match the frozen control contract')
    dataset = amass_root()
    roots = [dataset/'g1/CMU', dataset/'dataset/g1/CMU']
    if not any(any(p.rglob('*_jpos.npy')) for p in roots if p.is_dir()):
        errors.append(f'No retargeted AMASS *_jpos.npy clips under {dataset}; set ELF3_AMASS_DIR')
    if deployment:
        try:
            deployment_framework()
            if not mujoco_xml().is_file():
                errors.append(f'Missing MuJoCo XML: {mujoco_xml()}')
        except (RuntimeError, FileNotFoundError) as exc:
            errors.append(str(exc))
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', action='store_true')
    args = parser.parse_args()
    errors = check(args.deployment)
    print('\n'.join(errors) if errors else 'External resource checks passed')
    raise SystemExit(bool(errors))
