"""Check external inputs without initializing Isaac Sim or touching the GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legged_lab.paths import robot_asset_dir, amass_root, deployment_framework, mujoco_xml, ik_source_dir, ik_bank_dir


def check(deployment=False, bank=None, amass=False):
    errors = []
    asset = robot_asset_dir()
    expected = json.loads((ROOT/'legged_lab/assets/elf3_bxi_31/control_parameters.json').read_text())
    for relative in ['usd/elf3.usd', 'control_parameters.json', 'source/' + expected['urdf_relative_path'],
                     'motion_visualization/stand_back.txt', 'motion_visualization/walk_left.txt',
                     'motion_visualization/walk_right.txt']:
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
    if amass and not any(any(p.rglob('*_jpos.npy')) for p in roots if p.is_dir()):
        errors.append(f'No retargeted AMASS *_jpos.npy clips under {dataset}; set ELF3_AMASS_DIR')
    provenance = json.loads((ROOT/'legged_lab/assets/elf3_ik_upstream/PROVENANCE.json').read_text())
    for name, digest in provenance['sha256'].items():
        path = ik_source_dir()/name
        if not path.is_file():
            errors.append(f'Missing IK source: {path}; run scripts/prepare_ik_source.py --checkout PATH')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f'IK source checksum mismatch: {path}')
    bank = Path(bank) if bank else ik_bank_dir()
    if not (bank/'manifest.json').is_file():
        errors.append(f'Missing IK bank: {bank}; see docs/ik-data.md')
    else:
        manifest = json.loads((bank/'manifest.json').read_text())
        if manifest.get('version') != 3 or manifest.get('recipe', {}).get('executor') != 'predictive_ruckig_v3':
            errors.append('Default recipe requires a predictive_ruckig_v3 bank')
        if manifest.get('fixture_notice'):
            errors.append('Test fixtures cannot be used for production training')
        for split in ('train', 'test'):
            path = bank/(split+'.npz')
            if not path.is_file():
                errors.append(f'Missing IK trajectories: {path}')
            elif hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get('bank_sha256', {}).get(split):
                errors.append(f'IK bank checksum mismatch: {split}')
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
    parser.add_argument('--amass', action='store_true', help='Required for the supervisor full retention evaluation')
    parser.add_argument('--bank', type=Path)
    args = parser.parse_args()
    errors = check(args.deployment, bank=args.bank, amass=args.amass)
    print('\n'.join(errors) if errors else 'External resource checks passed')
    raise SystemExit(bool(errors))
