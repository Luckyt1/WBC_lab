"""CPU validation of GL-165 source identity, IK limits and trajectory endpoints."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from legged_lab.assets.gl165.model import (
    ARM_JOINT_NAMES, ASSEMBLY_COLLISION_FILTERS, CONTROL_PARAMETERS, CONTROL_SOURCE,
    DEFAULT_POSITIONS, asset_dir, ik_bank_dir,
)
from tools.gl165_kinematics import ArmChain
from tools.gl165_sources import sha256


def check_resources(asset, bank):
    asset, bank = Path(asset), Path(bank)
    model = json.loads((asset / 'model.json').read_text())
    if model.get('control_parameters_source') != CONTROL_SOURCE:
        raise ValueError('GL-165 asset controller metadata differs from the ELF3 parameter source; synchronize it first')
    for name, expected in CONTROL_PARAMETERS.items():
        for field, value in expected.items():
            if model['joints'].get(name, {}).get(field) != value:
                raise ValueError(f'GL-165 asset controller mismatch: {name}.{field}')
    manifest = json.loads((bank / 'manifest.json').read_text())
    urdf = asset / 'source/urdf/gl165.urdf'
    if model['source_urdf_sha256'] != sha256(asset / 'source/original.urdf'):
        raise ValueError('Original URDF snapshot changed')
    if model['training_urdf_sha256'] != sha256(urdf) or manifest['training_urdf_sha256'] != sha256(urdf):
        raise ValueError('IK bank / prepared URDF mismatch')
    if model['usd_sha256'] != sha256(asset / 'usd/gl165.usd'):
        raise ValueError('Converted USD changed')
    if model['assembly_collision_filters'] != [list(pair) for pair in ASSEMBLY_COLLISION_FILTERS]:
        raise ValueError('Missing assembly collision filters')
    for name, digest in model['source_mesh_sha256'].items():
        if digest != sha256(asset / 'source/meshes' / name):
            raise ValueError(f'Mesh snapshot changed: {name}')
    if manifest['robot'] != 'gl165' or manifest['arm_joint_names'] != ARM_JOINT_NAMES:
        raise ValueError('Wrong robot / IK arm joint order')
    chains = [ArmChain(urdf, side) for side in ('l', 'r')]
    limits = np.concatenate([chain.limits for chain in chains])
    home = np.array([DEFAULT_POSITIONS[n] for n in ARM_JOINT_NAMES])
    targets = {target['id']: target for target in manifest['targets']}
    if len(targets) != len(manifest['targets']):
        raise ValueError('Duplicate IK target IDs')
    all_ids = set()
    report = dict(robot='gl165', total_mass_kg=model['total_mass_kg'], joints=model['joint_count'],
                  control_parameters_source=CONTROL_SOURCE)
    for split in ('train', 'test'):
        path = bank / f'{split}.npz'
        if sha256(path) != manifest['bank_sha256'][split]:
            raise ValueError(f'IK bank checksum mismatch: {split}')
        with np.load(path, allow_pickle=False) as data:
            paths = data['paths']
            if paths.ndim != 4 or paths.shape[1] != 2 or paths.shape[-1] != 14 or not len(paths):
                raise ValueError(f'Invalid IK path shape: {paths.shape}')
            if not np.isfinite(paths).all() or (paths < limits[:, 0] - 1e-6).any() or (paths > limits[:, 1] + 1e-6).any():
                raise ValueError(f'IK paths contain invalid joint targets: {split}')
            if not (np.allclose(paths[:, 0, 0], home) and np.allclose(paths[:, 1, -1], home)
                    and np.allclose(paths[:, 0, -1], paths[:, 1, 0])):
                raise ValueError(f'IK reach/return discontinuity: {split}')
            dt = data['min_durations'] / (paths.shape[2] - 1)
            velocity = np.diff(paths, axis=2) / dt[:, :, None, None]
            acceleration = np.diff(velocity, axis=2) / dt[:, :, None, None]
            speed, accel = float(np.abs(velocity).max()), float(np.abs(acceleration).max())
            if speed > 2.01 or accel > 8.1:
                raise ValueError(f'IK trajectory exceeds reference limits: {split}')
            max_error = 0.
            for identity, q in zip(data['ids'], paths[:, 0, -1]):
                if identity in all_ids or targets[identity]['split'] != split:
                    raise ValueError('Train/test target leakage')
                all_ids.add(identity)
                target = targets[identity]
                for i, chain in enumerate(chains):
                    position, rotation, _ = chain.fk(q.reshape(2, 7)[i])
                    error = float(np.linalg.norm(position - target['positions'][i]))
                    expected = Rotation.from_quat(np.array(target['orientations_wxyz'][i])[[1, 2, 3, 0]])
                    angle = np.linalg.norm((expected.inv() * Rotation.from_matrix(rotation)).as_rotvec())
                    if error > 1e-5 or angle > 1e-5:
                        raise ValueError(f'FK endpoint mismatch: {identity}')
                    max_error = max(max_error, error)
            report[split] = dict(count=len(paths), max_fk_position_error_m=max_error,
                                 max_velocity_rad_s=speed, max_acceleration_rad_s2=accel)
            if manifest['recipe'].get('motion_mode') == 'continuous_targets_v1':
                neighbors, durations = data['neighbors'], data['edge_min_durations']
                overhead, goals = data['overhead'], paths[:, 0, -1]
                regions = np.array([targets[i]['region'] for i in data['ids']])
                high_probability = manifest['recipe']['overhead_fraction_of_active']
                probabilities = manifest['recipe'].get('region_probabilities',
                    {'low': 1 - high_probability, 'overhead': high_probability})
                if (set(probabilities) != set(regions) or abs(sum(probabilities.values()) - 1) > 1e-6
                        or any(not np.isfinite(p) or p <= 0 for p in probabilities.values())
                        or ('regions' in data and not np.array_equal(regions, data['regions']))):
                    raise ValueError(f'Invalid continuous target regions: {split}')
                if (neighbors.ndim != 2 or neighbors.shape[0] != len(paths)
                        or durations.shape != neighbors.shape or not np.isfinite(durations).all()
                        or (durations < 2.).any() or neighbors.min() < 0 or neighbors.max() >= len(paths)
                        or (neighbors == np.arange(len(paths))[:, None]).any()):
                    raise ValueError(f'Invalid continuous transition graph: {split}')
                for region in probabilities:
                    for group in ('both', 'left', 'right'):
                        mask = (regions == region) & (data['groups'] == group)
                        if not mask[neighbors].any(axis=1).all():
                            raise ValueError(f'Missing outgoing group/region transitions: {split}')
                delta = np.abs(goals[neighbors] - goals[:, None]).max(axis=-1)
                high_edge = overhead[:, None] | overhead[neighbors]
                peak_speed = 1.875 * delta / durations
                peak_acceleration = (10 / np.sqrt(3)) * delta / durations**2
                if ((peak_speed > np.where(high_edge, .8, 1.2) + 1e-5).any()
                        or (peak_acceleration > np.where(high_edge, 2.5, 3.) + 1e-5).any()
                        or not np.isfinite(data['edge_max_penetration_m']).all()
                        or (data['edge_max_penetration_m'] > .0005).any()):
                    raise ValueError(f'Continuous transition limits exceeded: {split}')
                for identity, high in zip(data['ids'], overhead):
                    target = targets[identity]
                    if high != (target['region'] == 'overhead'):
                        raise ValueError(f'Incorrect overhead label: {identity}')
                    if high:
                        heights = np.array(target['positions'])[:, 2][target['overhead_arms']]
                        if not len(heights) or (heights < manifest['head_top_above_torso_m'] + .06 - 1e-5).any():
                            raise ValueError(f'Overhead target is below the declared head clearance: {identity}')
                    if target['region'] == 'lateral':
                        workspace = manifest['lateral_workspace']
                        if not any(target['lateral_arms']):
                            raise ValueError(f'Lateral target has no lateral arm: {identity}')
                        for arm, active in enumerate(target['lateral_arms']):
                            if not active:
                                continue
                            delta = np.array(target['positions'][arm]) - chains[arm].origins[0]
                            delta[1] *= chains[arm].sign
                            bounds = np.array([workspace['forward_offset_m'], workspace['outward_offset_m'],
                                               workspace['height_bands_m'][target['height_band']]])
                            if (delta < bounds[:, 0] - 1e-5).any() or (delta > bounds[:, 1] + 1e-5).any():
                                raise ValueError(f'Lateral target outside declared shoulder-relative sector: {identity}')
                report[split].update(overhead_count=int(overhead.sum()), transitions=int(neighbors.size),
                    region_counts={region: int((regions == region).sum()) for region in probabilities},
                    max_transition_velocity_rad_s=float(peak_speed.max()),
                    max_transition_acceleration_rad_s2=float(peak_acceleration.max()))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=asset_dir())
    parser.add_argument('--bank', type=Path, default=ik_bank_dir())
    args = parser.parse_args()
    print(json.dumps(check_resources(args.asset, args.bank), indent=2))
