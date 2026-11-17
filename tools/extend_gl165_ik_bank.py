"""Extend a frozen GL165 bank with overhead goals and continuous safe transitions.

Run in the MuJoCo preview Python. The source bank is never overwritten. Both
splits retain their existing goals and receive independently sampled new poses.
"""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from legged_lab.assets.gl165.model import ARM_JOINT_NAMES, DEFAULT_POSITIONS, asset_dir
from tools.gl165_kinematics import ArmChain
from tools.gl165_mujoco import build_model
from tools.gl165_sources import sha256

PATTERNS = ('left', 'right', 'both', 'left_high_right_low', 'right_high_left_low')


def transition_duration(start, goal, overhead):
    delta = float(np.abs(goal - start).max())
    speed, acceleration = (.8, 2.5) if overhead else (1.2, 3.)
    return max(2., 1.875 * delta / speed, np.sqrt(5.774 * delta / acceleration))


class PathScreen:
    def __init__(self, asset):
        self.model = mujoco.MjModel.from_xml_path(str(build_model(asset)))
        self.data = mujoco.MjData(self.model)
        self.qids = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in ARM_JOINT_NAMES])
        self.home = np.array([DEFAULT_POSITIONS[n] for n in ARM_JOINT_NAMES])
        self.torso = self.model.body('body_y_link').id
        self.wrists = [self.model.body(n).id for n in ('lwrist_y_link', 'rwrist_y_link')]
        self.data.qpos[self.qids] = self.home
        mujoco.mj_forward(self.model, self.data)
        top = -np.inf
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != self.model.body('head_y_link').id:
                continue
            mesh = self.model.geom_dataid[g]
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            start, size = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            vertices = self.model.mesh_vert[start:start + size]
            vertices = vertices @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g]
            top = max(top, float(vertices[:, 2].max()))
        self.head_top = top - self.data.xpos[self.torso, 2]
        self.checked_edges = 0
        self.rejected_edges = 0

    def collision_depth(self):
        return max((max(0., -c.dist) for c in self.data.contact), default=0.)

    def geometric(self, start, goal):
        """Sample every <=0.02 rad of the largest-changing joint, at least 65 poses."""
        count = max(65, int(np.ceil(np.abs(goal - start).max() / .02)) + 1)
        depth = 0.
        for fraction in np.linspace(0., 1., count):
            self.data.qpos[self.qids] = start + fraction * (goal - start)
            mujoco.mj_forward(self.model, self.data)
            depth = max(depth, self.collision_depth())
            if depth > .0005:
                self.rejected_edges += 1
                return False, depth
        self.checked_edges += 1
        return True, depth

    def dynamics(self, goal, duration, positions):
        """Check an entire reset-to-goal trajectory and its settled tracking."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = self.home
        self.data.ctrl[:] = self.home
        mujoco.mj_forward(self.model, self.data)
        peak_torque, penetration = 0., 0.
        # Same 50 Hz target updates and 200 Hz PD integration as training.
        for step in range(int(np.ceil((duration + .8) / .02))):
            fraction = min(1., (step + 1) * .02 / duration)
            blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
            self.data.ctrl[:] = self.home + blend * (goal - self.home)
            for _ in range(4):
                mujoco.mj_step(self.model, self.data)
                peak_torque = max(peak_torque, float(np.max(
                    np.abs(self.data.actuator_force) / self.model.actuator_forcerange[:, 1])))
                penetration = max(penetration, self.collision_depth())
        actual = (self.data.xpos[self.wrists] - self.data.xpos[self.torso]) @ self.data.xmat[self.torso].reshape(3, 3)
        errors = np.linalg.norm(actual - positions, axis=1)
        report = dict(wrist_error_m=errors.tolist(), peak_torque_fraction=peak_torque,
                      max_collision_penetration_m=penetration,
                      actual_wrist_heights_m=actual[:, 2].tolist())
        passed = bool(np.isfinite(self.data.qpos).all() and errors.max() < .06
                      and peak_torque < .95 and penetration < .001)
        return passed, report


def extend_bank(asset, source, output, train_count=80, test_count=20, seed=20261114, overhead_fraction=.2):
    asset, source, output = Path(asset).resolve(), Path(source).resolve(), Path(output).resolve()
    if min(train_count, test_count) < 15 or train_count % 5 or test_count % 5:
        raise ValueError('New target counts must be multiples of five and at least 15 per split')
    if not 0 < overhead_fraction < 1:
        raise ValueError('Overhead sampling fraction must be between zero and one')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    original = json.loads((source / 'manifest.json').read_text())
    if original['robot'] != 'gl165' or original['training_urdf_sha256'] != sha256(asset / 'source/urdf/gl165.urdf'):
        raise ValueError('Source bank and GL165 asset do not match')
    if original['recipe'].get('motion_mode'):
        raise ValueError('Use the original reach/return bank as the extension source')
    screen = PathScreen(asset)
    chains = [ArmChain(asset / 'source/urdf/gl165.urdf', side) for side in ('l', 'r')]
    manifest = copy.deepcopy(original)
    manifest['recipe'].update(name='gl165_continuous_overhead_v2', motion_mode='continuous_targets_v1',
                              neutral_fraction=0., overhead_fraction_of_active=overhead_fraction,
                              duration_multiplier=[1., 1.25], transition_velocity_rad_s=1.2,
                              overhead_transition_velocity_rad_s=.8, transition_acceleration_rad_s2=3.,
                              overhead_transition_acceleration_rad_s2=2.5)
    for key in ('prepare_s', 'reach_s', 'hold_s', 'return_s', 'rest_s'):
        manifest['recipe'].pop(key, None)
    manifest.update(source_bank=str(source), source_manifest_sha256=sha256(source / 'manifest.json'),
                    extension_seed=seed, head_top_above_torso_m=screen.head_top,
                    note='Continuous goal-to-goal quintic transitions. No prepare/hold/return/rest cycle. '
                         'Reset starts from actual reset joints. Stored home paths are initialization/validation templates. '
                         'Transitions are screened against fixed neutral torso/legs; this is not a full-body collision guarantee.')
    for target in manifest['targets']:
        target['region'] = 'low'
    output.mkdir(parents=True)
    report = dict(source=str(source), head_top_above_torso_m=screen.head_top, splits={})
    for split, count, sequence in (('train', train_count, 0), ('test', test_count, 1)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, sequence]))
        path = source / (split + '.npz')
        if sha256(path) != original['bank_sha256'][split]:
            raise ValueError('Source bank checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        goals = list(arrays['paths'][:, 0, -1].astype(np.float64))
        ids, groups = list(arrays['ids']), list(arrays['groups'])
        old_count = len(ids)
        overhead = [False] * old_count
        new_paths, new_durations, dynamics = [], [], []
        times = np.linspace(0., 1., arrays['paths'].shape[2])
        blend = times**3 * (10 - 15 * times + 6 * times**2)
        for i in range(count):
            pattern = PATTERNS[i % len(PATTERNS)]
            for attempt in range(500):
                high = [pattern in ('left', 'both', 'left_high_right_low'),
                        pattern in ('right', 'both', 'right_high_left_low')]
                q = screen.home.copy().reshape(2, 7)
                for arm, chain in enumerate(chains):
                    if high[arm]:
                        q[arm] = chain.sample_overhead(rng, screen.head_top + .06, screen.head_top + .24)[0]
                    elif pattern in ('left_high_right_low', 'right_high_left_low'):
                        q[arm] = chain.sample(rng)[0]
                q = q.ravel()
                if min(np.linalg.norm(q - previous) for previous in goals) < .08:
                    continue
                safe, _ = screen.geometric(screen.home, q)
                if not safe:
                    continue
                poses = [chain.fk(angles) for chain, angles in zip(chains, q.reshape(2, 7))]
                positions = np.array([p for p, _, _ in poses])
                duration = transition_duration(screen.home, q, True)
                safe, metrics = screen.dynamics(q, duration, positions)
                if not safe or np.any(np.array(metrics['actual_wrist_heights_m'])[high] < screen.head_top + .02):
                    continue
                break
            else:
                raise RuntimeError(f'Could not validate {split}/{pattern} target {i}')
            identity = f'gl165_{split}_overhead_{i:05d}'
            group = pattern if pattern in ('left', 'right') else 'both'
            forward = screen.home + blend[:, None] * (q - screen.home)
            new_paths.append(np.stack((forward, forward[::-1])))
            new_durations.append([duration, duration])
            goals.append(q); ids.append(identity); groups.append(group); overhead.append(True)
            dynamics.append(dict(id=identity, pattern=pattern, **metrics))
            manifest['targets'].append(dict(id=identity, split=split, group=group, region='overhead',
                pattern=pattern, overhead_arms=high, positions=positions.tolist(),
                orientations_wxyz=[Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]].tolist() for _, r, _ in poses]))
            if i % 10 == 0 or i == count - 1:
                print('GL165_OVERHEAD_TARGETS', split, i + 1, '/', count, flush=True)
        goals, overhead, groups = np.array(goals), np.array(overhead), np.array(groups)
        # Every source has two validated edges into every region/arm-group
        # combination. This permits unbiased region/group sampling at runtime.
        neighbors, durations, penetrations = [], [], []
        for i, start in enumerate(goals):
            row, timing, depths = [], [], []
            for region in (False, True):
                for group in ('both', 'left', 'right'):
                    candidates = np.flatnonzero((overhead == region) & (groups == group) & (np.arange(len(goals)) != i))
                    accepted = 0
                    for j in rng.permutation(candidates):
                        safe, depth = screen.geometric(start, goals[j])
                        if not safe:
                            continue
                        row.append(int(j)); depths.append(depth)
                        timing.append(transition_duration(start, goals[j], bool(overhead[i] or overhead[j])))
                        accepted += 1
                        if accepted == 2:
                            break
                    if accepted != 2:
                        raise RuntimeError(f'Insufficient safe outgoing edges: {split}/{ids[i]} -> {region}/{group}')
            neighbors.append(row); durations.append(timing); penetrations.append(depths)
            if i % 40 == 0 or i == len(goals) - 1:
                print('GL165_CONTINUOUS_GRAPH', split, i + 1, '/', len(goals), flush=True)
        arrays.update(paths=np.concatenate((arrays['paths'], np.array(new_paths, dtype=np.float32))),
                      ids=np.array(ids), groups=groups, overhead=overhead,
                      hard=np.zeros(len(goals), dtype=bool), neighbors=np.array(neighbors, dtype=np.int64),
                      edge_min_durations=np.array(durations, dtype=np.float32),
                      edge_max_penetration_m=np.array(penetrations, dtype=np.float32),
                      min_durations=np.array([[transition_duration(screen.home, goal, bool(high))] * 2
                                              for goal, high in zip(goals, overhead)], dtype=np.float32))
        destination = output / (split + '.npz')
        np.savez_compressed(destination, **arrays)
        manifest['bank_sha256'][split] = sha256(destination)
        report['splits'][split] = dict(original_targets=old_count, added_overhead_targets=count,
            total_targets=len(goals), directed_transitions=len(goals) * 12,
            overhead_patterns=dict(Counter(d['pattern'] for d in dynamics)), dynamics=dynamics,
            min_transition_s=float(np.min(durations)), max_min_transition_s=float(np.max(durations)))
    report.update(accepted_geometric_checks=screen.checked_edges, rejected_geometric_checks=screen.rejected_edges)
    manifest['transition_validation'] = dict(collision_model='mujoco_convex_hull_fixed_neutral_torso_legs',
        max_joint_sampling_step_rad=.02, minimum_samples=65, penetration_tolerance_m=.0005,
        new_target_tracking_tolerance_m=.06, new_target_torque_fraction_limit=.95)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('GL165_CONTINUOUS_BANK_COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=asset_dir())
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=80)
    parser.add_argument('--test-count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--overhead-fraction', type=float, default=.2)
    args = parser.parse_args()
    extend_bank(args.asset, args.source, args.output, args.train_count, args.test_count, args.seed, args.overhead_fraction)
