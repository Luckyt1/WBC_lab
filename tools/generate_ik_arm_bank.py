"""Procedural wrist goals and actual deployment-IK paths; no motion capture.

Run in the isolated Pinocchio environment. IK uses nominal joint feedback from
the previous limited command when caching paths. Training subsequently applies
the shared collision/rate supervisor against live simulated body geometry.
Online feedback IK is evaluated separately; this cache is not its exact replay.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from elf3_ik_core import new_solver, forward_targets, SOURCE, PROVENANCE

DEFAULT = np.array([.2, .2, 0., .6, 0., 0., 0., .2, -.2, 0., .6, 0., 0., 0.])
MIRROR = np.array([1., -1., -1., 1., -1., 1., -1.])
GROUPS = ('forward', 'side', 'overhead', 'left_only', 'right_only', 'asymmetric', 'workspace')


def pose_key(q):
    # Keep mirrored counterparts in the same split even with symmetry augmentation.
    q = np.asarray(q).reshape(2, 7)
    mirrored = q[::-1] * MIRROR
    return min(np.round(q, 4).tobytes(), np.round(mirrored, 4).tobytes()).hex()


def make_path(target):
    module, solver, limiter, _ = new_solver()
    start_p, start_q = forward_targets(solver, DEFAULT)
    goal_p, goal_q = np.array(target['positions']), np.array(target['orientations_wxyz'])
    slerps = [Slerp([0, 1], Rotation.from_quat(np.array([start_q[s], goal_q[s]])[:, [1, 2, 3, 0]])) for s in (0, 1)]
    reference = DEFAULT.copy()
    paths, errors, reasons = [], [], Counter()
    # Both legs of the cycle are solved, rather than reversing joint samples.
    for reverse in (False, True):
        path = [reference.copy()]
        for k in range(1, 151):
            alpha = min(k / 125., 1.)
            alpha = alpha*alpha*(3-2*alpha)
            if reverse:
                alpha = 1-alpha
            positions = start_p+alpha*(goal_p-start_p)
            orientations = np.array([f(alpha).as_quat()[[3, 0, 1, 2]] for f in slerps])
            solver.update_measured(module.ARM_JOINTS, reference)
            result = solver.solve(positions, orientations)
            reasons[result.reason] += 1
            if result.positions is not None:
                reference = limiter._rate_limited_arm_reference(result.positions).astype(float)
            path.append(reference.copy())
        paths.append(path)
        errors.append([result.position_error_m, result.orientation_error_rad])
    return target['id'], np.asarray(paths, dtype=np.float32), errors, dict(reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-targets', type=Path, help='Optional historical regression targets; use with --previous-results')
    parser.add_argument('--previous-results', type=Path, help='Optional historical trial results; use with --previous-targets')
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--train-targets', type=int, default=512)
    parser.add_argument('--test-targets', type=int, default=128)
    parser.add_argument('--seed', type=int, default=202610071)
    parser.add_argument('--test-seed', type=int, default=202610072)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if bool(args.previous_targets) != bool(args.previous_results):
        parser.error('--previous-targets and --previous-results must be provided together')
    if min(args.train_targets, args.test_targets, args.workers) <= 0:
        parser.error('Target counts and worker count must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    module, solver, _, params = new_solver()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    qids = [model.joint(n).qposadr[0] for n in module.ARM_JOINTS]
    arms = {i for i in range(model.nbody) if any(x in model.body(i).name for x in ('shoulder', 'elbow', 'wrist'))}
    def clear(q):
        mujoco.mj_resetData(model, data)
        data.qpos[2] = .85
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        for c in data.contact:
            a, b = (int(model.geom_bodyid[i]) for i in (c.geom1, c.geom2))
            if a and b and a != b and c.dist < -.001 and (a in arms or b in arms):
                return False
        return True
    old = json.loads(args.previous_targets.read_text())['targets'] if args.previous_targets else []
    reports = json.loads(args.previous_results.read_text()) if args.previous_results else []
    hard = {r['target_id'] for r in reports if not r['completed'] or r['maximum_displacement_m'] > .15}
    targets, keys = [], set()
    for t in old:
        q = np.array(t['known_feasible_joint_positions'][15:29])
        targets.append(dict(t, id='regression_'+t['id'], split='train', hard=t['id'] in hard,
                            origin='previous_32k_workspace_test', witness_arm_q=q.tolist(), pose_key=pose_key(q)))
        keys.add(pose_key(q))
    # Split independent goals BEFORE trajectory variants and failure weighting.
    for split, count, seed in [('train', args.train_targets, args.seed), ('test', args.test_targets, args.test_seed)]:
        rng = np.random.default_rng(seed)
        accepted, attempts = 0, 0
        while accepted < count:
            group = GROUPS[accepted % len(GROUPS)]
            attempts += 1
            if attempts > count*200:
                raise RuntimeError('Could not find enough clear FK witnesses')
            q = DEFAULT.reshape(2, 7).copy()
            if group in ('forward', 'side', 'overhead'):
                ranges = {'forward': (-2.1, -.6, .05, .65), 'side': (-.8, .4, .9, 1.9),
                          'overhead': (-2.95, -1.8, .05, 1.)}
                p0, p1, r0, r1 = ranges[group]
                left = np.array([rng.uniform(p0,p1), rng.uniform(r0,r1), rng.uniform(-.6,.6),
                                 rng.uniform(.25,1.6), *rng.uniform(-.8,.8,3)])
                q[:] = [left, left*MIRROR]
            else:
                low, high = solver._soft_lower.reshape(2,7), solver._soft_upper.reshape(2,7)
                sampled = rng.uniform(low+.025, high-.025)
                if group == 'left_only': q[0] = sampled[0]
                elif group == 'right_only': q[1] = sampled[1]
                else: q[:] = sampled
            q = q.ravel()
            key = pose_key(q)
            if key in keys or not clear(q): continue
            p, r = forward_targets(solver, q)
            targets.append(dict(id=f'{split}_{accepted:04d}_{group}', split=split, group=group, hard=False,
                                origin='procedural_fk_witness', witness_arm_q=q.tolist(), pose_key=key,
                                positions=p.tolist(), orientations_wxyz=r.tolist()))
            keys.add(key); accepted += 1
    manifest = dict(version=1, seed=args.seed, test_seed=args.test_seed, targets=targets,
                    arm_joint_names=list(module.ARM_JOINTS), default_arm_q=DEFAULT.tolist(),
                    joint_limits=np.stack((solver._soft_lower,solver._soft_upper),-1).tolist(),
                    source=json.loads(PROVENANCE.read_text()), solver_parameters=params,
                    generation=dict(dt=.02, frames_per_leg=151, nominal_leg_seconds=3., nominal_feedback='previous_rate_limited_command',
                        collision_endpoint_filter='MuJoCo robot self penetration >1mm rejected; live path guard runs during training',
                        xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest()),
                    recipe=dict(neutral_fraction=.15, hard_fraction_of_active=.30, max_velocity=6., guard_margin=.012,
                        prepare_s=[.6,1.2], reach_s=[.5,6.], hold_s=[2.,8.], return_s=[.5,4.], rest_s=[.5,1.5],
                        gravity_feedforward=True, gravity_effort_limit_fraction=.8),
                    split_policy='Independent seeded FK targets; exact and mirrored duplicates excluded. All prior observed failures are training/regression only.',
                    limitations='Finite procedural coverage; nominal-feedback IK cache with live collision projection is not online feedback IK. FK feasibility does not certify the IK branch or guarded path reaches the wrist target.',
                    counts=dict(train=sum(t['split']=='train' for t in targets),test=args.test_targets,
                                hard=sum(t['hard'] for t in targets)))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['counts']), flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(make_path, target) for target in targets]
        for future in as_completed(futures):
            ident, path, errors, reasons = future.result()
            results[ident] = (path, errors, reasons)
            if len(results) % 20 == 0:
                print(json.dumps(dict(completed=len(results), total=len(targets))), flush=True)
    for split in ('train', 'test'):
        selected = [t for t in targets if t['split']==split]
        paths = np.stack([results[t['id']][0] for t in selected])
        if not np.isfinite(paths).all(): raise ValueError('Nonfinite IK path')
        np.savez_compressed(args.output/(split+'.npz'), paths=paths,
            ids=np.array([t['id'] for t in selected]), hard=np.array([t['hard'] for t in selected]),
            groups=np.array([t['group'] for t in selected]))
        for t in selected:
            t['nominal_endpoint_error'] = results[t['id']][1]
            t['ik_reasons'] = results[t['id']][2]
    manifest['bank_sha256'] = {s:hashlib.sha256((args.output/(s+'.npz')).read_bytes()).hexdigest() for s in ('train','test')}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete', **manifest['counts'])), flush=True)


if __name__ == '__main__':
    main()
