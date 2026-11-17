"""Add lateral reaches to a frozen continuous GL165 bank without replacing its goals."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from legged_lab.assets.gl165.model import asset_dir
from tools.check_gl165_resources import check_resources
from tools.extend_gl165_ik_bank import PathScreen, transition_duration
from tools.gl165_kinematics import ArmChain
from tools.gl165_sources import sha256

PATTERNS = ('left', 'right', 'both', 'left_side_right_front', 'right_side_left_front')
HEIGHTS = {'side_low': (-.22, -.08), 'side_level': (-.06, .08), 'side_high': (.13, .25)}
REGIONS = ('low', 'overhead', 'lateral')


def extend_lateral(asset, source, output, train_count=120, test_count=30, seed=20261115):
    asset, source, output = Path(asset).resolve(), Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if min(train_count, test_count) < 30 or train_count % 15 or test_count % 15:
        raise ValueError('Counts must be multiples of 15, with at least 30 per split')
    check_resources(asset, source)
    original = json.loads((source / 'manifest.json').read_text())
    if (original['recipe'].get('motion_mode') != 'continuous_targets_v1'
            or {t['region'] for t in original['targets']} != {'low', 'overhead'}):
        raise ValueError('Expected a continuous bank containing low and overhead targets')
    manifest = copy.deepcopy(original)
    manifest['recipe'].update(name='gl165_continuous_lateral_v3',
                              region_probabilities={'low': .5, 'lateral': .3, 'overhead': .2})
    manifest.update(source_bank=str(source), source_manifest_sha256=sha256(source / 'manifest.json'),
                    extension_seed=seed)
    screen = PathScreen(asset)
    chains = [ArmChain(asset / 'source/urdf/gl165.urdf', side) for side in ('l', 'r')]
    manifest['lateral_workspace'] = dict(shoulder_origins_m=[c.origins[0].tolist() for c in chains],
        outward_offset_m=[.35, .53], forward_offset_m=[-.12, .16], height_bands_m=HEIGHTS)
    output.mkdir(parents=True)
    report = dict(source=str(source), region_probabilities=manifest['recipe']['region_probabilities'], splits={})
    source_targets = {t['id']: t for t in original['targets']}
    reused_total = 0
    for split, count, sequence in (('train', train_count, 0), ('test', test_count, 1)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, sequence]))
        with np.load(source / (split + '.npz'), allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        goals = list(arrays['paths'][:, 0, -1].astype(np.float64))
        ids, groups = list(arrays['ids']), list(arrays['groups'])
        regions = [source_targets[i]['region'] for i in ids]
        old_count = len(goals)
        new_paths, new_durations, dynamics = [], [], []
        time = np.linspace(0., 1., arrays['paths'].shape[2])
        blend = time**3 * (10 - 15 * time + 6 * time**2)
        for i in range(count):
            pattern = PATTERNS[i % 5]
            height = list(HEIGHTS)[(i // 5) % 3]
            lateral = [pattern in ('left', 'both', 'left_side_right_front'),
                       pattern in ('right', 'both', 'right_side_left_front')]
            for _ in range(500):
                q = screen.home.copy().reshape(2, 7)
                for arm, chain in enumerate(chains):
                    if lateral[arm]:
                        q[arm] = chain.sample_lateral(rng, HEIGHTS[height])[0]
                    elif pattern in ('left_side_right_front', 'right_side_left_front'):
                        q[arm] = chain.sample(rng)[0]
                q = q.ravel()
                if min(np.linalg.norm(q - previous) for previous in goals) < .08:
                    continue
                safe, _ = screen.geometric(screen.home, q)
                if not safe:
                    continue
                poses = [c.fk(angles) for c, angles in zip(chains, q.reshape(2, 7))]
                positions = np.array([p for p, _, _ in poses])
                duration = transition_duration(screen.home, q, False)
                safe, metrics = screen.dynamics(q, duration, positions)
                if safe:
                    break
            else:
                raise RuntimeError(f'Could not validate lateral target {split}/{pattern}/{height}')
            identity = f'gl165_{split}_lateral_{i:05d}'
            group = pattern if pattern in ('left', 'right') else 'both'
            forward = screen.home + blend[:, None] * (q - screen.home)
            new_paths.append(np.stack((forward, forward[::-1])))
            new_durations.append([duration, duration])
            goals.append(q); ids.append(identity); groups.append(group); regions.append('lateral')
            manifest['targets'].append(dict(id=identity, split=split, region='lateral', group=group,
                pattern=pattern, height_band=height, lateral_arms=lateral, positions=positions.tolist(),
                orientations_wxyz=[Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]].tolist() for _, r, _ in poses]))
            dynamics.append(dict(id=identity, pattern=pattern, height_band=height, **metrics))
            if i % 10 == 0 or i == count - 1:
                print('GL165_LATERAL_TARGETS', split, i + 1, '/', count, flush=True)
        goals, groups, regions = np.array(goals), np.array(groups), np.array(regions)
        overhead = regions == 'overhead'
        neighbors, durations, penetrations = [], [], []
        reused = 0
        for i, start in enumerate(goals):
            row, timing, depths = [], [], []
            for region in REGIONS:
                for group in ('both', 'left', 'right'):
                    # Old goals and joint paths are unchanged. Reuse their
                    # already screened low/overhead edges and screen new edges.
                    if i < old_count and region != 'lateral':
                        indices = arrays['neighbors'][i]
                        columns = np.flatnonzero((regions[indices] == region) & (groups[indices] == group))[:2]
                        if len(columns) != 2:
                            raise ValueError('Source graph has fewer than two edges per region/group')
                        row.extend(indices[columns].tolist())
                        timing.extend(arrays['edge_min_durations'][i, columns].tolist())
                        depths.extend(arrays['edge_max_penetration_m'][i, columns].tolist())
                        reused += 2
                        continue
                    candidates = np.flatnonzero((regions == region) & (groups == group)
                                               & (np.arange(len(goals)) != i))
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
                        raise RuntimeError(f'Insufficient safe edges {split}/{ids[i]} -> {region}/{group}')
            neighbors.append(row); durations.append(timing); penetrations.append(depths)
            if i % 40 == 0 or i == len(goals) - 1:
                print('GL165_LATERAL_GRAPH', split, i + 1, '/', len(goals), flush=True)
        arrays.update(paths=np.concatenate((arrays['paths'], np.array(new_paths, dtype=np.float32))),
            min_durations=np.concatenate((arrays['min_durations'], np.array(new_durations, dtype=np.float32))),
            ids=np.array(ids), groups=groups, regions=regions, overhead=overhead,
            hard=np.zeros(len(goals), dtype=bool), neighbors=np.array(neighbors, dtype=np.int64),
            edge_min_durations=np.array(durations, dtype=np.float32),
            edge_max_penetration_m=np.array(penetrations, dtype=np.float32))
        destination = output / (split + '.npz')
        np.savez_compressed(destination, **arrays)
        manifest['bank_sha256'][split] = sha256(destination)
        reused_total += reused
        report['splits'][split] = dict(original_targets=old_count, added_lateral_targets=count,
            total_targets=len(goals), directed_transitions=len(goals) * 18, reused_transitions=reused,
            lateral_patterns=dict(Counter(d['pattern'] for d in dynamics)),
            lateral_height_bands=dict(Counter(d['height_band'] for d in dynamics)), dynamics=dynamics,
            min_transition_s=float(np.min(durations)), max_min_transition_s=float(np.max(durations)))
    report.update(reused_transitions=reused_total, accepted_new_geometric_checks=screen.checked_edges,
                  rejected_new_geometric_checks=screen.rejected_edges)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('GL165_LATERAL_BANK_COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=asset_dir())
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=120)
    parser.add_argument('--test-count', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261115)
    args = parser.parse_args()
    extend_lateral(args.asset, args.source, args.output, args.train_count, args.test_count, args.seed)
