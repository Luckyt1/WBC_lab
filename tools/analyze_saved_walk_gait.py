"""Audit saved Isaac contact traces without starting or changing training.

Raw touchdown rates are contact transitions, not guaranteed physical steps.
World XY/yaw, clearance and slip cannot be reconstructed from the old NPZs.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def bouts(mask, dt):
    edges = np.flatnonzero(np.diff(mask.astype(int))) + 1
    return [(b-a)*dt for a, b in zip(edges[:-1], edges[1:]) if mask[a]]


def analyze(directory):
    report = json.loads((directory/'report.json').read_text())
    dt, repeats = report['dt'], report['repeats']
    start = round(report['warmup_s']/dt)
    out = dict(checkpoint=report['checkpoint'], checkpoint_sha256=report['checkpoint_sha256'],
               iteration=report['iteration'], dt=dt, profiles={})
    for profile in report['profiles']:
        name = profile['settings']['name']
        arrays = np.load(directory/(name+'.npz'))
        rows = {}
        for i, scenario in enumerate(arrays['scenario_names']):
            group = slice(i*repeats, (i+1)*repeats)
            c = arrays['feet_contact'][start:, group]
            actual = arrays['actual'][start:, group]
            valid = arrays['valid'][start:, group]
            # Exclude failed trials from temporal gait statistics (tracking retains
            # the existing evaluator's all-valid-samples convention).
            survivors = arrays['first_failure'][group] < 0
            live = c[:, survivors]
            side_rows = []
            for side in range(2):
                durations = [duration for env in range(live.shape[1])
                             for duration in bouts(~live[:, env, side], dt)]
                cc = live[:, :, side]
                side_rows.append(dict(
                    duty=float(cc.mean()) if cc.size else None,
                    raw_touchdowns_per_s=float((cc[1:] & ~cc[:-1]).sum()/max(1, live.shape[1])/(len(live)*dt)),
                    median_air_s=float(np.median(durations)) if durations else None,
                    p95_air_s=float(np.quantile(durations, .95)) if durations else None,
                    air_bouts_at_least_80ms=int(sum(t >= .08-1e-9 for t in durations)),
                    never_lifted_envs=int(cc.all(0).sum())))
            err = actual-arrays['command'][start:, group]
            rows[str(scenario)] = dict(survivor_trials=int(survivors.sum()), trials=repeats,
                                      signed_velocity_mean=actual[valid].mean(0)[:3].tolist() if valid.any() else None,
                                      velocity_mae=np.abs(err[valid]).mean(0)[:3].tolist() if valid.any() else None,
                                      left=side_rows[0], right=side_rows[1],
                                      double_support=float(live.all(-1).mean()) if live.size else None,
                                      single_support=float((live.sum(-1) == 1).mean()) if live.size else None)
            if 'foot_clearance' in arrays:
                height = arrays['foot_clearance'][start:, group][:, survivors]
                rows[str(scenario)]['p95_sole_clearance_m'] = (np.quantile(height, .95, axis=(0, 1)).tolist()
                                                              if height.size else None)
                counts = [0, 0]
                for side in range(2):
                    for trial in range(live.shape[1]):
                        air = ~live[:, trial, side]
                        edges = np.flatnonzero(np.diff(air.astype(int)))+1
                        for lo, hi in zip(edges[:-1], edges[1:]):
                            if (air[lo] and .12-1e-6 <= (hi-lo)*dt <= .35+1e-6
                                    and height[lo:hi, trial, side].max() >= .02
                                    and live[lo:hi+1, trial, 1-side].all()):
                                counts[side] += 1
                rows[str(scenario)]['supported_swing_120_350ms_20mm_count'] = counts
            if 'pelvis_yaw' in arrays:
                yaw = np.unwrap(arrays['pelvis_yaw'][start:, group][:, survivors], axis=0)
                rows[str(scenario)]['median_heading_change_deg'] = (float(np.median(np.rad2deg(yaw[-1]-yaw[0])))
                                                                   if yaw.size else None)
        stand = arrays['actual'][:, :repeats, :3]
        windows = []
        for lo, hi in ((4, 8), (8, 12), (12, 16)):
            v = stand[round(lo/dt):round(hi/dt)]
            mask = arrays['valid'][round(lo/dt):round(hi/dt), :repeats]
            windows.append(dict(start_s=lo, end_s=hi, mean_abs_velocity=np.abs(v[mask]).mean(0).tolist() if mask.any() else None))
        out['profiles'][name] = dict(scenarios=rows, standing_time_windows=windows)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ancestors', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    paths = args.ancestors + sorted(args.evaluations.glob('model_*'), key=lambda p: int(p.name.split('_')[-1]))
    data = [analyze(p) for p in paths if (p/'report.json').is_file()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        method='Post-warmup contact metrics on surviving trials; left/right match preserve_order feet configuration. No de-bouncing; transitions can be contact chatter. Existing arrays lack global heading/position, joint traces and foot clearance.',
        runs=data), indent=2)+'\n')
    print('Wrote', len(data), 'checkpoint audits to', args.output)


if __name__ == '__main__':
    main()
