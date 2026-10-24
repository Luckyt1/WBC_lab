"""Explicit torso-relative yaw curriculum, checkpoint metadata and frozen tests."""
from copy import deepcopy
import math
from pathlib import Path

import yaml


def validate_waist_yaw_range(bounds):
    if len(bounds) != 2:
        raise ValueError('Expected two waist yaw bounds in radians')
    low, high = map(float, bounds)
    if not all(map(math.isfinite, (low, high))) or not -.6 <= low < 0 < high <= .6:
        raise ValueError('Waist yaw must cover both signs within [-0.6, 0.6] radians')
    return low, high


def waist_yaw_fields(bounds, start_iteration=5000, ramp_updates=15000):
    low, high = validate_waist_yaw_range(bounds)
    if start_iteration < 0 or ramp_updates <= 0:
        raise ValueError('Invalid waist yaw curriculum timing')
    return dict(waist_yaw_initial_range=(max(low, -.05), min(high, .05)),
                waist_yaw_range=(low, high), sampling_waist_yaw_range=None,
                waist_yaw_sampling_all_modes=True, rel_zero_waist_yaw=.25,
                waist_yaw_start_iteration=int(start_iteration),
                waist_yaw_end_iteration=int(start_iteration+ramp_updates))


def configure_waist_yaw(cfg, bounds, start_iteration=5000, ramp_updates=15000):
    fields = waist_yaw_fields(bounds, start_iteration, ramp_updates)
    for name, value in fields.items():
        setattr(cfg.commands, name, value)
    return dict(recipe='waist_yaw_v1', commands=fields,
                yaw_frame='torso heading relative to pelvis; not world heading or joint angle',
                rate_rad_s=cfg.commands.waist_yaw_rate, rewards_unchanged=True,
                neutral_probability=.25, all_velocity_and_height_modes=True)


def expected_waist_yaw_recipe(before, bounds, start_iteration=5000, ramp_updates=15000):
    expected = deepcopy(before)
    fields = waist_yaw_fields(bounds, start_iteration, ramp_updates)
    # Match the scalar representation used by the strict BaseLoader audit.
    expected['commands'].update(yaml.load(yaml.safe_dump(fields), Loader=yaml.BaseLoader))
    return expected


def checkpoint_waist_yaw(checkpoint):
    cfg = yaml.load((Path(checkpoint).parent/'params/env.yaml').read_text(), Loader=yaml.BaseLoader)['commands']
    if cfg.get('waist_yaw_sampling_all_modes', 'false') != 'true':
        return None
    bounds = validate_waist_yaw_range(cfg['waist_yaw_range'])
    start, end = int(cfg['waist_yaw_start_iteration']), int(cfg['waist_yaw_end_iteration'])
    waist_yaw_fields(bounds, start, end-start)
    return dict(bounds=bounds, start_iteration=start, ramp_updates=end-start,
                rate_rad_s=float(cfg['waist_yaw_rate']))


def waist_yaw_scenarios(height_bounds, yaw_bounds):
    """Both signs at low/mid/high heights, stationary and moving, at full range."""
    low, high = map(float, height_bounds)
    yaw_low, yaw_high = validate_waist_yaw_range(yaw_bounds)
    rows = []
    for height in sorted(set((low, (low+high)/2, high))):
        for label, velocity in (('stand', (0., 0., 0.)), ('forward', (.3, 0., 0.)),
                                ('lateral', (0., .2, 0.)), ('turn', (0., 0., .5))):
            for sign, yaw in (('neg', yaw_low), ('pos', yaw_high)):
                rows.append((f'waist_yaw_{height:.3f}_{label}_{sign}', [*velocity, height, yaw]))
    return rows


def evaluate_waist_yaw(report):
    """Require yaw/height tracking as well as stationary-base or moving control."""
    if __package__:
        from .walk_robust_gates import evaluate_gates
    else:
        from walk_robust_gates import evaluate_gates
    bounds = report.get('waist_yaw_command_range')
    if bounds is None:
        return None
    rows = waist_yaw_scenarios(report['height_command_range'], bounds)
    tasks = [name for name, _ in rows]
    gates = evaluate_gates(report, tasks)
    for profile_name, profile in gates['profiles'].items():
        scenarios = {r['scenario']: r for p in report['profiles'] if p['settings']['name'] == profile_name
                     for r in p['scenarios']}
        for name, _ in rows:
            check = profile.get('checks', {}).get(name)
            if check is None:
                continue
            mae = check['mae']
            pose_ok = len(mae) == 5 and all(math.isfinite(x) and x >= 0 for x in mae) and mae[3] <= .03 and mae[4] <= .10
            # A waist turn while standing must not be achieved by moving the base.
            quiet_ok = ('_stand_' not in name or
                        (len(mae) == 5 and all(x <= t for x, t in zip(mae[:3], (.02, .02, .10)))
                         and 0 <= scenarios[name].get('single_support_fraction', math.inf) <= .10))
            check.update(pose_tracking_passed=pose_ok, quiet_standing_passed=quiet_ok)
            check['passed'] = check['passed'] and pose_ok and quiet_ok
            if not check['passed'] and name not in profile['failures']:
                profile['failures'].append(name)
        profile['passed'] = not profile['failures']
    gates['passed'] = all(p['passed'] for p in gates['profiles'].values())
    gates['pose_mae_limits'] = dict(height_m=.03, waist_yaw_rad=.10)
    return gates
