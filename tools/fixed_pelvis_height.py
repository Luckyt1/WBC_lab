"""Keep fixed-height training, continuation audits and evaluation in agreement."""
from copy import deepcopy
import math
from pathlib import Path

import yaml

HEIGHT_RANGES = ('height_initial_range', 'height_range', 'sampling_height_range')


def validate_height(height):
    height = float(height)
    if not math.isfinite(height) or not .4 <= height <= .8235:
        raise ValueError('ELF3 pelvis height must be between 0.4 and 0.8235 m')
    return height


def validate_height_range(bounds):
    if len(bounds) != 2:
        raise ValueError('Expected minimum and maximum pelvis height')
    low, high = map(validate_height, bounds)
    if low >= high:
        raise ValueError('Height range must be increasing')
    return low, high


def height_range_fields(bounds, start_iteration, ramp_updates=10000):
    low, high = validate_height_range(bounds)
    if start_iteration < 0 or ramp_updates <= 0:
        raise ValueError('Invalid height curriculum timing')
    return dict(nominal_height=high, height_initial_range=(max(low, high-.05), high),
                height_range=(low, high), sampling_height_range=None,
                height_sampling_all_modes=True, rel_nominal_height=.25,
                pose_start_iteration=int(start_iteration),
                pose_end_iteration=int(start_iteration+ramp_updates))


def configure_height_range(cfg, bounds, start_iteration, ramp_updates=10000):
    fields = height_range_fields(bounds, start_iteration, ramp_updates)
    for name, value in fields.items():
        setattr(cfg.commands, name, value)
    return dict(recipe='variable_height_v1', commands=fields,
                height_rate_m_s=cfg.commands.height_rate, height_frame='waist_z_link',
                rewards_unchanged=True)


def expected_height_range_recipe(before, bounds, start_iteration, ramp_updates=10000):
    expected = deepcopy(before)
    fields = height_range_fields(bounds, start_iteration, ramp_updates)
    for name, value in fields.items():
        if value is None:
            value = 'null'
        elif isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, tuple):
            value = [str(v) for v in value]
        else:
            value = str(value)
        expected['commands'][name] = value
    return expected


def checkpoint_height_range(checkpoint):
    cfg = yaml.load((Path(checkpoint).parent/'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    return tuple(map(float, cfg['commands']['height_range']))


def range_height_scenarios(scenarios, bounds):
    """Retain the nominal benchmark and add lower-height locomotion tests."""
    low, high = validate_height_range(bounds)
    rows = height_scenarios(scenarios, high)
    for height in (low, (low+high)/2):
        for name, velocity in [('stand', (0,0,0)), ('forward', (.3,0,0)),
                               ('backward', (-.3,0,0)), ('lateral_pos', (0,.2,0)),
                               ('lateral_neg', (0,-.2,0)), ('turn_pos', (0,0,.5)),
                               ('turn_neg', (0,0,-.5))]:
            rows.append((f'height_{height:.3f}_{name}', [*velocity, height, 0.]))
    return rows


def configure_fixed_height(cfg, height):
    height = validate_height(height)
    cfg.commands.nominal_height = height
    for name in HEIGHT_RANGES:
        setattr(cfg.commands, name, (height, height))
    return height


def checkpoint_height(checkpoint, require_fixed=True):
    cfg = yaml.load((Path(checkpoint).parent/'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    commands = cfg['commands']
    height = validate_height(commands['nominal_height'])
    if require_fixed and any([float(v) for v in commands[name]] != [height, height] for name in HEIGHT_RANGES):
        raise ValueError('Expected a fixed-height checkpoint')
    return height


def expected_height_recipe(before, height):
    """Permit exactly four commanded-height fields to change; nothing else."""
    height = validate_height(height)
    old = float(before['commands']['nominal_height'])
    if any([float(v) for v in before['commands'][name]] != [old, old] for name in HEIGHT_RANGES):
        raise ValueError('Height adaptation requires a fixed-height source recipe')
    expected = deepcopy(before)
    expected['commands']['nominal_height'] = str(height)
    for name in HEIGHT_RANGES:
        expected['commands'][name] = [str(height), str(height)]
    return expected


def height_scenarios(scenarios, height):
    height = validate_height(height)
    return [(name, [*command[:3], height, command[4]]) for name, command in scenarios]


def apply_onnx_height(policy, path):
    """Set the in-memory simulator adapter only; never modify deployment files."""
    import onnx
    metadata = {v.key: v.value for v in onnx.load(path, load_external_data=False).metadata_props}
    if 'height_command' in metadata:
        policy.info['height_command'] = validate_height(metadata['height_command'])
    return float(policy.info['height_command'])
