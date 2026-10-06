"""Keep fixed-height training, continuation audits and evaluation in agreement."""
from copy import deepcopy
import math
from pathlib import Path

import yaml

HEIGHT_RANGES = ('height_initial_range', 'height_range', 'sampling_height_range')


def validate_height(height):
    height = float(height)
    if not math.isfinite(height) or not .5 <= height <= .8235:
        raise ValueError('ELF3 fixed pelvis height must be between 0.5 and 0.8235 m')
    return height


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
