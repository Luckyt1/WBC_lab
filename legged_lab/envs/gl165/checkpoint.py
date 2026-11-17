"""Restore saved observations and controller parameters before building the environment."""

import math
from pathlib import Path

import yaml

from legged_lab.assets.gl165.model import JOINT_NAMES, POLICY_JOINT_NAMES


def restore_checkpoint_config(cfg, checkpoint):
    """Restore the saved gait, named joint gains and policy action scaling."""
    # BaseLoader reads scalar metadata without constructing YAML Python objects.
    saved = yaml.load((Path(checkpoint).parent / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    gait = saved.get('gait')
    if gait is None:
        cfg.gait.enabled = False
    else:
        if int(gait['observation_version']) != 1:
            raise ValueError('Unsupported GL165 gait observation version')
        for name, default in vars(cfg.gait).items():
            if name.startswith('_'):
                continue
            if name not in gait:
                raise ValueError(f'Missing checkpoint gait setting: {name}')
            value = gait[name]
            setattr(cfg.gait, name, value.lower() == 'true' if isinstance(default, bool) else type(default)(value))
    if not cfg.gait.enabled:
        for name in ('feet_lateral_band', 'gait_contact', 'gait_clearance', 'gait_fore_aft', 'stand_foot_alignment'):
            getattr(cfg.reward, name).weight = 0.
        cfg.reward.feet_air_time.weight = .5
    # Changing the fresh-training defaults must not silently change an old
    # policy's action interpretation or PD controller during play / resume.
    try:
        actuators = saved['scene']['robot']['actuators']['joints']
        gains = {
            name: {'kp': float(actuators['stiffness'][name]), 'kd': float(actuators['damping'][name])}
            for name in JOINT_NAMES
        }
        scales = saved['robot']['action_scale']
        scales = [float(value) for value in scales] if isinstance(scales, list) else float(scales)
        clip_actions = float(saved['normalization']['clip_actions'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Missing or invalid saved GL-165 controller parameters beside {checkpoint}') from exc
    scale_values = scales if isinstance(scales, list) else [scales] * len(POLICY_JOINT_NAMES)
    if (len(scale_values) != len(POLICY_JOINT_NAMES)
            or any(not math.isfinite(v) or v <= 0 for v in scale_values)
            or not math.isfinite(clip_actions) or clip_actions <= 0
            or any(not math.isfinite(v) for pair in gains.values() for v in pair.values())
            or any(pair['kp'] <= 0 or pair['kd'] < 0 for pair in gains.values())):
        raise ValueError(f'Invalid saved GL-165 controller values beside {checkpoint}')
    cfg.checkpoint_joint_gains = gains
    cfg.robot.action_scale = scales
    cfg.normalization.clip_actions = clip_actions
    cfg.control_parameters_source = {'checkpoint_env': str(Path(checkpoint).parent / 'params/env.yaml')}
    return cfg.gait.enabled


# Preserve the original helper name for existing external playback scripts.
configure_checkpoint_gait = restore_checkpoint_config
