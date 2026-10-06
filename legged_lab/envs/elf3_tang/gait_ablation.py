"""One-factor continuation experiments; never applied to normal training."""

VARIANTS = ('control', 'slide', 'air_time', 'swing_quality', 'mixed_yaw')


def configure_gait_ablation(cfg, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown gait ablation: {variant}')
    before = {name: getattr(cfg.reward, name).weight for name in ('feet_slide', 'feet_air_time')}
    # Fail closed if the baseline changes: these are comparisons against the
    # retained 26k standing-refinement recipe, not generic multipliers.
    if before != {'feet_slide': -.25, 'feet_air_time': .15}:
        raise ValueError(f'Unexpected baseline gait weights: {before}')
    details = {}
    if variant == 'slide':
        cfg.reward.feet_slide.weight = -.5
    elif variant == 'air_time':
        cfg.reward.feet_air_time.weight = .6
    elif variant == 'swing_quality':
        from isaaclab.managers import RewardTermCfg
        from .swing_quality_term import CompletedSwingReward
        params = dict(min_air=.12, max_air=.35, min_clearance=.02, target_clearance=.03)
        cfg.reward.feet_air_time.weight = 0.
        cfg.reward.completed_swing_quality = RewardTermCfg(func=CompletedSwingReward, weight=1., params=params)
        details = {'replacement': 'completed_swing_quality', 'params': params,
                   'clearance_geometry': 'foot_collision_hulls.json; flat plane only'}
    elif variant == 'mixed_yaw':
        if cfg.commands.mixed_yaw_fraction != 0 or cfg.commands.sagittal_speed_range is None:
            raise ValueError('Mixed-yaw comparison requires the isolated sagittal baseline')
        cfg.commands.mixed_yaw_fraction = .5
        cfg.commands.mixed_yaw_range = (.02, .15)
        details = {'mixed_yaw_fraction_before': 0., 'mixed_yaw_fraction_after': .5,
                   'mixed_yaw_range': [.02, .15], 'applies_to': 'sagittal mode only'}
    return {'variant': variant, 'before': before,
            'after': {name: getattr(cfg.reward, name).weight for name in before}, 'details': details}
