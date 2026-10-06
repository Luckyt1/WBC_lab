"""Standing-first command mixture for the AMASS arm-motion robust task."""


def configure_teleop_standing(cfg, standing_fraction=.80):
    """Change sampling only; arm amplitude and disturbance curricula stay intact.

    train.py also enables the stop-position penalty for this named recipe.
    These are command-draw probabilities, not guaranteed rollout time fractions:
    failures and resets can change the observed distribution.
    """
    if not 0. <= standing_fraction <= 1.:
        raise ValueError('Standing command fraction must be between 0 and 1')
    # Preserve the established 2:1:1 movement mixture and exact legacy defaults.
    movement_fraction = round(1. - standing_fraction, 15)
    cfg.commands.rel_standing_envs = standing_fraction
    cfg.commands.rel_turning_envs = movement_fraction / 4
    cfg.commands.rel_lateral_envs = movement_fraction / 4
    cfg.commands.rel_crouch_yaw_envs = 0.
    cfg.commands.initial_mode_fractions = None
    return dict(command_probabilities=dict(standing=standing_fraction, forward_backward=movement_fraction / 2,
                                          lateral=movement_fraction / 4, turning=movement_fraction / 4),
                arm_neutral_probability=cfg.arm_motion.amass_neutral_fraction,
                arm_amass_probability=1. - cfg.arm_motion.amass_neutral_fraction,
                standing_position_penalty=True)
