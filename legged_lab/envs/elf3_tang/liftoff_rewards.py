"""Explicit B+C+D reward recipe; other training recipes remain unchanged."""
from pathlib import Path

RECIPE = 'bcd_v1'
LOAD_TRANSFER_WEIGHT = .2


def load_transfer(env):
    """Bounded pre-liftoff credit for unequal vertical loading while moving.

    This is a reward rate, integrated by RewardManager with dt. Standing,
    airborne and strongly tilted states get no credit. It does not by itself
    certify a step: clearance and alternating support remain evaluation metrics.
    """
    force = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2].clamp(min=0)
    total = force.sum(-1)
    imbalance = (force[:, 0] - force[:, 1]) / total.clamp(min=1.)
    eligible = ((~env.command_generator.is_standing_env)
                & (-env.base_projected_gravity_b[:, 2] > .9) & (total > 50.))
    return imbalance.square() * eligible


def configure_liftoff_rewards(cfg):
    from isaaclab.managers import RewardTermCfg

    cfg.reward.center_of_gravity_tracking.params = dict(
        cfg.reward.center_of_gravity_tracking.params, standing_only=True)
    cfg.reward.knee_stance_width.params = dict(
        cfg.reward.knee_stance_width.params, lateral_only=True)
    cfg.reward.liftoff_load_transfer = RewardTermCfg(func=load_transfer, weight=LOAD_TRANSFER_WEIGHT)
    cfg.liftoff_reward_recipe = RECIPE
    return dict(recipe=RECIPE, center_of_gravity='standing_only',
                knee_distance='signed_lateral_in_pelvis_heading_frame',
                load_transfer_weight=LOAD_TRANSFER_WEIGHT,
                feet_air_time_weight=cfg.reward.feet_air_time.weight)


def checkpoint_recipe(checkpoint):
    import yaml

    cfg = yaml.load((Path(checkpoint).parent / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    recipe = cfg.get('liftoff_reward_recipe')
    if recipe not in (None, RECIPE):
        raise ValueError(f'Unknown liftoff reward recipe: {recipe}')
    return recipe
