"""Experimental terms used only by the stalled-policy ablation tool."""
import torch


def lateral_knee_width(env, asset_cfg, ideal_width=.27, std=.1):
    positions = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
    delta = positions[:, 0] - positions[:, 1]
    w, x, y, z = env.base_quat_w.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    lateral = -yaw.sin()*delta[:, 0] + yaw.cos()*delta[:, 1]
    return torch.exp(-((lateral-ideal_width)/std).square())


def moving_upright(env):
    return (~env.command_generator.is_standing_env).float() * (-env.base_projected_gravity_b[:, 2] > .9)


def load_transfer(env):
    """A bounded signal before liftoff; both-airborne states earn nothing."""
    force = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2].clamp(min=0)
    total = force.sum(-1)
    return ((force[:, 0]-force[:, 1]).abs()/total.clamp(min=1)).square() * (total > 50) * moving_upright(env)


def swing_clearance(env, clearance=.035):
    """Modest relative ankle clearance with exactly one supported foot."""
    height = env.robot.data.body_pos_w[:, env.feet_body_ids, 2]
    contact = env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids] > 0
    raised = (height-height.flip(-1)).clamp(min=0)
    credit = (raised/clearance).clamp(max=1) * ~contact
    return credit.sum(-1) * (contact.sum(-1)==1) * moving_upright(env)


def alternating_touchdown(env, min_air_time=.08):
    """Credit alternating landings after real airtime, once per touchdown."""
    data = env.contact_sensor.data
    times = data.current_contact_time[:, env.feet_cfg.body_ids]
    air = data.last_air_time[:, env.feet_cfg.body_ids]
    first = (times > 0) & (times <= env.step_dt*1.01) & (air >= min_air_time)
    if not hasattr(env, '_ablation_last_landed'):
        env._ablation_last_landed = torch.full_like(env.episode_length_buf, -1)
    # No episodic memory crosses reset or a standing segment.
    reset = (env.episode_length_buf <= 1) | env.command_generator.is_standing_env
    env._ablation_last_landed[reset] = -1
    side = first.long().argmax(-1)
    eligible = (first.sum(-1)==1) & ~reset & (moving_upright(env)>0)
    alternating = eligible & (env._ablation_last_landed>=0) & (side != env._ablation_last_landed)
    env._ablation_last_landed[eligible] = side[eligible]
    # RewardManager multiplies by dt: this is an event reward, not a rate.
    return alternating.float()/env.step_dt


def scaled_during_motion(env, original_func, original_params, moving_scale):
    value = original_func(env, **original_params)
    scale = torch.where(env.command_generator.is_standing_env, 1., moving_scale)
    return value*scale
