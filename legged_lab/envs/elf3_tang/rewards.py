"""WBC-style tracking, with pelvis locomotion independent of torso yaw."""

import math

import torch

from .commands import quat_yaw, rotate_inverse, wrap_to_pi, yaw_frame_velocity


def pelvis_height(env):
    # This task uses a plane; env origin z is the ground height, not torso height.
    return env.robot.data.body_pos_w[:, env.pelvis_body_id, 2] - env.scene.env_origins[:, 2]


def waist_yaw(env):
    quat = env.robot.data.body_quat_w
    return wrap_to_pi(quat_yaw(quat[:, env.torso_body_id]) - quat_yaw(quat[:, env.pelvis_body_id]))


def track_pelvis_velocity(env, std):
    velocity = yaw_frame_velocity(env.robot.data.body_quat_w[:, env.pelvis_body_id],
                                  env.robot.data.body_lin_vel_w[:, env.pelvis_body_id])
    error = (velocity - env.command_generator.command[:, :2]).square().sum(-1)
    return torch.exp(-error / std**2)


def track_pelvis_turn_rate(env, std):
    actual = env.robot.data.body_ang_vel_w[:, env.pelvis_body_id, 2]
    return torch.exp(-(actual - env.command_generator.command[:, 2]).square() / std**2)


def track_pelvis_height(env, std):
    return torch.exp(-(pelvis_height(env) - env.command_generator.full_command[:, 3]).square() / std**2)


def track_waist_yaw(env, std):
    error = wrap_to_pi(waist_yaw(env) - env.command_generator.full_command[:, 4])
    return torch.exp(-error.square() / std**2)


def pelvis_vertical_velocity(env):
    return env.robot.data.body_lin_vel_w[:, env.pelvis_body_id, 2].square()


def pelvis_roll_pitch_rate(env):
    return env.robot.data.body_ang_vel_w[:, env.pelvis_body_id, :2].square().sum(-1)


def body_upright(env, body_id_attr):
    quat = env.robot.data.body_quat_w[:, getattr(env, body_id_attr)]
    gravity = rotate_inverse(quat, env.robot.data.GRAVITY_VEC_W)
    return gravity[:, :2].square().sum(-1)


def feet_air_time(env, threshold):
    """Reward single support even for small translation or pure-turn commands."""
    data = env.contact_sensor.data
    contact_time = data.current_contact_time[:, env.feet_cfg.body_ids]
    air_time = data.current_air_time[:, env.feet_cfg.body_ids]
    contact = contact_time > 0
    duration = torch.where(contact, contact_time, air_time).amin(-1).clamp(max=threshold)
    return duration * (contact.sum(-1) == 1) * ~env.command_generator.is_standing_env


def standing_feet_contact(env, min_contact_time=0.5):
    duration = env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids].amin(-1)
    return (torch.tanh(duration / min_contact_time) * (duration > min_contact_time)
            * env.command_generator.is_standing_env)


def standing_support_gate(env, min_contact_time=0.5, contact_ramp_s=0.25):
    duration = env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids].amin(-1)
    return ((duration - min_contact_time) / contact_ramp_s).clamp(0, 1) * env.command_generator.is_standing_env


def standing_feet_flat(env, dead_zone_deg=3.0, std_deg=10.0):
    """Foot roll/pitch only; yaw stays free, and swing feet receive no incentive."""
    quat = env.robot.data.body_quat_w[:, env.feet_body_ids]
    gravity = env.robot.data.GRAVITY_VEC_W[:, None, :].expand(-1, 2, -1)
    local = rotate_inverse(quat, gravity)
    tilt = torch.atan2(local[..., :2].norm(dim=-1), -local[..., 2])
    excess = (tilt - math.radians(dead_zone_deg)).clamp(min=0)
    return torch.exp(-excess.square() / math.radians(std_deg)**2).mean(-1) * standing_support_gate(env)


def standing_center_of_mass(env, dead_zone=0.08, std=0.15):
    """Soft ankle-midpoint reference, only during sustained standing support.

    The dead zone is an approximation, not a geometric support polygon or a
    dynamic balance criterion. Include externally driven arms in whole-body CoM.
    """
    if not hasattr(env, '_tang_mass_weights'):
        # First reward runs after startup mass randomization (masses stay fixed at reset).
        masses = env.robot.root_physx_view.get_masses().to(env.device)
        env._tang_mass_weights = masses / masses.sum(-1, keepdim=True)
    com = (env.robot.data.body_com_pos_w * env._tang_mass_weights[..., None]).sum(1)
    center = env.robot.data.body_pos_w[:, env.feet_body_ids].mean(1)
    excess = ((com[:, :2] - center[:, :2]).norm(dim=-1) - dead_zone).clamp(min=0)
    return torch.exp(-excess.square() / std**2) * standing_support_gate(env)


def knee_inward(env, min_width=0.14):
    """Penalize signed lateral knee narrowing/crossing, without fixing step length."""
    knees = env.robot.data.body_pos_w[:, env.knee_body_ids]
    separation = yaw_frame_velocity(env.robot.data.body_quat_w[:, env.pelvis_body_id],
                                    knees[:, 0] - knees[:, 1])[:, 1]
    return ((min_width - separation).clamp(min=0) / min_width).square()


def raw_action_target_excess(env):
    # Use the newest policy action before delay and clipping, in policy joint order.
    raw = env.raw_actions
    ids = env.policy_joint_ids
    target = env.robot.data.default_joint_pos[:, ids] + raw * env.action_scale
    limits = env.robot.data.soft_joint_pos_limits[:, ids]
    return (limits[..., 0] - target).clamp(min=0) + (target - limits[..., 1]).clamp(min=0)


def raw_action_target_limits(env):
    return raw_action_target_excess(env).sum(-1)


def policy_torque_ratio(env):
    ids = env.policy_joint_ids
    return env.robot.data.computed_torque[:, ids].abs() / env.robot.data.joint_effort_limits[:, ids].clamp(min=1e-6)


def joint_torque_excess(env, soft_limit=0.9):
    # Mean, normalized by each joint's limit; external arm torques are excluded.
    return (policy_torque_ratio(env) - soft_limit).clamp(min=0).mean(-1)


def transition_metrics(env):
    """Measure before command resampling and resets so logs describe this action."""
    command = env.command_generator.full_command
    velocity = yaw_frame_velocity(env.robot.data.body_quat_w[:, env.pelvis_body_id],
                                  env.robot.data.body_lin_vel_w[:, env.pelvis_body_id])
    target_excess = raw_action_target_excess(env)
    torque_ratio = policy_torque_ratio(env)
    metrics = {
        'Tracking/velocity_xy_error': (velocity - command[:, :2]).norm(dim=-1).mean(),
        'Tracking/turn_rate_mae': (env.robot.data.body_ang_vel_w[:, env.pelvis_body_id, 2] - command[:, 2]).abs().mean(),
        'Tracking/pelvis_height_mae': (pelvis_height(env) - command[:, 3]).abs().mean(),
        'Tracking/waist_yaw_mae': wrap_to_pi(waist_yaw(env) - command[:, 4]).abs().mean(),
        'ActionTargets/raw_excess_rad_sum': target_excess.sum(-1).mean(),
        'ActionTargets/outside_fraction': (target_excess > 0).float().mean(),
        'ActionTargets/clipped_fraction': (env.raw_actions != env.bounded_actions).float().mean(),
        # The runner averages step maxima over the rollout (not a rollout maximum).
        'ActionTargets/raw_abs_step_max': env.raw_actions.abs().max(),
        'Torques/over_90_percent_fraction': (torque_ratio > 0.9).float().mean(),
        'Torques/mean_peak_ratio': torque_ratio.amax(-1).mean(),
    }
    # Resolve anatomical names instead of assuming indices in the 15-action vector.
    if hasattr(env.robot.data, 'joint_names'):
        for column, joint_id in enumerate(env.policy_joint_ids):
            name = env.robot.data.joint_names[joint_id]
            if 'knee' in name:
                metrics[f'ActionTargets/{name}_clipped_fraction'] = (
                    env.raw_actions[:, column] != env.bounded_actions[:, column]).float().mean()
    return metrics
