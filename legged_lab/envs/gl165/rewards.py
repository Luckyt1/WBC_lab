"""GL-165 balance and posture costs, without height or waist pose commands."""

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat


def track_linear_velocity(env, std):
    velocity = quat_apply_inverse(yaw_quat(env.robot.data.root_quat_w), env.robot.data.root_lin_vel_w)
    error = (env.command_generator.command[:, :2] - velocity[:, :2]).square().sum(-1)
    return torch.exp(-error / std**2)


def standing_mask(env):
    return torch.linalg.vector_norm(env.command_generator.command, dim=-1) < .1


def zero_velocity_command(env, tolerance=1e-6):
    """Use commanded vx/vy/wz, so push-induced motion cannot disable standing support."""
    return (env.command_generator.command.abs() <= tolerance).all(-1)


def standing_foot_load(env, sensor_cfg, threshold, command_tolerance):
    """Mean per-foot vertical-load deficit, active only for a zero velocity command.

    Excess load on one foot cannot compensate for an unloaded opposite foot.
    Tangential force does not count toward supporting the robot's weight.
    """
    forces = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp_min(0.)
    deficit = ((threshold - forces) / threshold).clamp_min(0.)
    return deficit.mean(-1) * zero_velocity_command(env, command_tolerance)


def moving_feet_distance(env, asset_cfg, max_distance, command_tolerance):
    """Penalize excessive horizontal separation of the two feet while moving.

    Use foot-link positions, not contact forces or foot orientations. Vertical
    swing height does not enlarge the distance, and an airborne foot is still
    counted. Turning and lateral commands also activate this soft upper bound.
    """
    feet = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids]
    distance = torch.linalg.vector_norm(feet[:, 0, :2] - feet[:, 1, :2], dim=-1)
    excess = (distance / max_distance - 1.).clamp_min(0.)
    return excess.square() * ~zero_velocity_command(env, command_tolerance)


def signed_foot_separation(data, body_ids):
    """Left minus right in pelvis yaw coordinates; negative Y means crossed feet."""
    feet = data.body_pos_w[:, body_ids]
    return quat_apply_inverse(yaw_quat(data.root_quat_w), feet[:, 0] - feet[:, 1])


def feet_lateral_band(env, asset_cfg, minimum, maximum, scale):
    width = signed_foot_separation(env.scene[asset_cfg.name].data, asset_cfg.body_ids)[:, 1]
    excess = (minimum - width).clamp_min(0.) + (width - maximum).clamp_min(0.)
    return (excess / scale).square()


def gait_contact(env, sensor_cfg, contact_force):
    """Support/swing agreement, with unscored windows around contact transitions."""
    force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp_min(0.)
    contact = (force / contact_force).clamp(0., 1.)
    mismatch = (1. - contact) * env.gait.stance_core + contact * env.gait.swing_core
    return mismatch.mean(-1) * env.gait.activity


def gait_swing_clearance(env, asset_cfg, scale):
    # The link origin is 32 mm above a level sole. This is foot clearance over
    # the plane, not a body-height command. Stance feet have no height target.
    height = (env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
              - env.scene.env_origins[:, 2, None])
    target = env.cfg.gait.foot_origin_ground_height_m + env.gait.clearance
    return (((height - target) / scale).square() * env.gait.swing).mean(-1) * env.gait.activity


def gait_fore_aft(env, asset_cfg, tolerance, scale):
    separation = signed_foot_separation(env.scene[asset_cfg.name].data, asset_cfg.body_ids)[:, 0]
    excess = ((separation - env.gait.fore_aft).abs() - tolerance).clamp_min(0.)
    return (excess / scale).square() * env.gait.activity * env.gait.fore_aft_gate


def standing_foot_alignment(env, asset_cfg, tolerance, scale):
    separation = signed_foot_separation(env.scene[asset_cfg.name].data, asset_cfg.body_ids)[:, 0]
    excess = (separation.abs() - tolerance).clamp_min(0.)
    return (excess / scale).square() * env.gait.standing_alignment * standing_mask(env)


def foot_heading_error(data, body_ids):
    """Unsigned angle of each physical foot +X to the pelvis's horizontal +X.

    Horizontal projection leaves normal swing pitch/roll free. Clamp the
    projection norm so pointing the toe nearly vertically cannot hide a large
    yaw error. The reference follows the pelvis through turns, not velocity:
    lateral/backward walking does not require sideways/backward-facing feet.
    """
    forward = data.FORWARD_VEC_B[:, None, :].expand(-1, len(body_ids), -1)
    foot_forward = quat_apply(data.body_quat_w[:, body_ids], forward)
    local = quat_apply_inverse(yaw_quat(data.root_quat_w)[:, None, :].expand_as(
        data.body_quat_w[:, body_ids]), foot_forward)
    cosine = local[..., 0] / torch.linalg.vector_norm(local[..., :2], dim=-1).clamp_min(.1)
    return torch.acos(cosine.clamp(-1., 1.))


def feet_heading(env, asset_cfg, tolerance):
    """Soft per-foot facing band, active in standing, walking and turning."""
    angle = foot_heading_error(env.scene[asset_cfg.name].data, asset_cfg.body_ids)
    return (angle - tolerance).clamp_min(0.).square().mean(-1)


def stand_still(env):
    error = env.robot.data.root_lin_vel_b[:, :2].square().sum(-1)
    error += .5 * env.robot.data.root_ang_vel_b[:, 2].square()
    return torch.exp(-error / .1) * standing_mask(env)


def stand_feet_contact(env):
    contacts = env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids] > 0
    return contacts.all(-1).float() * standing_mask(env)


def standing_drift(env):
    displacement = env.robot.data.root_pos_w[:, :2] - env.standing_anchor
    return displacement.square().sum(-1) * standing_mask(env)


def alive(env):
    return (~env.reset_buf).float()


def torso_upright(env, asset_cfg):
    """Squared distance between chest up and world up; independent of heading.

    Unlike horizontal gravity alone, this stays costly past a 90-degree tilt.
    The selected GL165 chest frame is upright at the URDF waist zero position.
    """
    data = env.scene[asset_cfg.name].data
    gravity = quat_apply_inverse(data.body_quat_w[:, asset_cfg.body_ids[0]], data.GRAVITY_VEC_W)
    return gravity[:, :2].square().sum(-1) + (gravity[:, 2] + 1.).square()


def joint_excursion(env, asset_cfg, tolerance):
    """Soft cost outside a neutral-pose band; angles inside it remain free."""
    data = env.scene[asset_cfg.name].data
    delta = data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids]
    margin = delta.new_tensor(tolerance)
    return (delta.abs() - margin).clamp_min(0.).square().sum(-1)


def standing_leg_posture(env, asset_cfg, tolerance):
    """Discourage a persistent folded/spread stance, without fixing leg angles."""
    return joint_excursion(env, asset_cfg, tolerance) * standing_mask(env)


def support_knee_bend(env, asset_cfg, sensor_cfg, standing_limit, moving_limit):
    """Penalize deep knees in standing or stance, leaving moving swing legs free.

    Knee joints and foot sensors must both be ordered left, right. All knees
    count during standing, so holding a leg airborne cannot bypass the cost.
    """
    data = env.scene[asset_cfg.name].data
    knees = data.joint_pos[:, asset_cfg.joint_ids]
    contacts = env.scene[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0.
    standing = standing_mask(env)[:, None]
    limit = torch.where(standing, standing_limit, moving_limit)
    excess = (knees - limit).clamp_min(0.)
    return (excess.square() * (standing | contacts)).sum(-1)
