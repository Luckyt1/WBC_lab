"""Dense, command-gated stationary objectives; keep walking and pose changes free."""
import torch

from .commands import rotate_inverse, yaw_frame_velocity, wrap_to_pi
from .rewards import pelvis_height, waist_yaw


def velocity_tracking_error(env):
    """Yaw-frame planar error, with only extreme failure states capped."""
    data = env.robot.data
    velocity = yaw_frame_velocity(data.body_quat_w[:, env.pelvis_body_id],
                                  data.body_lin_vel_w[:, env.pelvis_body_id])
    return (velocity - env.command_generator.command[:, :2]).norm(dim=-1).clamp(max=2.)


def turn_tracking_error(env):
    rate = env.robot.data.body_ang_vel_w[:, env.pelvis_body_id, 2]
    return (rate - env.command_generator.command[:, 2]).abs().clamp(max=3.)


def height_tracking_error(env):
    # Follow the rate-limited current target, including squat transitions.
    return (pelvis_height(env) - env.command_generator.full_command[:, 3]).abs().clamp(max=0.5)


def waist_yaw_tracking_error(env):
    """Dense shortest-angle cost, including far outside the tracking bonus."""
    return wrap_to_pi(waist_yaw(env) - env.command_generator.full_command[:, 4]).abs()


def transition_metrics(env):
    """Separate command modes so a better global mean cannot hide lost turning."""
    data = env.robot.data
    command = env.command_generator.command
    stand = env.command_generator.is_standing_env
    move = command[:, :2].norm(dim=-1) > 1e-6
    turn = ~stand & ~move
    contact = env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids]
    height_error = (pelvis_height(env) - env.command_generator.full_command[:, 3]).abs()
    velocity = yaw_frame_velocity(data.body_quat_w[:, env.pelvis_body_id],
                                  data.body_lin_vel_w[:, env.pelvis_body_id])

    def masked_mean(value, mask):
        return (value * mask).sum() / mask.sum().clamp(min=1)

    result = {
        'Commands/standing_fraction': stand.float().mean(),
        'Commands/turning_fraction': turn.float().mean(),
        'Commands/moving_fraction': move.float().mean(),
        'Standing/horizontal_speed': masked_mean(velocity.norm(dim=-1), stand),
        'Standing/single_support_fraction': masked_mean(((contact > 0).sum(-1) == 1).float(), stand),
        'Standing/sustained_double_support_fraction': masked_mean((contact.amin(-1) > .5).float(), stand),
        'Standing/height_mae': masked_mean(height_error, stand),
        'Turning/turn_rate_mae': masked_mean((data.body_ang_vel_w[:, env.pelvis_body_id, 2] - command[:, 2]).abs(), turn),
        'Moving/velocity_xy_error': masked_mean((velocity - command[:, :2]).norm(dim=-1), move),
    }
    yaw_command = env.command_generator.full_command[:, 4]
    yaw_error = wrap_to_pi(waist_yaw(env) - yaw_command).abs()
    yaw_active = yaw_command.abs() > .02
    result.update({
        'Commands/waist_yaw_active_fraction': yaw_active.float().mean(),
        'Commands/waist_yaw_abs_target_rad': yaw_command.abs().mean(),
        'WaistYaw/active_mae_rad': masked_mean(yaw_error, yaw_active),
        'WaistYaw/standing_mae_rad': masked_mean(yaw_error, stand & yaw_active),
        'WaistYaw/moving_mae_rad': masked_mean(yaw_error, ~stand & yaw_active),
        'WaistYaw/neutral_mae_rad': masked_mean(yaw_error, ~yaw_active),
    })
    mode = getattr(env.command_generator, 'sampled_mode', None)
    if mode is not None:
        lateral, crouch = mode == 3, mode == 4
        result.update({
            'Commands/lateral_fraction': lateral.float().mean(),
            'Commands/crouch_yaw_fraction': crouch.float().mean(),
            'Focused/lateral_vy_mae': masked_mean((velocity[:, 1] - command[:, 1]).abs(), lateral),
            'Focused/crouch_vx_mae': masked_mean((velocity[:, 0] - command[:, 0]).abs(), crouch),
            'Focused/crouch_height_mae': masked_mean(height_error, crouch),
            'Focused/crouch_waist_yaw_mae': masked_mean(wrap_to_pi(
                waist_yaw(env) - env.command_generator.full_command[:, 4]).abs(), crouch),
        })
    if getattr(env, 'use_amass', False):
        # Measure the joint sampling coverage, not just the global arm mixture.
        # A non-neutral AMASS reference may itself contain pauses, so this is
        # deliberately not labelled as actual arm motion or teleoperation.
        amass_reference = (~env.amass_neutral_mask).float()
        result['Standing/amass_reference_fraction'] = masked_mean(amass_reference, stand)
        result['Commands/standing_with_amass_reference_fraction'] = (stand * amass_reference).mean()
    result.update(getattr(env, 'standing_position_metrics', {}))
    return result


def standing_contact(env):
    """Reward each contacting foot immediately, without a half-second dead zone."""
    times = env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids]
    contact = (times > 0).float()
    return contact.mean(-1) * env.command_generator.is_standing_env


def standing_quiet_base(env, velocity_std=0.08, yaw_rate_std=0.15):
    """Near-zero root velocity matters even when its signed average cancels out."""
    data = env.robot.data
    speed_sq = data.body_lin_vel_w[:, env.pelvis_body_id, :2].square().sum(-1)
    yaw_rate_sq = data.body_ang_vel_w[:, env.pelvis_body_id, 2].square()
    return torch.exp(-speed_sq / velocity_std**2 - yaw_rate_sq / yaw_rate_std**2) * env.command_generator.is_standing_env


def standing_foot_motion(env):
    # Include swing as well as stance; the existing slide cost only covers contact.
    speed = env.robot.data.body_lin_vel_w[:, env.feet_body_ids].norm(dim=-1).mean(-1)
    return speed.clamp(max=5.) * env.command_generator.is_standing_env


def standing_base_motion(env, velocity_scale=0.15, yaw_rate_scale=0.3):
    """Dense cost outside the narrow quiet-state bonus, with bounded outliers."""
    data = env.robot.data
    speed_sq = data.body_lin_vel_w[:, env.pelvis_body_id, :2].square().sum(-1)
    yaw_rate_sq = data.body_ang_vel_w[:, env.pelvis_body_id, 2].square()
    return (speed_sq / velocity_scale**2 + yaw_rate_sq / yaw_rate_scale**2).clamp(max=10.) * env.command_generator.is_standing_env


def standing_foot_tilt(env, std=0.20):
    data = env.robot.data
    quat = data.body_quat_w[:, env.feet_body_ids]
    gravity = data.GRAVITY_VEC_W[:, None].expand(-1, 2, -1)
    local = rotate_inverse(quat, gravity)
    tilt = torch.atan2(local[..., :2].norm(dim=-1), -local[..., 2])
    contact = (env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids] > 0).float()
    return (torch.exp(-tilt.square() / std**2) * contact).mean(-1) * env.command_generator.is_standing_env
