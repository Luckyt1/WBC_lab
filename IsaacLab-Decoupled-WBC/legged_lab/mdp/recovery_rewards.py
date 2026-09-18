"""Rewards used only by the pelvis tracking recovery task."""

import torch
from isaaclab.utils import math as math_utils
from legged_lab.utils.recovery_tracking import angular_tracking, smooth_stance, standing_foot_lift, yaw_rate_tracking


def track_yaw_rate(env, min_std=.15, max_std=.3, command_scale=.5):
    return yaw_rate_tracking(env.command_generator.command[:, 2], env.base_ang_vel_w[:, 2],
                             min_std, max_std, command_scale)


def feet_stance(env, sensor_cfg, ramp_s=.15):
    contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids]
    return smooth_stance(contact, env.command_generator.command, ramp_s)


def feet_lift_standing(env, sensor_cfg):
    contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids]
    return standing_foot_lift(contact, env.command_generator.command)


def track_torso_yaw(env, std, asset_cfg):
    quat = env.scene[asset_cfg.name].data.body_quat_w
    pelvis = math_utils.yaw_quat(quat[:, asset_cfg.body_ids[0]])
    torso = math_utils.yaw_quat(quat[:, asset_cfg.body_ids[1]])
    relative = math_utils.quat_mul(math_utils.quat_conjugate(pelvis), torso)
    yaw = 2 * torch.atan2(relative[:, 3], relative[:, 0])
    return angular_tracking(yaw, env.command_generator.command[:, 6], std)
