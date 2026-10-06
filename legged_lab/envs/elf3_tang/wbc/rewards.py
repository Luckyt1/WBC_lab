# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from __future__ import annotations

import math
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.envs.mdp.rewards import joint_pos_limits

# Ported from WBC; only the 7-channel command accessor is adapted.
# Tang keeps five policy command channels; roll and pitch targets are zero.

def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    vel_yaw = quat_apply_inverse(
        math_utils.yaw_quat(env.base_quat_w), env.base_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.wbc_command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    ang_vel_error = torch.square(env.command_generator.wbc_command[:, 2] - env.base_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.square(env.base_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.sum(torch.square(env.base_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    # print(env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :])
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.sum(torch.square(env.base_projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg,
                                use_standing_mask: bool = False) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    moving = (~env.command_generator.is_standing_env if use_standing_mask else
              (torch.norm(env.command_generator.wbc_command[:, :2], dim=1)
               + torch.abs(env.command_generator.wbc_command[:, 2])) > 0.1)
    reward *= moving
    return reward


def feet_contact_standing_stable(env: BaseEnv, sensor_cfg: SceneEntityCfg, min_contact_time: float = 0.5,
                                 use_standing_mask: bool = False) -> torch.Tensor:
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]  # (num_envs, 2)
    
    
    lin_vel_cmd = torch.norm(env.command_generator.wbc_command[:, :2], dim=1)
    ang_vel_cmd = torch.abs(env.command_generator.wbc_command[:, 2])
    is_zero_command = (env.command_generator.is_standing_env if use_standing_mask
                       else (lin_vel_cmd + ang_vel_cmd) <= 0.1)
    
    
    stable_contact = contact_time > min_contact_time  # (num_envs, 2)
    num_stable_feet = torch.sum(stable_contact.int(), dim=1)  # (num_envs,)
    
   
    min_contact_duration = torch.min(contact_time, dim=1)[0]  # (num_envs,)
    
   
    reward = torch.where(
        num_stable_feet == 2,
        torch.tanh(min_contact_duration / min_contact_time),  
        torch.zeros_like(num_stable_feet, dtype=torch.float32)  
    )
    
    
    reward = torch.where(is_zero_command, reward, torch.zeros_like(reward))
    
    return reward


def feet_flat_standing(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    dead_zone_deg: float = 3.0,
    std_deg: float = 10.0,
    min_contact_time: float = 0.5,
    contact_ramp_s: float = 0.25,
    roll_dead_zone: float = 0.70,
    pitch_dead_zone: float = 1.20,
    roll_scale: float = 0.35,
    pitch_scale: float = 0.45,
) -> torch.Tensor:
    """Encourage level feet in command-generator standing environments.

    The foot tilt is measured from gravity expressed in each ankle-roll link
    frame, so foot yaw is unconstrained.  A small angular dead zone avoids
    rewarding unnecessary ankle micro-corrections.  A sustained-contact gate
    keeps this soft objective out of swing.  The reward remains fully active
    for moderate torso roll/pitch commands and decays only beyond the configured
    command dead zones; torso yaw does not affect it.
    """
    if min(std_deg, contact_ramp_s, roll_scale, pitch_scale) <= 0.0:
        raise ValueError("foot-flat reward scales must be positive")
    if min(dead_zone_deg, min_contact_time, roll_dead_zone, pitch_dead_zone) < 0.0:
        raise ValueError("foot-flat reward dead zone and minimum contact time must be non-negative")
    if len(asset_cfg.body_ids) != 2 or len(sensor_cfg.body_ids) != 2:
        raise ValueError("foot-flat reward requires exactly two foot bodies")

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity_w = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, 2, -1)
    gravity_foot = quat_apply_inverse(
        foot_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)
    ).reshape(-1, 2, 3)

    # atan2(||gravity_xy||, -gravity_z) is the full [0, pi] foot tilt
    # relative to world-up and is independent of foot yaw.
    foot_tilt = torch.atan2(
        torch.linalg.vector_norm(gravity_foot[..., :2], dim=-1), -gravity_foot[..., 2]
    )
    dead_zone = math.radians(dead_zone_deg)
    decay_scale = math.radians(std_deg)
    tilt_excess = torch.clamp(foot_tilt - dead_zone, min=0.0)
    flatness = torch.exp(-torch.square(tilt_excess / decay_scale)).mean(dim=1)

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    min_foot_contact_time = torch.min(contact_time, dim=1).values
    contact_gate = torch.clamp(
        (min_foot_contact_time - min_contact_time) / contact_ramp_s, 0.0, 1.0
    )

    command = env.command_generator.wbc_command
    roll_excess = torch.clamp(torch.abs(command[:, 4]) - roll_dead_zone, min=0.0)
    pitch_excess = torch.clamp(torch.abs(command[:, 5]) - pitch_dead_zone, min=0.0)
    pose_gate = torch.exp(
        -torch.square(roll_excess / roll_scale)
        -torch.square(pitch_excess / pitch_scale)
    )
    standing_gate = env.command_generator.is_standing_env.to(dtype=flatness.dtype)

    return standing_gate * contact_gate * pose_gate * flatness


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def track_body_height_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    # print(body_height)
    height_error = torch.square(env.command_generator.wbc_command[:, 3] - body_height)
    return torch.exp(-height_error / std**2)


def track_body_roll_exp(
    env: BaseEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link", "pelvis"]),
) -> torch.Tensor:
    """Track torso roll angle relative to pelvis yaw frame.
    
    Reference frame: pelvis yaw frame (horizontal plane aligned with pelvis heading)
    Measures: torso's roll deviation from pelvis yaw frame
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    quat_w_torso = asset.data.body_quat_w[:, asset_cfg.body_ids[1]]   # torso in world
    quat_w_pelvis = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # pelvis in world
    
    # Step 1: Get pelvis yaw frame quaternion (horizontal, same yaw as pelvis)
    pelvis_yaw_quat = math_utils.yaw_quat(quat_w_pelvis)
    
    # Step 2: Transform torso orientation to pelvis yaw frame
    # quat_torso_in_pelvis_yaw = pelvis_yaw_quat^(-1) * quat_w_torso
    quat_torso_in_pelvis_yaw = math_utils.quat_mul(
        math_utils.quat_conjugate(pelvis_yaw_quat),
        quat_w_torso
    )
    
    # Step 3: Extract roll angle directly from the relative quaternion
    # This is more direct than using gravity vector
    # `euler_xyz_from_quat` returns [0, 2pi) on Isaac Lab <= 2.1.0 and (-pi, pi]
    # on >= 2.2.0; wrap_to_pi normalizes to [-pi, pi] on every version so negative
    # roll commands are tracked correctly regardless of the installed Isaac Lab.
    roll_approx, _, _ = math_utils.euler_xyz_from_quat(quat_torso_in_pelvis_yaw)
    roll_approx = math_utils.wrap_to_pi(roll_approx)
    
    roll_error = torch.square(roll_approx - env.command_generator.wbc_command[:, 4])
    return torch.exp(-roll_error / (std**2))


def track_body_pitch_exp(
    env: BaseEnv, 
    std: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link", "pelvis"])
) -> torch.Tensor:
    """Track torso pitch angle relative to pelvis yaw frame.
    
    Reference frame: pelvis yaw frame (horizontal plane aligned with pelvis heading)
    Measures: torso's pitch deviation from pelvis yaw frame
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    quat_w_torso = asset.data.body_quat_w[:, asset_cfg.body_ids[1]]   # torso in world
    quat_w_pelvis = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # pelvis in world
    
    # Step 1: Get pelvis yaw frame quaternion (horizontal, same yaw as pelvis)
    pelvis_yaw_quat = math_utils.yaw_quat(quat_w_pelvis)
    
    # Step 2: Transform torso orientation to pelvis yaw frame
    quat_torso_in_pelvis_yaw = math_utils.quat_mul(
        math_utils.quat_conjugate(pelvis_yaw_quat),
        quat_w_torso
    )
    
    # Step 3: Extract pitch angle directly from the relative quaternion
    # This is more direct than using gravity vector
    # See track_body_roll_exp: wrap_to_pi normalizes the [0, 2pi) vs (-pi, pi]
    # difference between Isaac Lab versions so the reward matches on both envs.
    _, pitch_approx, _ = math_utils.euler_xyz_from_quat(quat_torso_in_pelvis_yaw)
    pitch_approx = math_utils.wrap_to_pi(pitch_approx)

    pitch_error = torch.square(env.command_generator.wbc_command[:, 5] - pitch_approx)
    return torch.exp(-pitch_error / std**2)


def track_body_yaw_exp(
    env: BaseEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link", "pelvis"]),
    wrap_error: bool = False,
) -> torch.Tensor:
    """Track torso yaw angle relative to pelvis yaw frame.
    
    Reference frame: pelvis yaw frame (horizontal plane aligned with pelvis heading)
    Measures: torso's yaw deviation from pelvis heading direction
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get world quaternions for both bodies
    quat_w_torso = asset.data.body_quat_w[:, asset_cfg.body_ids[1]]   # torso_link
    quat_w_pelvis = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # pelvis

    # Extract yaw-only quaternions (eliminates roll/pitch)
    yaw_quat_torso = math_utils.yaw_quat(quat_w_torso)
    yaw_quat_pelvis = math_utils.yaw_quat(quat_w_pelvis)
    
    # Compute relative rotation: q_rel = q_pelvis^(-1) * q_torso
    # This gives the yaw rotation from pelvis heading to torso heading
    quat_rel = math_utils.quat_mul(
        math_utils.quat_conjugate(yaw_quat_pelvis),
        yaw_quat_torso
    )
    
    # Extract yaw angle from relative quaternion
    # For a yaw-only quaternion [w, 0, 0, z], yaw = 2 * atan2(z, w)
    yaw_diff = 2.0 * torch.atan2(quat_rel[:, 3], quat_rel[:, 0])
    
    error = yaw_diff - env.command_generator.wbc_command[:, 6]
    if wrap_error:
        error = torch.atan2(torch.sin(error), torch.cos(error))
    yaw_error = error.square()
    reward = torch.exp(-yaw_error / (std**2))

    return reward


def knee_stance_width(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ideal_width: float = 0.26,
    std: float = 0.1,
    lateral_only: bool = False,
) -> torch.Tensor:
    """Penalize knee stance width deviation from ideal width.
    
    Prevents "knee-in"  posture especially during squatting.
    Encourages the robot to maintain proper knee alignment with appropriate spacing.
    
    
    Args:
        env: The environment.
        asset_cfg: Asset configuration. Should specify 2 knee bodies (left and right).
        ideal_width: Ideal distance between knees in meters (default: 0.20m for G1).
                     Typically narrower than feet width (0.28m) but not too close.
        std: Standard deviation for exponential reward kernel (default: 0.08).
             Smaller than feet_stance_width to be more strict about knee alignment.
    
    Returns:
        Reward tensor encouraging knees to maintain ideal width.
        Range [0, 1], where 1 = perfect knee spacing.
    """
    assert len(asset_cfg.body_ids) == 2, "Must specify exactly 2 knee bodies (left and right)"
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get knee positions in world frame
    knee_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (num_envs, 2, 3)
    
    # Calculate distance between knees (only XY plane)
    # We ignore Z because knees should be at similar height anyway
    knee_pos_xy = knee_pos[:, :, :2]  # (num_envs, 2, 2) - only X and Y
    delta = knee_pos_xy[:, 0] - knee_pos_xy[:, 1]
    if lateral_only:
        # Signed left-minus-right width in the pelvis heading frame. Fore/aft
        # stepping is unconstrained; crossing the knees still loses reward.
        w, x, y, z = env.base_quat_w.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        distance = -yaw.sin() * delta[:, 0] + yaw.cos() * delta[:, 1]
    else:
        distance = torch.norm(delta, dim=-1)
    
    # Penalize deviation from ideal width using exponential kernel
    width_error = torch.square(distance - ideal_width)
    reward = torch.exp(-width_error / (std ** 2))
    
    return reward


def center_of_gravity_tracking(
    env: BaseEnv,
    sigma: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    standing_only: bool = False,
) -> torch.Tensor:
    """Center-of-Gravity (CoG) tracking reward for maintaining stability.
    
    This reward maintains stability by keeping the whole-body center of gravity
    projection near the support base (midpoint between ankles).
    
    IMPORTANT: Correctly handles domain randomization by:
    1. Querying actual body masses from PhysX (includes add_base_mass effects)
    2. Using body_com_pos_w (center of mass positions) for CoG calculation
    
    Args:
        env: The RL environment
        sigma: Standard deviation for exponential kernel (default: 0.2)
        asset_cfg: Must specify 2 foot bodies (e.g., ankle links)
        
    Returns:
        Reward tensor of shape (num_envs,), range [0, 1]
    """
    # Validate configuration
    assert len(asset_cfg.body_ids) == 2, (
        f"Must specify exactly 2 foot bodies, got {len(asset_cfg.body_ids)}. "
        f"Example: SceneEntityCfg('robot', body_names=['left_ankle_roll_link', 'right_ankle_roll_link'])"
    )
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # ========== Step 1: Get randomized body masses (OPTIMIZED with caching) ==========
    # Use cached masses from environment (set once after mode="startup" randomization)
    # This avoids expensive PhysX API call (get_masses) on every step, improving performance by 30-50%
    if not hasattr(env, '_cached_body_masses'):
        # Fallback: First call or cache not initialized yet, query once and cache
        env._cached_body_masses = asset.root_physx_view.get_masses().to(device=env.device)
        
    
    body_masses = env._cached_body_masses  # (num_envs, num_bodies) - Zero overhead!
    
    # ========== Step 2: Compute whole-body center of gravity ==========
    #  FIXED: Use body_com_pos_w (center of mass) NOT body_pos_w (link frame)
    body_com_pos_w = asset.data.body_com_pos_w  # (num_envs, num_bodies, 3)
    
    # Compute total mass per environment
    total_mass = body_masses.sum(dim=1)  # (num_envs,)
    
    # Mass-weighted CoG: Σ(m_i × p_CoM,i) / Σ(m_i)
    masses_expanded = body_masses.unsqueeze(-1)  # (num_envs, num_bodies, 1)
    weighted_positions = body_com_pos_w * masses_expanded  #  Now positions match masses!
    cog_position = weighted_positions.sum(dim=1) / total_mass.unsqueeze(-1)  # (num_envs, 3)
    
    # Extract horizontal projection
    cog_xy = cog_position[:, :2]  # (num_envs, 2)
    
    # ========== Step 3: Compute feet support reference ==========
    # Get ankle positions (use body_pos_w for actual link positions, which is fine here)
    left_ankle_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]   # (num_envs, 3)
    right_ankle_pos = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :]  # (num_envs, 3)
    
    # Midpoint between ankles
    feet_center = (left_ankle_pos + right_ankle_pos) / 2.0
    feet_xy = feet_center[:, :2]  # (num_envs, 2)
    
    # ========== Step 4: Compute reward ==========
    distance = torch.norm(cog_xy - feet_xy, dim=-1)  # (num_envs,)
    reward = torch.exp(-torch.square(distance) / (sigma ** 2))
    if standing_only:
        reward = reward * env.command_generator.is_standing_env
    
    return reward


def joint_torque_limits(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_torque_limit: float = 0.6,
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # waist_ids = [2,5,8] # usd order for waist yaw,roll,pitch 
    torques = asset.data.computed_torque[:, asset_cfg.joint_ids]  
    torque_limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]  
    normalized_torques = torch.abs(torques) / torque_limits
    excess = (normalized_torques - soft_torque_limit).clip(min=0.0)
    out_of_limits = torch.sum(excess, dim=1)  
    
    return out_of_limits
