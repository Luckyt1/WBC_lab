"""GL-165 PPO environment: waist/leg balance and locomotion under arm IK motion."""

import json
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager, SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from legged_lab.assets.gl165.gl165 import make_robot_cfg
from legged_lab.assets.gl165.model import ARM_JOINT_NAMES, JOINT_NAMES, POLICY_JOINT_NAMES
from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.elf3_tang.wbc.ik_arm_motion import IkArmMotion
from .arm_motion import ContinuousIkArmMotion
from .gait import AlternatingGait
from .rewards import foot_heading_error, standing_mask, zero_velocity_command


class Gl165Env(BaseEnv):
    def __init__(self, cfg, headless):
        self.cfg, self.headless, self.device = cfg, headless, cfg.device
        self.physics_dt = cfg.sim.dt
        self.step_dt = cfg.sim.dt * cfg.sim.decimation
        self.num_envs = cfg.scene.num_envs
        self.seed(getattr(cfg.scene, "seed", 42))
        if abs(self.step_dt - .02) > 1e-8:
            raise ValueError("GL-165 IK replay requires 50 Hz control")
        bank_manifest = json.loads((Path(cfg.ik_bank_path) / "manifest.json").read_text())
        model = json.loads((Path(cfg.asset_path) / "model.json").read_text())
        if bank_manifest.get('robot') != 'gl165' or bank_manifest['training_urdf_sha256'] != model['training_urdf_sha256']:
            raise ValueError("GL-165 IK bank does not match the robot URDF")
        cfg.scene.robot = make_robot_cfg(cfg.asset_path, joint_gains=cfg.checkpoint_joint_gains)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            device=self.device, dt=self.physics_dt, render_interval=cfg.sim.decimation,
            physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count)))
        scene_cfg = InteractiveSceneCfg(num_envs=self.num_envs, env_spacing=cfg.scene.env_spacing)
        scene_cfg.robot = cfg.scene.robot
        # Procedural local ground and lights avoid a dependency on Nucleus assets.
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/ground", collision_group=-1,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.05)),
            spawn=sim_utils.CuboidCfg(
                size=(1000., 1000., .1), collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1.),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.18, .2, .23))))
        scene_cfg.light = AssetBaseCfg(prim_path="/World/light",
                                      spawn=sim_utils.DomeLightCfg(intensity=1500.))
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True,
            update_period=self.physics_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.robot = self.scene['robot']
        self.contact_sensor = self.scene.sensors['contact_sensor']
        self.command_generator = UniformVelocityCommand(UniformVelocityCommandCfg(
            asset_name="robot", ranges=cfg.commands.ranges,
            resampling_time_range=cfg.commands.resampling_time_range,
            rel_standing_envs=cfg.commands.rel_standing_envs, rel_heading_envs=0.,
            heading_command=False, debug_vis=cfg.commands.debug_vis), self)
        self.reward_manager = RewardManager(cfg.reward, self)
        self.init_buffers()
        self.event_manager = EventManager(cfg.domain_rand.events, self)
        if 'startup' in self.event_manager.available_modes:
            self.event_manager.apply(mode='startup')
        self.reset(torch.arange(self.num_envs, device=self.device))

    def init_buffers(self):
        if set(self.robot.joint_names) != set(JOINT_NAMES) or self.robot.is_fixed_base:
            raise ValueError("Expected a floating GL-165 with 31 movable joints")
        self.policy_joint_ids, _ = self.robot.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
        self.arm_joint_ids, _ = self.robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        self.wrist_body_ids, _ = self.robot.find_bodies(['lwrist_y_link', 'rwrist_y_link'], preserve_order=True)
        torso_ids, _ = self.robot.find_bodies('body_y_link')
        self.torso_body_id = torso_ids[0]
        self.knee_joint_ids, _ = self.robot.find_joints(['lknee_y_joint', 'rknee_y_joint'], preserve_order=True)
        self.observed_joint_ids = self.policy_joint_ids + self.arm_joint_ids
        self.num_actions = len(self.policy_joint_ids)
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.step_dt))
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        self.action_scale = torch.as_tensor(self.cfg.robot.action_scale, dtype=torch.float32, device=self.device)
        if self.action_scale.ndim == 0:
            self.action_scale = self.action_scale.expand(self.num_actions).clone()
        if (self.action_scale.shape != (self.num_actions,) or not torch.isfinite(self.action_scale).all()
                or (self.action_scale <= 0).any()):
            raise ValueError('GL-165 action_scale must be positive and follow the 15 named policy joints')
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise
        self.extras = {}
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_buf = self.time_out_buf.clone()
        self.sim_step_counter = 0
        self.action_buffer = DelayBuffer(2, self.num_envs, device=self.device)
        self.action_buffer.compute(torch.zeros(self.num_envs, self.num_actions, device=self.device))
        self.termination_contact_cfg = SceneEntityCfg(
            'contact_sensor', body_names=self.cfg.robot.terminate_contacts_body_names)
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg('contact_sensor', body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)
        self.standing_anchor = self.robot.data.root_pos_w[:, :2].clone()
        self.was_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        manifest = json.loads((Path(self.cfg.ik_bank_path) / 'manifest.json').read_text())
        motion_class = (ContinuousIkArmMotion if manifest['recipe'].get('motion_mode') == 'continuous_targets_v1'
                        else IkArmMotion)
        self.arm_motion = motion_class(self.cfg.ik_bank_path, self.cfg.ik_split, ARM_JOINT_NAMES,
                                       self.robot.data.default_joint_pos[:, self.arm_joint_ids], self.step_dt)
        self.arm_target = self.arm_motion.defaults.clone()
        self.endpoint_count = torch.zeros((), device=self.device, dtype=torch.long)
        self.endpoint_error_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self.endpoint_within_5cm = torch.zeros((), device=self.device, dtype=torch.long)
        self.gait = AlternatingGait(self.cfg.gait, self.num_envs, self.device, self.step_dt)
        self.last_leading_foot = torch.zeros(self.num_envs, device=self.device)
        self.lead_change = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.init_obs_buffer()

    def init_obs_buffer(self):
        actor, _ = self.compute_current_observations()
        self.noise_scale_vec = torch.zeros_like(actor[0])
        noise, scale = self.cfg.noise.noise_scales, self.obs_scales
        n = len(self.observed_joint_ids)
        self.noise_scale_vec[:3] = noise.ang_vel * scale.ang_vel
        self.noise_scale_vec[3:6] = noise.projected_gravity * scale.projected_gravity
        self.noise_scale_vec[9:9+n] = noise.joint_pos * scale.joint_pos
        self.noise_scale_vec[9+n:9+2*n] = noise.joint_vel * scale.joint_vel
        self.actor_obs_buffer = CircularBuffer(self.cfg.robot.actor_obs_history_length, self.num_envs, self.device)
        self.critic_obs_buffer = CircularBuffer(self.cfg.robot.critic_obs_history_length, self.num_envs, self.device)

    def compute_current_observations(self):
        data, ids, scale = self.robot.data, self.observed_joint_ids, self.obs_scales
        actor = torch.cat((data.root_ang_vel_b * scale.ang_vel, data.projected_gravity_b,
                           self.command_generator.command * scale.commands,
                           (data.joint_pos[:, ids] - data.default_joint_pos[:, ids]) * scale.joint_pos,
                           data.joint_vel[:, ids] * scale.joint_vel,
                           self.action_buffer._circular_buffer.buffer[:, -1]), dim=-1)
        if self.cfg.gait.enabled:
            actor = torch.cat((actor, self.gait.observations()), dim=-1)
        contacts = self.contact_sensor.data.current_contact_time[:, self.feet_cfg.body_ids] > 0
        critic = torch.cat((actor, data.root_lin_vel_b * scale.lin_vel, contacts,
                            self.arm_target - data.default_joint_pos[:, self.arm_joint_ids]), dim=-1)
        return actor, critic

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset(env_ids)
        self.arm_motion.reset(env_ids, self.robot.data.joint_pos[env_ids][:, self.arm_joint_ids])
        self.arm_target[env_ids] = self.robot.data.joint_pos[env_ids][:, self.arm_joint_ids]
        self.standing_anchor[env_ids] = self.robot.data.root_pos_w[env_ids, :2]
        self.was_standing[env_ids] = standing_mask(self)[env_ids]
        if self.cfg.gait.enabled:
            self.gait.reset(env_ids, self.command_generator.command)
        self.last_leading_foot[env_ids] = 0.
        self.lead_change[env_ids] = False

    def step(self, actions):
        self.extras['log'] = {}
        actions = actions.to(self.device).clamp(-self.clip_actions, self.clip_actions)
        delayed = self.action_buffer.compute(actions)
        targets = self.robot.data.default_joint_pos.clone()
        limits = self.robot.data.soft_joint_pos_limits[:, self.policy_joint_ids]
        targets[:, self.policy_joint_ids] = (targets[:, self.policy_joint_ids] + self.action_scale * delayed).clamp(
            limits[..., 0], limits[..., 1])
        requested = self.arm_motion.step()
        # Numerical path interpolation and reset connections are also rate limited.
        max_delta = self.step_dt * self.arm_motion.manifest['recipe']['max_reference_velocity_rad_s']
        self.arm_target += (requested - self.arm_target).clamp(-max_delta, max_delta)
        arm_limits = self.robot.data.joint_pos_limits[:, self.arm_joint_ids]
        self.arm_target.clamp_(arm_limits[..., 0], arm_limits[..., 1])
        self.arm_motion.target.copy_(self.arm_target)
        targets[:, self.arm_joint_ids] = self.arm_target
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(targets)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        if not self.headless:
            self.sim.render()
        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if self.cfg.gait.enabled:
            self.gait.update(self.command_generator.command)
        standing = standing_mask(self)
        changed = standing & ~self.was_standing
        self.standing_anchor[changed] = self.robot.data.root_pos_w[changed, :2]
        self.was_standing.copy_(standing)
        if 'interval' in self.event_manager.available_modes:
            self.event_manager.apply(mode='interval', dt=self.step_dt)
        self.reset_buf, self.time_out_buf = self.check_reset()
        reward = self.reward_manager.compute(self.step_dt)
        # Measure the physical state before resets replace a failed pose.
        posture_log = self.posture_metrics()
        self.reset(self.reset_buf.nonzero().flatten())
        self.extras['time_outs'] = self.time_out_buf
        self.extras.setdefault('log', {}).update(posture_log)
        self.extras.setdefault('log', {}).update(self.arm_motion.metrics)
        self.extras['log']['IK/joint_tracking_error_rad'] = (
            self.robot.data.joint_pos[:, self.arm_joint_ids] - self.arm_target).abs().mean()
        torso_p = self.robot.data.body_pos_w[:, self.torso_body_id]
        torso_q = self.robot.data.body_quat_w[:, self.torso_body_id]
        measured = quat_apply_inverse(
            torso_q[:, None, :].expand(-1, 2, -1),
            self.robot.data.body_pos_w[:, self.wrist_body_ids] - torso_p[:, None, :])
        errors = (measured - self.arm_motion.goal_positions[self.arm_motion.index]).norm(dim=-1).amax(-1)
        holding = (self.arm_motion.phase == 2) & ~self.arm_motion.neutral
        count = holding.sum().clamp(min=1)
        if isinstance(self.arm_motion, ContinuousIkArmMotion):
            # Endpoints last only one frame. Count arrivals across frames so a
            # frame without arrivals does not log zero error / zero success.
            self.endpoint_count += holding.sum()
            self.endpoint_error_sum += (errors * holding).sum()
            self.endpoint_within_5cm += ((errors < .05) & holding).sum()
            count = self.endpoint_count.clamp(min=1)
            self.extras['log']['IK/endpoint_count'] = self.endpoint_count.clone()
            self.extras['log']['IK/endpoint_wrist_position_error_m'] = self.endpoint_error_sum / count
            self.extras['log']['IK/endpoint_within_5cm_fraction'] = self.endpoint_within_5cm / count
        else:
            self.extras['log']['IK/hold_wrist_position_error_m'] = (errors * holding).sum() / count
            self.extras['log']['IK/hold_within_5cm_fraction'] = ((errors < .05) & holding).sum() / count
        actor, critic = self.compute_observations()
        self.extras['observations'] = {'critic': critic}
        return actor, reward, self.reset_buf, self.extras

    def posture_metrics(self):
        data = self.robot.data
        gravity = quat_apply_inverse(data.body_quat_w[:, self.torso_body_id], data.GRAVITY_VEC_W)
        tilt = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1., 1.)))
        waist = torch.rad2deg(data.joint_pos[:, self.policy_joint_ids[:3]]).abs()
        knees = torch.rad2deg(data.joint_pos[:, self.knee_joint_ids])
        height = data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        standing = standing_mask(self).float()
        count = standing.sum().clamp_min(1.)
        load_cfg = self.reward_manager.get_term_cfg('stand_foot_load').params
        foot_load = self.contact_sensor.data.net_forces_w[:, load_cfg['sensor_cfg'].body_ids, 2].clamp_min(0.)
        zero_command = zero_velocity_command(self, load_cfg['command_tolerance']).float()
        zero_count = zero_command.sum().clamp_min(1.)
        loaded = (foot_load >= load_cfg['threshold']).all(-1).float()
        spacing_cfg = self.reward_manager.get_term_cfg('moving_feet_distance').params
        feet = data.body_pos_w[:, spacing_cfg['asset_cfg'].body_ids]
        separation = feet[:, 0] - feet[:, 1]
        distance = torch.linalg.vector_norm(separation[:, :2], dim=-1)
        local_separation = quat_apply_inverse(yaw_quat(data.root_quat_w), separation)
        signed_width = local_separation[:, 1]
        lateral = signed_width.abs()
        lead = torch.where(local_separation[:, 0].abs() > .03, local_separation[:, 0].sign(), 0.)
        self.lead_change = (lead != 0.) & (self.last_leading_foot != 0.) & (lead != self.last_leading_foot)
        self.last_leading_foot = torch.where(lead != 0., lead, self.last_leading_foot)
        moving = (~zero_velocity_command(self, spacing_cfg['command_tolerance'])).float()
        moving_count = moving.sum().clamp_min(1.)
        too_wide = (distance > spacing_cfg['max_distance']).float()
        heading_cfg = self.reward_manager.get_term_cfg('feet_heading').params
        heading = foot_heading_error(data, heading_cfg['asset_cfg'].body_ids)
        heading_deg = torch.rad2deg(heading)
        hip_cfg = self.reward_manager.get_term_cfg('hip_yaw_range').params
        hip_ids = hip_cfg['asset_cfg'].joint_ids
        hip_delta_deg = torch.rad2deg((data.joint_pos[:, hip_ids] - data.default_joint_pos[:, hip_ids]).abs())
        metrics = {
            'Posture/torso_tilt_deg': tilt.mean(),
            'Posture/torso_above_20deg_fraction': (tilt > 20.).float().mean(),
            'Posture/waist_yaw_abs_deg': waist[:, 0].mean(),
            'Posture/waist_roll_abs_deg': waist[:, 1].mean(),
            'Posture/waist_pitch_abs_deg': waist[:, 2].mean(),
            'Posture/knee_bend_deg': knees.mean(),
            'Posture/base_height_m': height.mean(),
            'Posture/standing_fraction': standing.mean(),
            'Posture/standing_base_height_m': (height * standing).sum() / count,
            'Posture/standing_torso_tilt_deg': (tilt * standing).sum() / count,
            'FootLoad/zero_command_fraction': zero_command.mean(),
            'FootLoad/left_mean_N': (foot_load[:, 0] * zero_command).sum() / zero_count,
            'FootLoad/right_mean_N': (foot_load[:, 1] * zero_command).sum() / zero_count,
            'FootLoad/both_above_threshold_fraction': (loaded * zero_command).sum() / zero_count,
            'FootSpacing/moving_fraction': moving.mean(),
            'FootSpacing/moving_distance_m': (distance * moving).sum() / moving_count,
            'FootSpacing/moving_lateral_width_m': (lateral * moving).sum() / moving_count,
            'FootSpacing/moving_signed_lateral_width_m': (signed_width * moving).sum() / moving_count,
            'FootSpacing/moving_crossed_fraction': ((signed_width < 0.) * moving).sum() / moving_count,
            'FootSpacing/standing_fore_aft_abs_m': (local_separation[:, 0].abs() * standing).sum() / count,
            'FootSpacing/moving_above_limit_fraction': (too_wide * moving).sum() / moving_count,
            'FootSpacing/moving_excess_m': ((distance - spacing_cfg['max_distance']).clamp_min(0.) * moving).sum() / moving_count,
            'FootHeading/left_abs_deg': heading_deg[:, 0].mean(),
            'FootHeading/right_abs_deg': heading_deg[:, 1].mean(),
            'FootHeading/moving_left_abs_deg': (heading_deg[:, 0] * moving).sum() / moving_count,
            'FootHeading/moving_right_abs_deg': (heading_deg[:, 1] * moving).sum() / moving_count,
            'FootHeading/moving_either_above_30deg_fraction': ((heading_deg > 30.).any(-1).float() * moving).sum() / moving_count,
            'FootHeading/hip_yaw_default_delta_deg': hip_delta_deg.mean(),
        }
        if self.cfg.gait.enabled:
            walking = (self.gait.activity > .99).float()
            fore_aft = walking * (self.gait.fore_aft_gate > .99)
            gait_count, fore_aft_count = walking.sum().clamp_min(1.), fore_aft.sum().clamp_min(1.)
            settled = self.gait.standing_alignment * standing
            metrics.update({
                'Gait/activity': self.gait.activity.mean(),
                'Gait/fore_aft_tracking_error_m': ((local_separation[:, 0] - self.gait.fore_aft).abs()
                                                  * fore_aft).sum() / fore_aft_count,
                'Gait/fore_aft_left_leads_fraction': ((lead > 0.) * fore_aft).sum() / fore_aft_count,
                'Gait/fore_aft_right_leads_fraction': ((lead < 0.) * fore_aft).sum() / fore_aft_count,
                'Gait/fore_aft_lead_changes_per_s': (self.lead_change * fore_aft).sum()
                                                   / fore_aft_count / self.step_dt,
                'Gait/contact_mismatch': (self.reward_manager.get_term_cfg('gait_contact').func(
                    self, **self.reward_manager.get_term_cfg('gait_contact').params) * walking).sum() / gait_count,
                'Gait/settled_standing_fore_aft_abs_m': (local_separation[:, 0].abs() * settled).sum()
                                                       / settled.sum().clamp_min(1.),
            })
        return metrics

    def check_reset(self):
        terminated, time_out = super().check_reset()
        terminated |= self.robot.data.root_pos_w[:, 2] < self.cfg.min_root_height
        terminated |= self.robot.data.projected_gravity_b[:, 2] > -.5
        return terminated, time_out
