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

import os
from pathlib import Path
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
import isaaclab.utils.math as math_utils

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # Isaac Lab 2.1 compatibility.
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from legged_lab.utils.joint_layout import resolve_action_scales, resolve_joint_layout
from rsl_rl.env import VecEnv

from legged_lab.envs.base.base_env_config import BaseEnvCfg
from legged_lab.mdp.commands import (
    CommandAxisCurriculumCfg as _CommandAxisCurriculumCfg,
    CommandCurriculumCfg as _CommandCurriculumCfg,
    UniformVelocityBodyCommand,
    UniformVelocityBodyCommandCfg,
)
from legged_lab.utils.env_utils.scene import SceneCfg
from legged_lab.utils.amass import sample_amass_arm_poses
from legged_lab.utils.elf3_arm_motion import Elf3ArmMotion

from isaaclab.assets import RigidObject


class BaseEnv(VecEnv):
    def __init__(self, cfg: BaseEnvCfg, headless):
        self.cfg: BaseEnvCfg

        self.cfg = cfg
        self.training_diagnostics = None
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        self.sequence_length = 1000
        if cfg.robot.arm_motion_source not in ("default", "amass", "amass_elf3"):
            raise ValueError("robot.arm_motion_source must be 'default', 'amass' or 'amass_elf3'")
        self.use_amass = cfg.robot.arm_motion_source in ("amass", "amass_elf3")
        self.elf3_arm_motion = None
        if self.use_amass and len(cfg.robot.arm_joint_names) != 14:
            raise ValueError("The G1 AMASS arm clips require exactly 14 configured arm joints")
        self.amass = None
        if cfg.robot.arm_motion_source == "amass_elf3":
            # Filled by the initial reset, once per env (avoid loading twice).
            self.amass = torch.empty(self.num_envs, self.sequence_length, 14, device=self.device)
        elif self.use_amass:
            self.amass = sample_amass_arm_poses(
                num_envs=self.num_envs,
                sequence_length=self.sequence_length,
                device=self.device,
                dataset_path=str(Path(__file__).resolve().parents[3]),
                interpolation_factor=1,
            )
        self.amass_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.arm_targets = None
        self.dgp_flag = True

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.action_joint_names = list(cfg.robot.action_joint_names)
        self.num_actions = len(self.action_joint_names)
        self.num_arm_joints = len(cfg.robot.arm_joint_names)
        self.custom_joint_ids, self.inverse_joint_ids = resolve_joint_layout(
            self.robot.data.joint_names, self.action_joint_names, cfg.robot.arm_joint_names
        )
        if cfg.robot.arm_motion_source == "amass_elf3":
            if cfg.robot.arm_motion_timing not in ("legacy_frames", "source_fps"):
                raise ValueError("arm_motion_timing must be 'legacy_frames' or 'source_fps'")
            arm_ids = self.custom_joint_ids[self.num_actions:self.num_actions + self.num_arm_joints]
            self.elf3_arm_motion = Elf3ArmMotion(
                cfg.robot.arm_joint_names,
                self.robot.data.default_joint_pos[:, arm_ids],
                self.robot.data.soft_joint_pos_limits[:, arm_ids],
                self.step_dt,
                scale=cfg.robot.arm_motion_scale,
                ramp_s=cfg.robot.arm_motion_ramp_s,
                max_velocity=cfg.robot.arm_motion_max_velocity,
            )
            target_rate = (
                "disabled" if cfg.robot.arm_motion_max_velocity is None
                else f"{cfg.robot.arm_motion_max_velocity} rad/s"
            )
            print(
                f"[ArmMotion] AMASS -> ELF3: 14 joints, scale={cfg.robot.arm_motion_scale}, "
                f"ramp={cfg.robot.arm_motion_ramp_s}s, target rate limit={target_rate}, "
                f"control={1 / self.step_dt:g} Hz, timing={cfg.robot.arm_motion_timing}"
            )
        self.base_body_id = None
        if cfg.robot.base_body_name is not None:
            self.base_body_id = self.robot.data.body_names.index(cfg.robot.base_body_name)

        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        curriculum_cfg = self.cfg.commands.curriculum
        # Mirror the env-level per-axis curriculum config onto the command term cfg.
        # We keep the conversion explicit so the command module stays self-contained
        # and doesn't depend on the env config classes.
        def _mirror_axis(axis):
            if axis is None:
                return None
            # ``--set`` can replace an optional axis (whose default is None)
            # with a literal dictionary. Accept both that representation and
            # the strongly typed config used by task defaults.
            def _get(name):
                return axis[name] if isinstance(axis, dict) else getattr(axis, name)

            return _CommandAxisCurriculumCfg(
                start_range=tuple(_get("start_range")),
                end_range=tuple(_get("end_range")),
                start_iter=int(_get("start_iter")),
                end_iter=int(_get("end_iter")),
            )

        command_curriculum_cfg = _CommandCurriculumCfg(
            enable=curriculum_cfg.enable,
            lin_vel_x=_mirror_axis(curriculum_cfg.lin_vel_x),
            lin_vel_y=_mirror_axis(curriculum_cfg.lin_vel_y),
            ang_vel_z=_mirror_axis(curriculum_cfg.ang_vel_z),
            body_height=_mirror_axis(curriculum_cfg.body_height),
            body_roll=_mirror_axis(curriculum_cfg.body_roll),
            body_pitch=_mirror_axis(curriculum_cfg.body_pitch),
            body_yaw=_mirror_axis(curriculum_cfg.body_yaw),
        )
        command_cfg = UniformVelocityBodyCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            rel_zero_vel_yaw_envs=self.cfg.commands.rel_zero_vel_yaw_envs,
            rel_in_place_turn_envs=self.cfg.commands.rel_in_place_turn_envs,
            in_place_small_turn_fraction=self.cfg.commands.in_place_small_turn_fraction,
            rel_zero_pose_envs=self.cfg.commands.rel_zero_pose_envs,
            rel_single_axis_pose_envs=self.cfg.commands.rel_single_axis_pose_envs,
            rel_nominal_height_envs=self.cfg.commands.rel_nominal_height_envs,
            rel_high_stance_turn_envs=self.cfg.commands.rel_high_stance_turn_envs,
            rel_high_stance_move_turn_envs=self.cfg.commands.rel_high_stance_move_turn_envs,
            rel_low_stance_move_envs=self.cfg.commands.rel_low_stance_move_envs,
            low_stance_height_range=self.cfg.commands.low_stance_height_range,
            low_stance_speed_range=self.cfg.commands.low_stance_speed_range,
            low_stance_yaw_rate_range=self.cfg.commands.low_stance_yaw_rate_range,
            low_stance_turn_fraction=self.cfg.commands.low_stance_turn_fraction,
            high_stance_height_range=self.cfg.commands.high_stance_height_range,
            high_stance_speed_range=self.cfg.commands.high_stance_speed_range,
            high_stance_yaw_rate_range=self.cfg.commands.high_stance_yaw_rate_range,
            nominal_body_height=self.cfg.robot.nominal_height,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            body_command=True,
            ranges=self.cfg.commands.ranges,
            curriculum=command_curriculum_cfg,
        )
        self.command_generator = UniformVelocityBodyCommand(cfg=command_cfg, env=self)
        # Cache the iteration-conversion constants so the per-step curriculum update is cheap.
        self._curriculum_num_steps_per_env = max(int(curriculum_cfg.num_steps_per_env), 1)
        self._curriculum_decimation = max(int(self.cfg.sim.decimation), 1)
        # Iteration offset for the curriculum schedule. Stays 0 for fresh runs; on
        # resume it is set to the checkpoint's iteration so the schedule continues
        # rather than restarting (the step counter below restarts at 0 each process).
        self._curriculum_iteration_offset = 0.0
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        
        # ========== Cache body masses after startup randomization ==========
        # This avoids expensive PhysX API calls in reward functions (e.g., center_of_gravity_tracking)
        # Since mass randomization uses mode="startup", masses are set once and never change
        self._cached_body_masses = self.robot.root_physx_view.get_masses().to(device=self.device)
        
        
        self.reset(env_ids)
        self.custom_default_joint_pos = self.robot.data.default_joint_pos[:,self.custom_joint_ids]

    @property
    def base_quat_w(self):
        if self.base_body_id is None:
            return self.robot.data.root_quat_w
        return self.robot.data.body_quat_w[:, self.base_body_id]

    @property
    def base_lin_vel_w(self):
        if self.base_body_id is None:
            return self.robot.data.root_lin_vel_w
        return self.robot.data.body_lin_vel_w[:, self.base_body_id]

    @property
    def base_ang_vel_w(self):
        if self.base_body_id is None:
            return self.robot.data.root_ang_vel_w
        return self.robot.data.body_ang_vel_w[:, self.base_body_id]

    @property
    def base_lin_vel_b(self):
        if self.base_body_id is None:
            return self.robot.data.root_lin_vel_b
        return quat_apply_inverse(self.base_quat_w, self.base_lin_vel_w)

    @property
    def base_ang_vel_b(self):
        if self.base_body_id is None:
            return self.robot.data.root_ang_vel_b
        return quat_apply_inverse(self.base_quat_w, self.base_ang_vel_w)

    @property
    def base_projected_gravity_b(self):
        if self.base_body_id is None:
            return self.robot.data.projected_gravity_b
        return quat_apply_inverse(self.base_quat_w, self.robot.data.GRAVITY_VEC_W)

    @property
    def base_heading_w(self):
        forward = math_utils.quat_apply(self.base_quat_w, self.robot.data.FORWARD_VEC_B)
        return torch.atan2(forward[:, 1], forward[:, 0])

    def set_curriculum_start_iteration(self, iteration: float) -> None:
        """Offset the command curriculum so it continues from ``iteration``.

        Called on resume: the per-step iteration estimate is derived from a step
        counter that restarts at 0 each process, so without this offset the
        curriculum would restart from iteration 0 even though training resumes
        from a later checkpoint iteration.
        """
        self._curriculum_iteration_offset = float(iteration)

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.clip_actions = self.cfg.normalization.clip_actions
        # print(f"Action clip value: {self.clip_actions}")
        self.clip_obs = self.cfg.normalization.clip_observations
        # print(f"Observation clip value: {self.clip_obs}")

        if isinstance(self.cfg.robot.action_scale, dict):
            self.action_scale = torch.tensor(
                resolve_action_scales(self.action_joint_names, self.cfg.robot.action_scale),
                device=self.device, dtype=torch.float32,
            )
        else:
            self.action_scale = self.cfg.robot.action_scale

        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.init_obs_buffer()

    def compute_current_observations(self):
        robot = self.robot
        custom_default = robot.data.default_joint_pos[:,self.custom_joint_ids]
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = self.base_ang_vel_b
        projected_gravity = self.base_projected_gravity_b
        command = self.command_generator.command
        custom_pos = robot.data.joint_pos[:,self.custom_joint_ids]
        custom_vel = robot.data.joint_vel[:,self.custom_joint_ids]
        joint_pos = custom_pos[:, :self.num_actions] - custom_default[:, :self.num_actions]
        joint_vel = custom_vel[:, :self.num_actions]
        action = self.action_buffer._circular_buffer.buffer[:, -1, :self.num_actions]
        
        # Compute feet contact (now shared by both actor and critic)
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        
        # Actor now includes feet_contact
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                feet_contact.float(),  # Add feet_contact to actor (2 dims)
            ],
            dim=-1,
        )

        root_lin_vel = self.base_lin_vel_b
        # Critic still has additional privileged information
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        # Only resample sequences for environments that need reset. Skipped entirely
        # when the arms are externally driven: the clip would go unread, and each
        # resample re-reads 1000 frames off disk, which is slow enough to visibly
        # stall a run if something is resetting often.
        if len(env_ids) > 0 and self.use_amass and self.arm_targets is None:
            # Get project root directory (3 levels up from this file)
            project_root = Path(__file__).resolve().parents[3]

            new_sequences = sample_amass_arm_poses(
                num_envs=len(env_ids),
                sequence_length=self.sequence_length,
                device=self.device,
                dataset_path=str(project_root),
                interpolation_factor=1,
                control_dt=(
                    self.step_dt
                    if self.elf3_arm_motion is not None and self.cfg.robot.arm_motion_timing == "source_fps"
                    else None
                ),
            )
            # Update only the environments that need reset
            self.amass[env_ids] = new_sequences
            # Reset indices for these environments
            self.amass_index[env_ids] = 0
        
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        if self.elf3_arm_motion is not None:
            arm_ids = self.custom_joint_ids[self.num_actions:self.num_actions + self.num_arm_joints]
            self.elf3_arm_motion.reset(env_ids, self.robot.data.joint_pos[env_ids][:, arm_ids])

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):

        # print(f"Actions before clipping: {actions.shape}")
        delayed_actions = self.action_buffer.compute(actions)

        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        processed_actions = self.custom_default_joint_pos.clone()

        processed_actions[:, :self.num_actions] += cliped_actions * self.action_scale

        if self.arm_targets is not None:
            # Externally authored arms (demo choreography). Set to None to hand the
            # arms back to AMASS; resets do not clear it, unlike writing to self.amass.
            processed_actions[:, self.num_actions:self.num_actions + self.num_arm_joints] = self.arm_targets
        elif self.use_amass:
            # Use per-environment indices with advanced indexing
            # Wrap around when reaching sequence end
            self.amass_index = torch.where(
                self.amass_index >= self.sequence_length,
                torch.zeros_like(self.amass_index),
                self.amass_index
            )

            # Gather arm poses using per-environment indices
            # amass shape: (num_envs, sequence_length, 14)
            # amass_index shape: (num_envs,)
            env_indices = torch.arange(self.num_envs, device=self.device)
            arm_pose = self.amass[env_indices, self.amass_index]
            if self.elf3_arm_motion is not None:
                arm_pose = self.elf3_arm_motion.step(arm_pose)
            processed_actions[:, self.num_actions:self.num_actions + self.num_arm_joints] = arm_pose

            # Increment indices for next step
            self.amass_index += 1
        
        processed_actions = processed_actions[:, self.inverse_joint_ids]
        # Match the original WBC controller: send position targets directly to PD.
        # USD joint limits and actuator effort limits still constrain execution.

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        # Update the command curriculum before resampling so the latest scale applies to
        # any environments that resample on this step.
        env_steps = self.sim_step_counter // self._curriculum_decimation
        iteration_estimate = self._curriculum_iteration_offset + env_steps / self._curriculum_num_steps_per_env
        self.command_generator.update_curriculum(iteration_estimate)
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            if self.training_diagnostics is not None:
                self.training_diagnostics.before_interval_events()
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        if self.training_diagnostics is not None:
            self.training_diagnostics.after_step_before_reset(reward_buf, self.reset_buf, self.time_out_buf)
        arm_logs = {}
        if self.elf3_arm_motion is not None:
            arm_ids = self.custom_joint_ids[self.num_actions:self.num_actions + self.num_arm_joints]
            arm_logs = {
                "ArmMotion/enabled": float(self.arm_targets is None),
                "ArmMotion/scale": self.elf3_arm_motion.scale,
                "ArmMotion/target_deviation_rad": (
                    self.robot.data.joint_pos_target[:, arm_ids] - self.elf3_arm_motion.defaults
                ).abs().mean(),
                "ArmMotion/actual_speed_rad_s": self.robot.data.joint_vel[:, arm_ids].abs().mean(),
                "ArmMotion/clipped_fraction": self.elf3_arm_motion.clipped_fraction,
            }
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        # RSL-RL retains these dictionaries until the rollout is logged. Create
        # a fresh snapshot each step, including steps without episode resets.
        self.extras["log"] = {
            **self.extras.get("log", {}),
            **self.command_generator.curriculum_log(),
            **arm_logs,
            **getattr(self, "action_target_logs", {}),
        }

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        # asset_cfg = SceneEntityCfg("robot", body_names="torso_link")
        # asset_cfg.resolve(self.scene)   
        # asset: Articulation = self.scene[asset_cfg.name]
        # quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
        # roll, pitch, yaw = quat_to_euler_xyz(quat_w)
        # print("roll, pitch, yaw:",roll, pitch, yaw)

        if self.dgp_flag:
            asset_cfg=SceneEntityCfg("robot", body_names=self.cfg.robot.terminate_contacts_body_names)
            asset: RigidObject = self.scene[asset_cfg.name]
            # print(asset.data.root_pos_w[:, 2] < 0.3)
            reset_buf = (asset.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]) < self.cfg.robot.termination_height
        else:
            net_contact_forces = self.contact_sensor.data.net_forces_w_history
            reset_buf = torch.any(
                torch.max(
                    torch.norm(
                        net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                > 1.0,
                dim=1,
            )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        # print(reset_buf)
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:13] = 0  # commands (7 dims, no noise)
            noise_vec[13 : 13 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[13 + self.num_actions : 13 + self.num_actions * 2] = (noise_scales.joint_vel * self.obs_scales.joint_vel)
            noise_vec[13 + self.num_actions * 2 : 13 + self.num_actions * 3] = 0.0  # actions (no noise)
            noise_vec[13 + self.num_actions * 3 : 13 + self.num_actions * 3 + 2] = 0.0  # feet_contact (no noise, binary signal)
            self.noise_scale_vec = noise_vec


            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

def quat_to_euler_xyz(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    q: [num_envs, 4]  (x, y, z, w)
    return: roll, pitch, yaw
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
