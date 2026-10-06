# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
# from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from isaaclab.utils.math import quat_apply, quat_conjugate
from scipy.spatial.transform import Rotation

# from legged_lab.envs.elf3.run_cfg import Elf3RunFlatEnvCfg
# from legged_lab.envs.elf3.run_with_sensor_cfg import Elf3RunWithSensorFlatEnvCfg
from legged_lab.envs.elf3_tang.walk_cfg import Elf3WalkTangFlatEnvCfg
from legged_lab.envs.elf3_tang.commands import TangCommands, rotate_inverse
from legged_lab.envs.elf3_tang.actions import policy_action_limits
from legged_lab.envs.elf3_tang import rewards as tang_rewards
from legged_lab.envs.elf3_tang.wbc.amass import AMassDatasetLoader
from legged_lab.envs.elf3_tang.wbc.elf3_arm_motion import Elf3ArmMotion
from legged_lab.envs.elf3_tang.arm_diagnostics import ArmMotionDiagnostics
# from legged_lab.envs.elf3.walk_with_sensor_cfg import (
#     Elf3WalkWithSensorFlatEnvCfg,
# )
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay


class Elf3Env(VecEnv):
    @property
    def base_quat_w(self):
        return self.robot.data.body_quat_w[:, self.pelvis_body_id]

    @property
    def base_lin_vel_w(self):
        return self.robot.data.body_lin_vel_w[:, self.pelvis_body_id]

    @property
    def base_ang_vel_w(self):
        return self.robot.data.body_ang_vel_w[:, self.pelvis_body_id]

    @property
    def base_lin_vel_b(self):
        return rotate_inverse(self.base_quat_w, self.base_lin_vel_w)

    @property
    def base_ang_vel_b(self):
        return rotate_inverse(self.base_quat_w, self.base_ang_vel_w)

    @property
    def base_projected_gravity_b(self):
        return rotate_inverse(self.base_quat_w, self.robot.data.GRAVITY_VEC_W)

    @property
    def arm_position_target(self):
        if self.arm_targets is not None:
            return self.arm_targets
        return self.amass_motion.target if self.use_amass else self.random_arm_target

    def __init__(
        self,
        # cfg: (
        #     Elf3RunFlatEnvCfg
        #     | Elf3WalkFlatEnvCfg
        #     | Elf3WalkWithSensorFlatEnvCfg
        #     | Elf3RunWithSensorFlatEnvCfg
        # ),
        cfg: (
            Elf3WalkTangFlatEnvCfg
        ),
        headless,
    ):
        # self.cfg: (
        #     Elf3RunFlatEnvCfg
        #     | Elf3WalkFlatEnvCfg
        #     | Elf3WalkWithSensorFlatEnvCfg
        #     | Elf3RunWithSensorFlatEnvCfg
        # )
        self.cfg: (
            Elf3WalkTangFlatEnvCfg
        )

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

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
        if headless:
            # Training needs local physics assets only, not remote display textures.
            scene_cfg.terrain.visual_material = None
            scene_cfg.sky_light = None
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        # self.robot: Articulation = self.scene["robot"]
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        # Instantiate LiDAR and Depth Camera Sensors if enabled
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        self.command_generator = TangCommands(cfg.commands, self.num_envs, self.device)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.pelvis_body_id = self.robot.data.body_names.index(self.cfg.robot.pelvis_body_name)
        self.torso_body_id = self.robot.data.body_names.index(self.cfg.robot.torso_body_name)
        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)
        
        ###########################################################################add joint and body ids
        self.waist_ids, _ = self.robot.find_joints(
            name_keys=[
                "waist_y_joint",
                "waist_x_joint",
                "waist_z_joint",
            ],
            preserve_order=True,
        )
        self.left_wrist_ids, _ = self.robot.find_joints(
            name_keys=[
                "l_wrist_x_joint",
                "l_wrist_y_joint",
                "l_wrist_z_joint",
            ],
            preserve_order=True,
        )
        self.right_wrist_ids, _ = self.robot.find_joints(
            name_keys=[
                "r_wrist_x_joint",
                "r_wrist_y_joint",
                "r_wrist_z_joint",
            ],
            preserve_order=True,
        )
        ###########################################################################
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["l_ankle_x_link", "r_ankle_x_link"], preserve_order=True
        )
        self.knee_body_ids, _ = self.robot.find_bodies(
            name_keys=["l_knee_y_link", "r_knee_y_link"], preserve_order=True
        )
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["l_elbow_y_link", "r_elbow_y_link"], preserve_order=True
        )
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "l_hip_y_joint",
                "l_hip_x_joint",
                "l_hip_z_joint",
                "l_knee_y_joint",
                "l_ankle_y_joint",
                "l_ankle_x_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "r_hip_y_joint",
                "r_hip_x_joint",
                "r_hip_z_joint",
                "r_knee_y_joint",
                "r_ankle_y_joint",
                "r_ankle_x_joint",
            ],
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "l_shoulder_y_joint",
                "l_shoulder_x_joint",
                "l_shoulder_z_joint",
                "l_elbow_y_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "r_shoulder_y_joint",
                "r_shoulder_x_joint",
                "r_shoulder_z_joint",
                "r_elbow_y_joint",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["l_ankle_y_joint", "r_ankle_y_joint", "l_ankle_x_joint", "r_ankle_x_joint"],
            preserve_order=True,
        )
#tang
        self.policy_joint_ids = self.waist_ids + self.left_leg_ids + self.right_leg_ids
        self.arm_joint_ids = self.left_arm_ids + self.left_wrist_ids + self.right_arm_ids + self.right_wrist_ids
        self.num_actions = len(self.policy_joint_ids)

        # 支持标量、15维策略关节列表，或29维全关节列表的action_scale。
        if isinstance(self.cfg.robot.action_scale, (list, tuple)):
            action_scale = torch.tensor(self.cfg.robot.action_scale, dtype=torch.float, device=self.device)
            if action_scale.numel() == self.robot.data.default_joint_pos.shape[1]:
                action_scale = action_scale[self.policy_joint_ids]
            elif action_scale.numel() != self.num_actions:
                raise ValueError(
                    f"action_scale length must be {self.num_actions} or "
                    f"{self.robot.data.default_joint_pos.shape[1]}, got {action_scale.numel()}."
                )
            self.action_scale = action_scale
        else:
            self.action_scale = torch.full(
                (self.num_actions,), float(self.cfg.robot.action_scale), dtype=torch.float, device=self.device
            )

        self.policy_target_limits = self.robot.data.soft_joint_pos_limits[:, self.policy_joint_ids].clone()
        self.policy_action_low, self.policy_action_high = policy_action_limits(
            self.robot.data.default_joint_pos[:, self.policy_joint_ids],
            self.policy_target_limits, self.action_scale, self.clip_actions,
        )
        self.raw_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.bounded_actions = torch.zeros_like(self.raw_actions)
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
# tang
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.learning_iteration = 0
        self.learning_iteration_dt = 24 * self.step_dt
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.left_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))
        self.random_arm_resample_interval = max(1, int(self.cfg.arm_motion.resample_interval_s / self.physics_dt))
        self.arm_stand_amplitudes = torch.tensor(
            self.cfg.arm_motion.stand_amplitudes, dtype=torch.float, device=self.device
        )
        self.arm_move_amplitudes = torch.tensor(
            self.cfg.arm_motion.move_amplitudes, dtype=torch.float, device=self.device
        )
        #arm tang
        self.random_arm_target = self.robot.data.default_joint_pos[:, self.arm_joint_ids].clone()
        self.random_arm_start_target = self.random_arm_target.clone()
        self.random_arm_goal_target = self.random_arm_target.clone()
        self.random_arm_transition_step = torch.full(
            (self.num_envs,), self.random_arm_resample_interval, dtype=torch.long, device=self.device
        )
        self._resample_random_arm_targets(torch.arange(self.num_envs, device=self.device))

        self.arm_targets = None  # Optional fixed targets for controlled evaluation.
        self.arm_diagnostics = ArmMotionDiagnostics(
            self.num_envs, len(self.arm_joint_ids), self.step_dt, self.device)
        source = getattr(self.cfg.arm_motion, "source", "random")
        if source not in ("random", "amass_elf3"):
            raise ValueError(f"Unknown arm motion source: {source}")
        self.use_amass = source == "amass_elf3"
        if self.use_amass:
            arm_cfg = self.cfg.arm_motion
            if arm_cfg.timing != "legacy_frames":
                raise ValueError("WBC synchronization requires legacy_frames timing")
            self.amass_loader = AMassDatasetLoader(arm_cfg.dataset_path, self.device, seed=self._seed)
            self.amass_source_sequence = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.amass_source_start = torch.zeros_like(self.amass_source_sequence)
            self.amass = torch.empty(
                self.num_envs, arm_cfg.sequence_length, 14, device=self.device
            )
            self.amass_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.amass_valid_length = torch.zeros_like(self.amass_index)
            if not 0 <= arm_cfg.amass_neutral_fraction <= 1:
                raise ValueError('amass_neutral_fraction must lie in [0,1]')
            self.amass_neutral_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.amass_motion = Elf3ArmMotion(
                [self.robot.data.joint_names[i] for i in self.arm_joint_ids],
                self.robot.data.default_joint_pos[:, self.arm_joint_ids],
                self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids],
                self.step_dt, scale=arm_cfg.scale, ramp_s=arm_cfg.ramp_s,
                max_velocity=arm_cfg.amass_max_velocity,
            )
            print(f"[ArmMotion] WBC AMASS -> ELF3: scale={arm_cfg.scale}, "
                  f"ramp={arm_cfg.ramp_s}s, max_velocity={arm_cfg.amass_max_velocity}, "
                  f"timing={arm_cfg.timing}, sequences={len(self.amass_loader.sequences)}")
            self.arm_guard = None
            if getattr(arm_cfg, 'collision_guard', False):
                from .wbc.arm_guard import ArmCollisionGuard

                self.arm_guard = ArmCollisionGuard(
                    [self.robot.data.joint_names[i] for i in self.arm_joint_ids],
                    self.robot.data.body_names, self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids],
                    self.device, margin=arm_cfg.guard_margin, max_velocity=arm_cfg.guard_max_velocity,
                    dt=self.step_dt)

        
        # # Init gait parameter
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        
        # self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)
        
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.init_obs_buffer()

    def visualize_motion(self, time):
        """
        根据给定时间的 AMP 运动捕捉数据更新机器人模拟状态。

        该函数设置关节位置和速度、根位置和方向，
        以及根据指定时间的 AMP 运动框架的线/角速度，
        然后逐步进行模拟并更新场景。

        参数：
            time（浮点数）：获取 AMP 运动帧的时间（以秒为单位）。

        返回：
            无
        """
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        device = self.device
        
        # # ====== 新增：关键调试信息 ======
        # print(f'[DEBUG] 运动数据总维度: {visual_motion_frame.shape}') # 应为70
        # print(f'[DEBUG] 机器人关节数: {self.robot.num_joints}') # 应为29

        # # 打印前35维（关节位置）和后35维（关节速度）的样例，了解数据范围
        # print(f'[DEBUG] 关节位置数据样例 (前10维): {visual_motion_frame[:10].cpu().numpy().round(3)}')
        # print(f'[DEBUG] 关节速度数据样例 (索引35-44): {visual_motion_frame[35:45].cpu().numpy().round(3)}')
        # # 打印最后几维，确认根信息
        # print(f'[DEBUG] 数据最后几维 (索引-12): {visual_motion_frame[-12:].cpu().numpy().round(3)}')
        # # ====== 调试结束 ======
        #数据输入

        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)#35
        dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)#35
        # root pos: 0-3
        # root euler: 3-6
        dof_pos[:, self.waist_ids] = visual_motion_frame[6:9]#
        dof_pos[:, self.left_leg_ids] = visual_motion_frame[9:15]#3+3+3+6
        dof_pos[:, self.right_leg_ids] = visual_motion_frame[15:21]#3+3+3+6+6
        dof_pos[:, self.left_arm_ids] = visual_motion_frame[21:25]#3+3+3+6+6+4
        dof_pos[:, self.left_wrist_ids] = visual_motion_frame[25:28]#3+3+3+6+6+4+3
        dof_pos[:, self.right_arm_ids] = visual_motion_frame[28:32]#3+3+3+6+6+4+3+4
        dof_pos[:, self.right_wrist_ids] = visual_motion_frame[32:35]#3+3+3+6+6+4+3+4+3
        #root_lin_vel: 35-38 
        #root_ang_vel: 38-41
        dof_vel[:, self.waist_ids] = visual_motion_frame[41:44]#35+3+3+3+6
        dof_vel[:, self.left_leg_ids] = visual_motion_frame[44:50]#35+3+3+3+6
        dof_vel[:, self.right_leg_ids] = visual_motion_frame[50:56]#35+3+3+6+6
        dof_vel[:, self.left_arm_ids] = visual_motion_frame[56:60]#35+3+3+6+6+4
        dof_vel[:, self.left_wrist_ids] = visual_motion_frame[60:63]#35+3+3+6+6+4+3
        dof_vel[:, self.right_arm_ids] = visual_motion_frame[63:67]#35+3+3+6+6+4+3+4
        dof_vel[:, self.right_wrist_ids] = visual_motion_frame[67:70]#35+3+3+6+6+4+3+4+3

        self.robot.write_joint_position_to_sim(dof_pos)#35
        self.robot.write_joint_velocity_to_sim(dof_vel)#35

        env_ids = torch.arange(self.num_envs, device=device)

        root_pos = visual_motion_frame[:3].clone()
        root_pos[2] += 0.3

        euler = visual_motion_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
        )

        lin_vel = visual_motion_frame[35:38].clone()
        ang_vel = torch.zeros_like(lin_vel)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)  # gmr转motion_data数据写入模拟器
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)

        self.waist_dof_pos =  dof_pos[:, self.waist_ids] 
        self.left_leg_dof_pos =  dof_pos[:, self.left_leg_ids] 
        self.right_leg_dof_pos = dof_pos[:, self.right_leg_ids]
        self.left_arm_dof_pos =  dof_pos[:, self.left_arm_ids +self.left_wrist_ids] 
        # self.left_wrist_dof_pos = dof_pos[:, self.left_wrist_ids]
        self.right_arm_dof_pos = dof_pos[:, self.right_arm_ids + self.right_wrist_ids]
        # self.right_wrist_dof_pos = dof_pos[:, self.right_wrist_ids]
        
        
        self.waist_dof_vel =  dof_vel[:, self.waist_ids]
        self.left_leg_dof_vel =  dof_vel[:, self.left_leg_ids] 
        self.right_leg_dof_vel = dof_vel[:, self.right_leg_ids]
        self.left_arm_dof_vel =  dof_vel[:, self.left_arm_ids + self.left_wrist_ids] 
        # self.left_wrist_dof_vel = dof_vel[:, self.left_wrist_ids]
        self.right_arm_dof_vel = dof_vel[:, self.right_arm_ids + self.right_wrist_ids]
        # self.right_wrist_dof_vel = dof_vel[:, self.right_wrist_ids]

        return torch.cat(
            (   
                
                self.right_arm_dof_pos,
                self.left_arm_dof_pos,
                self.waist_dof_pos,
                self.right_leg_dof_pos,
                self.left_leg_dof_pos,
                
                
                self.right_arm_dof_vel,
                self.left_arm_dof_vel,
                self.waist_dof_vel,
                self.right_leg_dof_vel,
                self.left_leg_dof_vel,
                
                
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )


    def compute_current_observations(self):
        data = self.robot.data
        pelvis_quat = data.body_quat_w[:, self.pelvis_body_id]
        ang_vel = rotate_inverse(pelvis_quat, data.body_ang_vel_w[:, self.pelvis_body_id])
        projected_gravity = rotate_inverse(pelvis_quat, data.GRAVITY_VEC_W)
        command = self.command_generator.full_command
        joint_pos = (data.joint_pos - data.default_joint_pos)[:, self.policy_joint_ids]
        joint_vel = (data.joint_vel - data.default_joint_vel)[:, self.policy_joint_ids]
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        current_actor_obs = torch.cat([
            ang_vel * self.obs_scales.ang_vel,
            projected_gravity * self.obs_scales.projected_gravity,
            command * self.obs_scales.commands,  # vx, vy, wz, pelvis height, relative torso yaw
            joint_pos * self.obs_scales.joint_pos,
            joint_vel * self.obs_scales.joint_vel,
            action * self.obs_scales.actions,
        ], dim=-1)  # 56 values; 10 history frames -> 560 actor inputs
        if getattr(self.cfg, 'observe_arm_state', False):
            # Actual arm encoders are available on the robot, unlike privileged root velocity.
            current_actor_obs = torch.cat((current_actor_obs,
                (data.joint_pos[:, self.arm_joint_ids] - data.default_joint_pos[:, self.arm_joint_ids]) * self.obs_scales.joint_pos,
                data.joint_vel[:, self.arm_joint_ids] * self.obs_scales.joint_vel), -1)
        net_forces = self.contact_sensor.data.net_forces_w_history
        contact = net_forces[:, :, self.feet_cfg.body_ids].norm(dim=-1).amax(dim=1) > 0.5
        lin_vel = rotate_inverse(pelvis_quat, data.body_lin_vel_w[:, self.pelvis_body_id])
        pose = torch.stack((tang_rewards.pelvis_height(self), tang_rewards.waist_yaw(self)), dim=-1)
        current_critic_obs = torch.cat([
            current_actor_obs, lin_vel * self.obs_scales.lin_vel, contact.float(), pose,
        ], dim=-1)  # 63 values; height is privileged, not an extra deployment sensor.
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

        if self.cfg.scene.depth_camera.enable_depth_camera:
            depth_image = self.depth_camera.data.output["distance_to_image_plane"]

            # (num_envs, height, width, 1) --> (num_envs, height * width)
            flattened_depth = depth_image.view(self.num_envs, -1)

            # Append the flattened depth data to the end of the actor and critic observation vectors.
            actor_obs = torch.cat([actor_obs, flattened_depth], dim=-1)
            critic_obs = torch.cat([critic_obs, flattened_depth], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self._update_reset_velocity_curriculum()
        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.raw_actions[env_ids] = 0.0
        self.arm_diagnostics.reset(env_ids)
        self.bounded_actions[env_ids] = 0.0
        self.random_arm_target[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self.arm_joint_ids]
        if self.use_amass:
            if self.arm_targets is None:
                self.amass[env_ids] = self.amass_loader.sample_arm_sequence(
                    len(env_ids), sequence_length=self.cfg.arm_motion.sequence_length,
                    interpolation_factor=1, control_dt=None,
                )
                self.amass_index[env_ids] = 0
                self.amass_valid_length[env_ids] = torch.as_tensor(
                    self.amass_loader.last_valid_lengths, device=self.device)
                self.amass_source_sequence[env_ids] = torch.as_tensor(
                    self.amass_loader.last_sequence_ids, device=self.device)
                self.amass_source_start[env_ids] = torch.as_tensor(
                    self.amass_loader.last_start_frames, device=self.device)
                if self.cfg.arm_motion.amass_neutral_fraction > 0:
                    self.amass_neutral_mask[env_ids] = (
                        torch.rand(len(env_ids), device=self.device) < self.cfg.arm_motion.amass_neutral_fraction)
            self.amass_motion.reset(env_ids, self.robot.data.joint_pos[env_ids][:, self.arm_joint_ids])
        else:
            self._resample_random_arm_targets(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def set_training_iteration(self, iteration, num_steps_per_env=24):
        """Restore all curriculum clocks together when loading a checkpoint."""
        self.learning_iteration = int(iteration)
        self.learning_iteration_dt = num_steps_per_env * self.step_dt
        self.command_generator.iteration = self.learning_iteration

    def _get_reset_velocity_curriculum_progress(self) -> float:
        duration_s = max(self.cfg.reset_velocity_curriculum.curriculum_duration_s, self.physics_dt)
        progress = min((self.learning_iteration * self.learning_iteration_dt) / duration_s, 1.0)
        initial = self.cfg.reset_velocity_curriculum.initial_scale
        final = self.cfg.reset_velocity_curriculum.final_scale
        return initial + (final - initial) * progress

    def _update_reset_velocity_curriculum(self):
        scale = self._get_reset_velocity_curriculum_progress()
        reset_base = self.event_manager.get_term_cfg("reset_base")
        velocity_range = reset_base.params["velocity_range"]
        for axis, target_range in self.cfg.reset_velocity_curriculum.velocity_ranges.items():
            velocity_range[axis] = (target_range[0] * scale, target_range[1] * scale)
        self.extras["log"]["Curriculum/reset_velocity_scale"] = scale

    def _get_push_velocity_curriculum_progress(self) -> float:
        duration_s = max(self.cfg.push_velocity_curriculum.curriculum_duration_s, self.physics_dt)
        progress = min((self.learning_iteration * self.learning_iteration_dt) / duration_s, 1.0)
        initial = self.cfg.push_velocity_curriculum.initial_scale
        final = self.cfg.push_velocity_curriculum.final_scale
        return initial + (final - initial) * progress

    def _update_push_velocity_curriculum(self):
        if "interval" not in self.event_manager.active_terms:
            return
        if "push_robot" not in self.event_manager.active_terms["interval"]:
            return
        scale = self._get_push_velocity_curriculum_progress()
        push_robot = self.event_manager.get_term_cfg("push_robot")
        velocity_range = push_robot.params["velocity_range"]
        for axis, target_range in self.cfg.push_velocity_curriculum.velocity_ranges.items():
            velocity_range[axis] = (target_range[0] * scale, target_range[1] * scale)
        self.extras["log"]["Curriculum/push_velocity_scale"] = scale

    def _resample_random_arm_targets(self, env_ids):
        default_pos = self.robot.data.default_joint_pos[env_ids][:, self.arm_joint_ids]
        amplitudes = self._get_arm_amplitudes(env_ids)
        target = default_pos + (2.0 * torch.rand_like(default_pos) - 1.0) * amplitudes
        limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_ids]
        self.random_arm_start_target[env_ids] = self.random_arm_target[env_ids]
        self.random_arm_goal_target[env_ids] = torch.clamp(target, limits[..., 0], limits[..., 1])
        self.random_arm_transition_step[env_ids] = 0

    def _get_arm_curriculum_progress(self) -> torch.Tensor:
        if getattr(self, "use_amass", False):
            cfg = self.cfg.arm_motion
            if cfg.amass_scale_end_iteration is None:
                scale = cfg.scale
            else:
                from .commands import curriculum_progress
                p = curriculum_progress(self.learning_iteration, cfg.amass_scale_start_iteration,
                                        cfg.amass_scale_end_iteration)
                scale = cfg.amass_initial_scale + p * (cfg.scale - cfg.amass_initial_scale)
            return torch.tensor(scale, dtype=torch.float, device=self.device)
        elapsed_iterations = self.learning_iteration - self.cfg.arm_motion.start_iteration
        if elapsed_iterations < 0:
            return torch.tensor(0.0, dtype=torch.float, device=self.device)
        elapsed_s = elapsed_iterations * self.learning_iteration_dt
        duration_s = max(self.cfg.arm_motion.curriculum_duration_s, self.physics_dt)
        progress = min(1.0, elapsed_s / duration_s)
        scale = self.cfg.arm_motion.initial_scale + (1.0 - self.cfg.arm_motion.initial_scale) * progress
        return torch.tensor(scale, dtype=torch.float, device=self.device)

    def _get_arm_amplitudes(self, env_ids) -> torch.Tensor:
        command = self.command_generator.command[env_ids]
        cmd_norm = torch.linalg.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2]) * 0.25
        stand_blend = 1.0 - (cmd_norm / self.cfg.arm_motion.command_threshold).clamp(0.0, 1.0)
        amplitudes = self.arm_move_amplitudes + (
            self.arm_stand_amplitudes - self.arm_move_amplitudes
        ) * stand_blend.unsqueeze(-1)
        return amplitudes * self._get_arm_curriculum_progress()

    def _maybe_resample_random_arm_targets(self):
        if self.sim_step_counter % self.random_arm_resample_interval == 0:
            self._resample_random_arm_targets(torch.arange(self.num_envs, device=self.device))

    def _update_random_arm_targets(self):
        progress = ((self.random_arm_transition_step + 1).float() / self.random_arm_resample_interval).clamp(0.0, 1.0)
        alpha = progress * progress * (3.0 - 2.0 * progress)
        alpha = alpha.unsqueeze(-1)
        desired = (1.0 - alpha) * self.random_arm_start_target + alpha * self.random_arm_goal_target
        max_delta = self.cfg.arm_motion.max_velocity * self.physics_dt
        self.random_arm_target += (desired - self.random_arm_target).clamp(-max_delta, max_delta)
        self.random_arm_transition_step = torch.clamp(
            self.random_arm_transition_step + 1, max=self.random_arm_resample_interval
        )

    def step(self, actions: torch.Tensor):
        self.extras["log"] = {}
        auxiliary_enabled = getattr(self.cfg, 'auxiliary_supervision', False)
        if auxiliary_enabled:
            from .auxiliary import capture_state, transition_targets
            auxiliary_before = capture_state(self)
        # PPO retains the original Gaussian sample and its log probability.
        # Never mutate that tensor. Only feasible actions enter delay/history
        # and action-rate rewards, preventing raw-output feedback amplification.
        self.raw_actions.copy_(actions.detach())
        self.bounded_actions = self.raw_actions.clamp(self.policy_action_low, self.policy_action_high)
        self.action = self.action_buffer.compute(self.bounded_actions)
        processed_actions = self.robot.data.default_joint_pos.clone()
        processed_actions[:, self.policy_joint_ids] = (
            self.action * self.action_scale
            + self.robot.data.default_joint_pos[:, self.policy_joint_ids]
        ).clamp(self.policy_target_limits[..., 0], self.policy_target_limits[..., 1])
        arm_reference = None
        arm_amass = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        arm_neutral = torch.zeros_like(arm_amass)
        arm_padded_tail = torch.zeros_like(arm_amass)
        if self.use_amass:
            previous_arm_target = self.amass_motion.target.clone() if self.arm_guard is not None else None
            if self.arm_targets is not None:
                arm_target = self.arm_targets
                arm_reference = arm_target.clone()
            else:
                self.amass_index.remainder_(self.cfg.arm_motion.sequence_length)
                poses = self.amass[torch.arange(self.num_envs, device=self.device), self.amass_index]
                self.amass_motion.scale = float(self._get_arm_curriculum_progress())
                if self.cfg.arm_motion.amass_neutral_fraction > 0:
                    arm_target = self.amass_motion.step(poses, neutral_mask=self.amass_neutral_mask)
                else:
                    arm_target = self.amass_motion.step(poses)
                arm_reference = self.amass_motion.requested_target.clone()
                arm_neutral = self.amass_neutral_mask.clone()
                arm_amass = ~arm_neutral
                arm_padded_tail = arm_amass & (self.amass_index >= self.amass_valid_length)
                self.amass_index += 1
            if self.arm_guard is not None and (self.arm_targets is None or self.cfg.arm_motion.guard_fixed_targets):
                arm_target = self.arm_guard.project(
                    arm_target, previous_arm_target, self.robot.data.joint_pos[:, self.arm_joint_ids],
                    self.robot.data.body_pos_w, self.robot.data.body_quat_w, self.torso_body_id)
                self.amass_motion.target.copy_(arm_target)
            processed_actions[:, self.arm_joint_ids] = arm_target
        elif self.arm_targets is not None:
            processed_actions[:, self.arm_joint_ids] = self.arm_targets
            arm_reference = self.arm_targets.clone()
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        if auxiliary_enabled:
            motor_ids = list(self.policy_joint_ids)+list(self.arm_joint_ids)
            # Actual clipped position targets, including external arm controls;
            # normalize radians by a fixed 0.5 rad, not by stochastic statistics.
            auxiliary_action = ((processed_actions[:, motor_ids]
                                 - self.robot.data.default_joint_pos[:, motor_ids])/.5).detach().clone()
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        for _ in range(self.cfg.sim.decimation):
            if not self.use_amass and self.arm_targets is None:
                self._maybe_resample_random_arm_targets()
                self._update_random_arm_targets()
                processed_actions[:, self.arm_joint_ids] = self.random_arm_target
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        # Capture final motor targets and post-physics feedback before resets or
        # command resampling. Diagnostics never alter observations/rewards/targets.
        if arm_reference is None:
            arm_reference = self.random_arm_goal_target.clone()
        arm_sample, arm_log = self.arm_diagnostics.update(
            arm_reference, processed_actions[:, self.arm_joint_ids],
            self.robot.data.joint_pos[:, self.arm_joint_ids],
            self.robot.data.joint_vel[:, self.arm_joint_ids],
            self.command_generator.is_standing_env, arm_amass, arm_neutral, arm_padded_tail)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self._calculate_gait_para()

        # Reward the command that generated this transition, then advance commands
        # for the next observation (a resample must not relabel the previous action).
        self.command_generator.iteration = self.learning_iteration
        self._update_push_velocity_curriculum()
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        if auxiliary_enabled:
            auxiliary_transition = transition_targets(auxiliary_before, capture_state(self),
                                                     auxiliary_action, self.reset_buf)
        reward_buf = self.reward_manager.compute(self.step_dt)
        transition_metrics = tang_rewards.transition_metrics(self)
        if getattr(self.cfg, "standing_diagnostics", False):
            from .standing import transition_metrics as standing_transition_metrics

            transition_metrics.update(standing_transition_metrics(self))
        self.command_generator.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        self.extras["log"]["Curriculum/arm_motion_scale"] = self._get_arm_curriculum_progress()
        if self.use_amass:
            self.extras["log"]["ArmMotion/clipped_fraction"] = self.amass_motion.clipped_fraction
            if self.arm_guard is not None and (self.arm_targets is None or self.cfg.arm_motion.guard_fixed_targets):
                self.extras["log"].update(self.arm_guard.metrics)
            if self.cfg.arm_motion.amass_neutral_fraction > 0:
                self.extras['log']['ArmMotion/neutral_reference_fraction'] = self.amass_neutral_mask.float().mean()
        self.extras["log"]["Curriculum/min_pelvis_height"] = self.command_generator.pose_ranges()[0][0]
        self.extras["log"]["Curriculum/max_waist_yaw"] = self.command_generator.pose_ranges()[1][1]
        self.extras["log"].update(transition_metrics)
        self.extras["log"].update(arm_log)
        self.extras['arm_motion'] = arm_sample
        if auxiliary_enabled:
            # Assign after reset(): per-step labels/logging must survive resets.
            self.extras['auxiliary'] = auxiliary_transition

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        forces = self.contact_sensor.data.net_forces_w_history[:, :, self.termination_contact_cfg.body_ids]
        reset_buf = (forces.norm(dim=-1).amax(dim=1) > self.cfg.robot.termination_contact_force).any(dim=-1)
        height = tang_rewards.pelvis_height(self)
        pelvis_quat = self.robot.data.body_quat_w[:, self.pelvis_body_id]
        gravity = rotate_inverse(pelvis_quat, self.robot.data.GRAVITY_VEC_W)
        reset_buf |= height < self.cfg.robot.termination_pelvis_height
        reset_buf |= -gravity[:, 2] < self.cfg.robot.termination_up_z
        # Walking across the plane is allowed; distance from reset is not a failure.
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales

            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel        #3   
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity     #3
            noise_vec[6:11] = 0  # five command channels, no observation noise
            noise_vec[11 : 11 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[11 + self.num_actions : 11 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[11 + self.num_actions * 2 : 11 + self.num_actions * 3] = 0.0
            if getattr(self.cfg, 'observe_arm_state', False):
                start = 11 + self.num_actions * 3
                noise_vec[start:start+14] = noise_scales.joint_pos * self.obs_scales.joint_pos
                noise_vec[start+14:start+28] = noise_scales.joint_vel * self.obs_scales.joint_vel
            
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
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_motion_mask(self, lin_threshold: float = 0.1, yaw_threshold: float = 0.1) -> torch.Tensor:
        """
        Returns a mask indicating whether AMP reward should be applied.
        Returns 1.0 for moving commands, 0.0 for near-zero commands.
        This allows disabling AMP's periodic motion encouragement when standing still.
        """
        cmd = self.command_generator.command
        is_moving = (torch.norm(cmd[:, :2], dim=-1) >= lin_threshold) | (torch.abs(cmd[:, 2]) >= yaw_threshold)
        return is_moving.float()

    def get_amp_obs_for_expert_trans(self):
        """Gets amp obs from policy"""
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        
        self.waist_dof_pos = self.robot.data.joint_pos[:, self.waist_ids]
        self.left_leg_dof_pos = self.robot.data.joint_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = self.robot.data.joint_pos[:, self.right_leg_ids]
        
        self.waist_dof_vel = self.robot.data.joint_vel[:, self.waist_ids]
        self.left_leg_dof_vel = self.robot.data.joint_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = self.robot.data.joint_vel[:, self.right_leg_ids]
        
        return torch.cat(
            (   
                self.waist_dof_pos,
                self.right_leg_dof_pos,
                self.left_leg_dof_pos,

                self.waist_dof_vel,
                self.right_leg_dof_vel,
                self.left_leg_dof_vel,
                
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )
        

    def seed(self, seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        self._seed = torch_utils.set_seed(seed)
        if hasattr(self, "amass_loader"):
            self.amass_loader.seed(self._seed)
        return self._seed

    def _calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on simulation time and offset.
        """
        t = self.episode_length_buf * self.step_dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0
