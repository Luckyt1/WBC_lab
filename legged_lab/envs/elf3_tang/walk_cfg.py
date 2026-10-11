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

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.elf3_lite import ELF3LITE_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.envs.elf3_tang.commands import TangCommandCfg
from legged_lab.envs.elf3_tang import rewards as tang_rewards
from legged_lab.envs.elf3_tang.wbc.reward_cfg import WBCRewardCfg
from pathlib import Path
from legged_lab.paths import amass_root


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.85


@configclass
class ArmMotionCurriculumCfg:
    # WBC playback: full amplitude from the first control step. The random
    # settings below remain available for reproducing the previous task.
    source: str = "amass_elf3"
    ik_bank_path: str | None = None
    ik_split: str = "train"
    ik_bank_sha256: str | None = None
    dataset_path: str = str(amass_root())
    sequence_length: int = 1000
    timing: str = "legacy_frames"
    scale: float = 1.0
    ramp_s: float = 0.0
    amass_max_velocity: float | None = None
    collision_guard: bool = False
    guard_margin: float = 0.012
    guard_max_velocity: float = 3.0
    guard_fixed_targets: bool = False  # opt-in; strict fixed-arm diagnostics stay explicit
    amass_neutral_fraction: float = 0.0
    amass_scale_start_iteration: int = 0
    amass_scale_end_iteration: int | None = None  # None retains legacy constant amplitude
    amass_initial_scale: float = 0.05
    # 手臂随机目标课程从第几个 PPO iteration 开始计时。
    start_iteration: int = 0

    # 课程持续的仿真 rollout 时间；期间手臂随机幅值从 initial_scale 线性增长到 1.0。
    curriculum_duration_s: float = 9600.0

    # 训练刚开始时使用的幅值比例，避免一开始手臂扰动太大导致站不稳。
    initial_scale: float = 0.02

    # 命令速度阈值；cmd_norm 小于该值时更偏向 stand_amplitudes，大于该值时更偏向 move_amplitudes。
    command_threshold: float = 0.6

    # 每隔多少秒重新采样一次手臂目标姿态，目标之间会平滑插值。
    # 远手臂动作如果切换太快，会通过躯干反作用力把腿部顶起来。
    resample_interval_s: float = 0.5
    max_velocity: float = 1.5  # rad/s limit for each externally driven arm joint

    # 零速度站立时手臂/手腕关节的随机幅值上限，单位 rad。
    # 顺序为：
    # l_shoulder_y, l_shoulder_x, l_shoulder_z, l_elbow_y, l_wrist_x, l_wrist_y, l_wrist_z,
    # r_shoulder_y, r_shoulder_x, r_shoulder_z, r_elbow_y, r_wrist_x, r_wrist_y, r_wrist_z。
    stand_amplitudes: list = [
        0.8,
        0.3,
        0.4,
        0.5,
        0.6,
        0.4,
        0.3,
        0.8,
        0.3,
        0.4,
        0.5,
        0.6,
        0.4,
        0.3,
    ]
    # stand_amplitudes: list = [
    #     0.18,
    #     0.10,
    #     0.18,
    #     0.10,
    #     0.06,
    #     0.06,
    #     0.06,
    #     0.18,
    #     0.10,
    #     0.18,
    #     0.10,
    #     0.06,
    #     0.06,
    #     0.06,
    # ]
    move_amplitudes: list = [
        0.3,
        0.2,
        0.2,
        0.2,
        0.3,
        0.3,
        0.3,
        0.3,
        0.2,
        0.2,
        0.2,
        0.3,
        0.3,
        0.3,
    ]


@configclass
class ResetVelocityCurriculumCfg:
    # reset 初始速度扰动课程持续时间；用于逐步增加每次重置时 base 的随机初速度。
    curriculum_duration_s: float = 4800.0

    # 训练刚开始时的扰动比例；0.0 表示一开始 reset 后 base 初速度完全为 0。
    initial_scale: float = 0.0

    # 课程结束后的扰动比例；1.0 表示使用下面 velocity_ranges 的完整范围。
    final_scale: float = 1.0

    # reset 时给 root state 采样的初始速度范围，单位：
    # x/y/z 为 m/s，roll/pitch/yaw 为 rad/s。
    # 实际范围会乘以当前课程 scale。
    velocity_ranges: dict = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
    }


@configclass
class PushVelocityCurriculumCfg:
    # 训练过程中外部推扰动课程持续时间；用于逐步增加 interval push 的强度。
    curriculum_duration_s: float = 4800.0

    # 训练刚开始时的推扰动比例；0.05 表示先用完整推力范围的 5%。
    initial_scale: float = 0.05

    # 课程结束后的推扰动比例；1.0 表示使用下面 velocity_ranges 的完整范围。
    final_scale: float = 1.0

    # push_robot 事件给 base 叠加的速度扰动范围，单位：
    # x/y 为 m/s，yaw 为 rad/s。
    # 实际范围会乘以当前课程 scale，并在 episode 中按 interval_range_s 周期触发。
    velocity_ranges: dict = {
        "x": (-0.8, 0.8),
        "y": (-0.8, 0.8),
        "yaw": (-0.2, 0.2),
    }


ELF3_TANG_INITIAL_JOINT_POS = dict(ELF3LITE_CFG.init_state.joint_pos)
ELF3_TANG_INITIAL_JOINT_POS.update(
    {
        "l_hip_y_joint": -0.30,
        "l_knee_y_joint": 0.60,
        "l_ankle_y_joint": -0.30,
        "r_hip_y_joint": -0.30,
        "r_knee_y_joint": 0.60,
        "r_ankle_y_joint": -0.30,
    }
)

ELF3_TANG_INITIAL_CFG = ELF3LITE_CFG.replace(
    init_state=ELF3LITE_CFG.init_state.replace(
        pos=(0.0, 0.0, 1.05),
        joint_pos=ELF3_TANG_INITIAL_JOINT_POS,
    )
)


@configclass
class TangRobotCfg(RobotCfg):
    pelvis_body_name: str = "waist_z_link"
    torso_body_name: str = "torso_link"
    termination_pelvis_height: float = 0.32
    termination_up_z: float = 0.5
    termination_contact_force: float = 50.0


@configclass
class LiteRewardCfg:
    # Locomotion uses pelvis heading, so commanded waist yaw does not steer the robot.
    track_lin_vel_xy_exp = RewTerm(func=tang_rewards.track_pelvis_velocity, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=tang_rewards.track_pelvis_turn_rate, weight=2.0, params={"std": 0.5})
    track_pelvis_height = RewTerm(func=tang_rewards.track_pelvis_height, weight=2.0, params={"std": 0.10})
    track_waist_yaw = RewTerm(func=tang_rewards.track_waist_yaw, weight=1.5, params={"std": 0.25})
    lin_vel_z_l2 = RewTerm(func=tang_rewards.pelvis_vertical_velocity, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=tang_rewards.pelvis_roll_pitch_rate, weight=-0.10)

    # Roll/pitch are stability constraints, not commanded axes. No yaw-to-zero penalty.
    body_orientation_l2 = RewTerm(
        func=tang_rewards.body_upright, weight=-2.0,
        params={"body_id_attr": "torso_body_id"},
    )
    pelvis_orientation_l2 = RewTerm(
        func=tang_rewards.body_upright, weight=-1.0,
        params={"body_id_attr": "pelvis_body_id"},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1_always, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_x_joint", "waist_y_joint"])},
    )
    energy = RewTerm(
        func=mdp.energy, weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*", ".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*", ".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*", ".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"])},
    )
    raw_action_target_limits = RewTerm(func=tang_rewards.raw_action_target_limits, weight=-1.0)
    joint_torque_excess = RewTerm(
        func=tang_rewards.joint_torque_excess, weight=-0.5, params={"soft_limit": 0.9},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_knee_y.*", ".*_hip_.*", "waist_z.*", "torso_link"]),
                "threshold": 3.0},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=tang_rewards.feet_air_time, weight=0.15, params={"threshold": 0.4},
    )
    stand_feet_contact = RewTerm(func=tang_rewards.standing_feet_contact, weight=0.5)
    stand_feet_flat = RewTerm(func=tang_rewards.standing_feet_flat, weight=0.25)
    stand_center_of_mass = RewTerm(func=tang_rewards.standing_center_of_mass, weight=0.2)
    knee_inward = RewTerm(func=tang_rewards.knee_inward, weight=-0.2, params={"min_width": 0.14})
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_x.*")},
    )
    feet_force = RewTerm(
        func=mdp.body_force, weight=-3e-3,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x.*"),
                "threshold": 500, "max_reward": 400},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_x.*"), "threshold": 0.24},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x.*")},
    )
    fly = RewTerm(
        func=mdp.fly, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x.*"), "threshold": 1.0},
    )


@configclass
class Elf3WalkTangFlatEnvCfg:
    amp_motion_files_display = [
                                # "legged_lab/envs/elf3/datasets/motion_visualization/stand.txt",
                                "legged_lab/envs/elf3/datasets/motion_visualization/stand_back.txt",
                                # "legged_lab/envs/elf3/datasets/motion_visualization/walk_run.txt",
                                # "legged_lab/envs/elf3/datasets/motion_visualization/walk_around.txt",
                                "legged_lab/envs/elf3/datasets/motion_visualization/walk_left.txt",
                                "legged_lab/envs/elf3/datasets/motion_visualization/walk_right.txt",
                                # "legged_lab/envs/elf3/datasets/motion_visualization/walk.txt",
                                ]
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=12288,
        env_spacing=2.5,
        robot=ELF3_TANG_INITIAL_CFG,
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=1,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="torso_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )
    robot: TangRobotCfg = TangRobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        # critic_obs_history_length=1,
        # Policy action order is defined in elf3_env.py as:
        # waist_y, waist_x, waist_z,
        # l_hip_y, l_hip_x, l_hip_z, l_knee_y, l_ankle_y, l_ankle_x,
        # r_hip_y, r_hip_x, r_hip_z, r_knee_y, r_ankle_y, r_ankle_x.
        # Keep this list 15-D to avoid silently slicing a 29-D full-body scale table with a different joint order.
        action_scale=[
            0.231, 0.154, 0.213,
            0.213, 0.213, 0.231, 0.213, 0.373, 0.230,
            0.213, 0.213, 0.231, 0.213, 0.373, 0.230,
        ],
        terminate_contacts_body_names=[".*_hip_z.*", "waist_z.*", "torso_link"],
        feet_body_names=[".*_ankle_x.*"],
    )
    reward = WBCRewardCfg()
    gait = GaitCfg()
    arm_motion = ArmMotionCurriculumCfg()
    reset_velocity_curriculum = ResetVelocityCurriculumCfg()
    push_velocity_curriculum = PushVelocityCurriculumCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: TangCommandCfg = TangCommandCfg()
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.2),
                    "dynamic_friction_range": (0.4, 1.0),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                    "mass_distribution_params": (-0.5, 0.5),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.10, 0.10)},
                    "velocity_range": {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.05, 0.05),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                # interval_range_s=(10.0, 15.0),
                # interval_range_s=(4.0, 8.0),
                interval_range_s=(8.0, 12.0),
                # params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
                params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.07, 0.07)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class Elf3WalkTangAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    # num_steps_per_env = 32
    max_iterations = 200000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        # learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        # symmetry_cfg=RslRlSymmetryCfg(),  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg() 
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "OnPolicyRunner"
    experiment_name = "walk_tang_height_yaw"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk_tang_height_yaw"
    wandb_project = "walk_tang_height_yaw"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

 
