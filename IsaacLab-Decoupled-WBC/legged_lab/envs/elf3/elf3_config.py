# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.elf3 import ELF3_CFG
from legged_lab.envs.base.base_config import CommandAxisCurriculumCfg
from legged_lab.envs.base.base_env_config import (
    BaseAgentCfg,
    BaseEnvCfg,
    RewardCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG


ELF3_ACTION_JOINT_NAMES = [
    "l_hip_y_joint",
    "l_hip_x_joint",
    "l_hip_z_joint",
    "l_knee_y_joint",
    "l_ankle_y_joint",
    "l_ankle_x_joint",
    "r_hip_y_joint",
    "r_hip_x_joint",
    "r_hip_z_joint",
    "r_knee_y_joint",
    "r_ankle_y_joint",
    "r_ankle_x_joint",
    "waist_z_joint",
    "waist_x_joint",
    "waist_y_joint",
]

ELF3_ARM_JOINT_NAMES = [
    "l_shoulder_y_joint",
    "l_shoulder_x_joint",
    "l_shoulder_z_joint",
    "l_elbow_y_joint",
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    "r_shoulder_y_joint",
    "r_shoulder_x_joint",
    "r_shoulder_z_joint",
    "r_elbow_y_joint",
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
]

ELF3_ACTION_SCALE = {
    ".*_hip_y_joint": 0.213,
    ".*_hip_x_joint": 0.213,
    ".*_hip_z_joint": 0.231,
    ".*_knee_y_joint": 0.213,
    ".*_ankle_y_joint": 0.373,
    ".*_ankle_x_joint": 0.230,
    "waist_z_joint": 0.213,
    "waist_x_joint": 0.154,
    "waist_y_joint": 0.231,
}


@configclass
class Elf3RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.75, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*waist.*",
                    ".*_hip_.*",
                    ".*_knee_.*",
                    ".*_ankle_.*",
                ],
            )
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[
                    ".*_hip_z.*",
                    ".*_knee_y.*",
                    "waist_z_link",
                    "torso_link",
                ],
            ),
            "threshold": 1.0,
        },
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x_link.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.10,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x_link.*"), "threshold": 0.4},
    )
    feet_contact_standing_stable = RewTerm(
        func=mdp.feet_contact_standing_stable,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x_link.*"), "min_contact_time": 0.5},
    )
    feet_flat_standing = RewTerm(
        func=mdp.feet_flat_standing,
        weight=0.175,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["l_ankle_x_link", "r_ankle_x_link"],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["l_ankle_x_link", "r_ankle_x_link"],
                preserve_order=True,
            ),
            "dead_zone_deg": 3.0,
            "std_deg": 10.0,
            "min_contact_time": 0.5,
            "contact_ramp_s": 0.25,
            "roll_dead_zone": 0.45,
            "pitch_dead_zone": 0.65,
            "roll_scale": 0.35,
            "pitch_scale": 0.45,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x_link.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_x_link.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x_link.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_x_link.*"]), "threshold": 0.24},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_x_link.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_x_joint", ".*_hip_z_joint"])},
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_y_joint", ".*_knee_y_joint", ".*_ankle_.*"])},
    )
    track_body_height_exp = RewTerm(
        func=mdp.track_body_height_exp,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "std": 0.1},
    )
    track_body_roll_exp = RewTerm(
        func=mdp.track_body_roll_exp,
        weight=0.75,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["waist_z_link", "torso_link"],
                preserve_order=True,
            ),
            "std": 0.12,
        },
    )
    track_body_pitch_exp = RewTerm(
        func=mdp.track_body_pitch_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["waist_z_link", "torso_link"],
                preserve_order=True,
            ),
            "std": 0.12,
        },
    )
    track_body_yaw_exp = RewTerm(
        func=mdp.track_body_yaw_exp,
        weight=0.75,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["waist_z_link", "torso_link"],
                preserve_order=True,
            ),
            "std": 0.12,
        },
    )
    center_of_gravity_tracking = RewTerm(
        func=mdp.center_of_gravity_tracking,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["l_ankle_x_link", "r_ankle_x_link"],
                preserve_order=True,
            ),
            "sigma": 0.2,
        },
    )
    knee_stance_width = RewTerm(
        func=mdp.knee_stance_width,
        weight=0.4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["l_knee_y_link", "r_knee_y_link"],
                preserve_order=True,
            ),
            "ideal_width": 0.27,
            "std": 0.1,
        },
    )
    joint_torque_limits_waists = RewTerm(
        func=mdp.joint_torque_limits,
        weight=-100.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"]), "soft_torque_limit": 0.9},
    )
    joint_torque_limits_ankles = RewTerm(
        func=mdp.joint_torque_limits,
        weight=-100.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*"]), "soft_torque_limit": 0.99},
    )


@configclass
class Elf3FlatEnvCfg(BaseEnvCfg):
    reward = Elf3RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ELF3_CFG
        self.robot.asset_revision = "elf3_dof31_urdf_6ede896d8718"
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.scene.num_envs = 1024
        self.scene.max_init_terrain_level = 1
        self.scene.height_scanner.prim_body_name = "torso_link"

        self.robot.action_joint_names = ELF3_ACTION_JOINT_NAMES
        self.robot.arm_joint_names = ELF3_ARM_JOINT_NAMES
        self.robot.arm_motion_source = "default"
        self.robot.nominal_height = 1.05
        self.robot.termination_height = 0.45
        self.robot.base_body_name = "waist_z_link"
        self.robot.action_scale = ELF3_ACTION_SCALE
        self.robot.terminate_contacts_body_names = ["waist_z_link", "torso_link"]
        self.robot.feet_body_names = ["l_ankle_x_link", "r_ankle_x_link"]
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1

        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)
        self.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (-0.10, 0.10),
        }
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.domain_rand.events.push_robot.interval_range_s = (8.0, 12.0)
        self.domain_rand.events.push_robot.params["velocity_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "yaw": (-0.07, 0.07),
        }

        self.commands.rel_standing_envs = 0.2
        self.commands.rel_heading_envs = 0.0
        self.commands.rel_zero_vel_yaw_envs = 0.2
        self.commands.heading_command = False
        self.commands.ranges.lin_vel_x = (-0.4, 0.6)
        self.commands.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.ranges.ang_vel_z = (-0.7, 0.7)
        self.commands.ranges.heading = None
        self.commands.ranges.body_height = (0.95, 1.12)
        self.commands.ranges.body_roll = (-0.35, 0.35)
        self.commands.ranges.body_pitch = (-0.35, 0.35)
        self.commands.ranges.body_yaw = (-0.6, 0.6)

        self.commands.curriculum.enable = True
        self.commands.curriculum.lin_vel_x = CommandAxisCurriculumCfg(
            start_range=(0.0, 0.0),
            end_range=(-0.4, 0.6),
            start_iter=10000,
            end_iter=50000,
        )
        self.commands.curriculum.lin_vel_y = CommandAxisCurriculumCfg(
            start_range=(0.0, 0.0),
            end_range=(-0.25, 0.25),
            start_iter=15000,
            end_iter=50000,
        )
        self.commands.curriculum.ang_vel_z = CommandAxisCurriculumCfg(
            start_range=(0.0, 0.0),
            end_range=(-0.7, 0.7),
            start_iter=15000,
            end_iter=50000,
        )
        self.commands.curriculum.body_height = CommandAxisCurriculumCfg(
            start_range=(1.02, 1.08),
            end_range=(0.95, 1.12),
            start_iter=10000,
            end_iter=50000,
        )
        self.commands.curriculum.body_roll = CommandAxisCurriculumCfg(
            start_range=(0.0, 0.0),
            end_range=(-0.35, 0.35),
            start_iter=20000,
            end_iter=60000,
        )
        self.commands.curriculum.body_pitch = CommandAxisCurriculumCfg(
            start_range=(0.0, 0.0),
            end_range=(-0.35, 0.35),
            start_iter=20000,
            end_iter=60000,
        )
        self.commands.curriculum.body_yaw = CommandAxisCurriculumCfg(
            start_range=(0.0, 0.0),
            end_range=(-0.6, 0.6),
            start_iter=20000,
            end_iter=60000,
        )


@configclass
class Elf3FlatAgentCfg(BaseAgentCfg):
    max_iterations: int = 60000
    experiment_name: str = "elf3_flat"
    wandb_project: str = "elf3_flat"
    logger: str = "tensorboard"
