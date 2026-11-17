"""Three velocity commands; 15 waist/leg actions; 14 externally driven IK joints."""

from math import radians

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.gl165.model import (
    CONTROL_SOURCE, FEET_BODY_NAMES, POLICY_JOINT_NAMES, asset_dir, ik_bank_dir, policy_action_scales,
)
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from . import rewards


@configclass
class Gl165Rewards:
    track_lin_vel = RewTerm(func=rewards.track_linear_velocity, weight=2.0, params={"std": .5})
    track_ang_vel = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": .5})
    alive = RewTerm(func=rewards.alive, weight=.5)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-.05)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    torso_upright = RewTerm(func=rewards.torso_upright, weight=-3.0,
                           params={"asset_cfg": SceneEntityCfg("robot", body_names=["body_y_link"])})
    # These are soft posture bands, not waist commands or mechanical locks.
    waist_range = RewTerm(func=rewards.joint_excursion, weight=-2.0,
                         params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES[:3],
                                                             preserve_order=True),
                                 "tolerance": tuple(radians(x) for x in (15., 8., 8.))})
    waist_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-.02,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES[:3])})
    stand_leg_posture = RewTerm(func=rewards.standing_leg_posture, weight=-.5,
                               params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES[3:],
                                                                   preserve_order=True),
                                       "tolerance": tuple(radians(x) for x in (15., 10., 15., 15., 10., 8.)) * 2})
    support_knee_bend = RewTerm(func=rewards.support_knee_bend, weight=-2.0,
                               params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lknee_y_joint", "rknee_y_joint"],
                                                                   preserve_order=True),
                                       "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET_BODY_NAMES,
                                                                    preserve_order=True),
                                       "standing_limit": radians(60.), "moving_limit": radians(75.)})
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7,
                       params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES)})
    energy = RewTerm(func=mdp.energy, weight=-2e-4,
                    params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES)})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-.01)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0,
                          params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES)})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-.25,
                        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET_BODY_NAMES),
                                "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES)})
    # Replaced by phase-aware contact, clearance and foot-placement costs.
    # Kept disabled here so legacy checkpoints can restore their old contract.
    feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped, weight=0.,
                           params={"threshold": .4,
                                   "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET_BODY_NAMES)})
    moving_feet_distance = RewTerm(func=rewards.moving_feet_distance, weight=-2.0,
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES,
                                                                     preserve_order=True),
                                          "max_distance": .4, "command_tolerance": 1e-6})
    feet_heading = RewTerm(func=rewards.feet_heading, weight=-2.0,
                           params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES,
                                                              preserve_order=True),
                                   "tolerance": radians(15.)})
    # Default-relative: the angled hip mounts need nonzero neutral hip yaw.
    hip_yaw_range = RewTerm(func=rewards.joint_excursion, weight=-.5,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lhip_z_joint", "rhip_z_joint"],
                                                               preserve_order=True),
                                    "tolerance": (radians(20.), radians(20.))})
    feet_lateral_band = RewTerm(func=rewards.feet_lateral_band, weight=-2.,
                               params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES,
                                                                   preserve_order=True),
                                       "minimum": .16, "maximum": .26, "scale": .1})
    gait_contact = RewTerm(func=rewards.gait_contact, weight=-1.,
                          params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET_BODY_NAMES,
                                                               preserve_order=True), "contact_force": 20.})
    gait_clearance = RewTerm(func=rewards.gait_swing_clearance, weight=-.5,
                            params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES,
                                                                preserve_order=True), "scale": .04})
    gait_fore_aft = RewTerm(func=rewards.gait_fore_aft, weight=-1.,
                           params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES,
                                                               preserve_order=True),
                                   "tolerance": .02, "scale": .08})
    stand_foot_alignment = RewTerm(func=rewards.standing_foot_alignment, weight=-1.,
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES,
                                                                      preserve_order=True),
                                          "tolerance": .06, "scale": .1})
    stand_still = RewTerm(func=rewards.stand_still, weight=1.0)
    stand_contact = RewTerm(func=rewards.stand_feet_contact, weight=.5)
    stand_foot_load = RewTerm(func=rewards.standing_foot_load, weight=-2.0,
                             params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET_BODY_NAMES,
                                                                  preserve_order=True),
                                     "threshold": 150.0, "command_tolerance": 1e-6})
    stand_drift = RewTerm(func=rewards.standing_drift, weight=-2.0)
    # No root-height target, waist-yaw target, or waist pose command tracking.


@configclass
class Gl165GaitCfg:
    enabled: bool = True
    observation_version: int = 1
    period_s: float = .9
    stance_fraction: float = .6
    contact_phase_margin: float = .03
    start_transition_s: float = .3
    stop_transition_s: float = .4
    stop_alignment_delay_s: float = .4
    stop_alignment_ramp_s: float = .6
    foot_origin_ground_height_m: float = .032
    swing_clearance_m: float = .06
    max_fore_aft_m: float = .30  # sqrt(.30^2 + .26^2) < the 40 cm total bound.


@configclass
class Gl165WbcEnvCfg(BaseEnvCfg):
    asset_path: str = str(asset_dir())
    ik_bank_path: str = str(ik_bank_dir())
    ik_split: str = "train"
    min_root_height: float = .55  # Fall detection only.
    control_parameters_source: dict = CONTROL_SOURCE
    checkpoint_joint_gains: dict | None = None
    gait = Gl165GaitCfg()
    reward = Gl165Rewards()

    def __post_init__(self):
        self.scene.num_envs = 4096
        self.scene.env_spacing = 3.0
        self.scene.max_episode_length_s = 40.0
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.scene.height_scanner.enable_height_scan = False
        self.robot.feet_body_names = FEET_BODY_NAMES
        self.robot.terminate_contacts_body_names = ["base_link", "body_y_link", "head_y_link", ".*smallleg_y_link"]
        self.robot.action_scale = policy_action_scales()
        self.normalization.clip_actions = 5.0
        self.normalization.obs_scales.ang_vel = .25
        self.normalization.obs_scales.joint_vel = .05
        self.commands.rel_standing_envs = .5
        self.commands.heading_command = False
        self.commands.rel_heading_envs = 0.0
        self.commands.debug_vis = False
        self.commands.resampling_time_range = (6.0, 10.0)
        self.commands.ranges.lin_vel_x = (-.5, .6)
        self.commands.ranges.lin_vel_y = (-.3, .3)
        self.commands.ranges.ang_vel_z = (-.8, .8)
        self.domain_rand.events.add_base_mass.params.update(
            asset_cfg=SceneEntityCfg("robot", body_names="base_link"), mass_distribution_params=(-.5, .5))
        self.domain_rand.events.reset_base.params.update(
            pose_range={"x": (-.25, .25), "y": (-.25, .25), "yaw": (-3.14, 3.14)},
            velocity_range={k: (-.05, .05) for k in ("x", "y", "z", "roll", "pitch", "yaw")})
        self.domain_rand.events.reset_robot_joints.params.update(
            position_range=(.98, 1.02), velocity_range=(0.0, 0.0))
        self.domain_rand.events.push_robot.params["velocity_range"] = {"x": (-.3, .3), "y": (-.3, .3)}


@configclass
class Gl165WbcAgentCfg(BaseAgentCfg):
    runner_class_name = "OnPolicyRunner"
    experiment_name = "gl165_wbc"
    logger = "tensorboard"
    max_iterations = 50000
    save_interval = 500
    resume = False

    def __post_init__(self):
        self.policy.init_noise_std = .6
