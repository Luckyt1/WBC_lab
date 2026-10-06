"""Standing and locomotion recovery with dense tracking and bounded exploration."""
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from . import standing
from .commands import TangCommandCfg
from .walk_bxi_cfg import Elf3WalkTangBxiEnvCfg, Elf3WalkTangBxiAgentCfg
from .wbc.reward_cfg import WBCRewardCfg
from .wbc import rewards as wbc_rewards


@configclass
class StandingRewardCfg(WBCRewardCfg):
    # Wider bonuses and dense costs retain a signal when a policy stands still
    # despite a turn command, or has settled below the requested height.
    track_lin_vel_xy_exp = RewTerm(func=wbc_rewards.track_lin_vel_xy_yaw_frame_exp, weight=5.0, params={'std': 0.35})
    track_ang_vel_z_exp = RewTerm(func=wbc_rewards.track_ang_vel_z_world_exp, weight=5.0, params={'std': 0.4})
    track_body_height_exp = RewTerm(func=wbc_rewards.track_body_height_exp, weight=3.0,
                                   params={'asset_cfg': SceneEntityCfg('robot', body_names=['waist_z_link'], preserve_order=True), 'std': 0.1})
    velocity_tracking_error = RewTerm(func=standing.velocity_tracking_error, weight=-2.0)
    turn_tracking_error = RewTerm(func=standing.turn_tracking_error, weight=-2.0)
    height_tracking_error = RewTerm(func=standing.height_tracking_error, weight=-10.0)
    waist_yaw_tracking_error = RewTerm(func=standing.waist_yaw_tracking_error, weight=-2.0)
    track_body_yaw_exp = RewTerm(func=wbc_rewards.track_body_yaw_exp, weight=1.0,
                                params={'asset_cfg': SceneEntityCfg('robot', body_names=['waist_z_link', 'torso_link'], preserve_order=True),
                                        'std': 0.3, 'wrap_error': True})
    # Use the same task mask as the dense standing terms, including small turns.
    feet_air_time = RewTerm(func=wbc_rewards.feet_air_time_positive_biped, weight=0.15,
                           params={'sensor_cfg': SceneEntityCfg('contact_sensor', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True),
                                   'threshold': 0.4, 'use_standing_mask': True})
    feet_contact_standing_stable = RewTerm(func=wbc_rewards.feet_contact_standing_stable, weight=0.5,
                                          params={'sensor_cfg': SceneEntityCfg('contact_sensor', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True),
                                                  'min_contact_time': 0.5, 'use_standing_mask': True})
    standing_contact = RewTerm(func=standing.standing_contact, weight=2.0)
    standing_quiet_base = RewTerm(func=standing.standing_quiet_base, weight=2.0)
    standing_foot_motion = RewTerm(func=standing.standing_foot_motion, weight=-1.0)
    standing_base_motion = RewTerm(func=standing.standing_base_motion, weight=-0.5)
    standing_foot_tilt = RewTerm(func=standing.standing_foot_tilt, weight=0.5)


@configclass
class StandingPolicyCfg(RslRlPpoActorCriticCfg):
    noise_std_bounds: tuple = (0.1, 0.3)


@configclass
class StandingAlgorithmCfg(RslRlPpoAlgorithmCfg):
    action_mean_bound_loss_coef: float = 0.1


@configclass
class Elf3StandWalkBxiEnvCfg(Elf3WalkTangBxiEnvCfg):
    reward = StandingRewardCfg()
    commands = TangCommandCfg(rel_standing_envs=0.3, rel_turning_envs=0.25,
                              rel_neutral_standing=0.7, rel_nominal_height=0.5)
    standing_diagnostics = True

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096


@configclass
class Elf3StandWalkBxiAgentCfg(Elf3WalkTangBxiAgentCfg):
    experiment_name = 'walk_tang_bxi31_standing'
    max_iterations = 10000
    policy = StandingPolicyCfg(**Elf3WalkTangBxiAgentCfg().policy.to_dict())
    algorithm = StandingAlgorithmCfg(**Elf3WalkTangBxiAgentCfg().algorithm.to_dict())

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.001
