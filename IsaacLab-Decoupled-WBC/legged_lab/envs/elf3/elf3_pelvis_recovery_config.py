"""Opt-in recovery branch from pelvis WBC iteration 40000."""

from copy import deepcopy
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from legged_lab.mdp import recovery_rewards
from legged_lab.envs.elf3.elf3_pelvis_height_config import Elf3PelvisHeightEnvCfg
from legged_lab.envs.elf3.elf3_wbc_config import Elf3WBCAgentCfg


@configclass
class Elf3PelvisRecoveryEnvCfg(Elf3PelvisHeightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (-.4, 1.)
        self.commands.ranges.lin_vel_y = (-1., 1.)
        self.commands.ranges.ang_vel_z = (-1.57, 1.57)
        self.commands.rel_standing_envs = .4
        self.commands.rel_in_place_turn_envs = .2
        # 40% stand / 10% high in-place turn / 10% general in-place turn /
        # 10% high moving turn / 30% general translation and combinations.
        self.commands.rel_high_stance_turn_envs = .1
        self.commands.rel_high_stance_move_turn_envs = .1
        self.commands.high_stance_height_range = (.78, .8235)
        self.commands.high_stance_speed_range = (.3, .6)
        self.commands.high_stance_yaw_rate_range = (.2, .6)
        self.commands.in_place_small_turn_fraction = .7
        self.commands.rel_zero_pose_envs = .2
        self.commands.rel_single_axis_pose_envs = .3
        reward = self.reward
        reward.track_body_height_exp.weight = 4.
        reward.track_body_height_exp.params['std'] = .15
        reward.track_lin_vel_xy_exp.weight = 2.5
        reward.track_lin_vel_xy_exp.params['std'] = .3
        reward.track_ang_vel_z_exp = RewardTermCfg(
            func=recovery_rewards.track_yaw_rate, weight=2.,
            params={'min_std': .15, 'max_std': .3, 'command_scale': .5})
        reward.feet_air_time.weight = .5
        reward.joint_deviation_hip.weight = -.1
        reward.action_rate_l2.weight = -.005
        sensor = deepcopy(reward.feet_contact_standing_stable.params['sensor_cfg'])
        reward.feet_contact_standing_stable = RewardTermCfg(
            func=recovery_rewards.feet_stance, weight=.5, params={'sensor_cfg': sensor, 'ramp_s': .15})
        reward.feet_lift_standing = RewardTermCfg(
            func=recovery_rewards.feet_lift_standing, weight=-.2, params={'sensor_cfg': deepcopy(sensor)})
        for name, weight in (('roll', 1.), ('pitch', 1.5), ('yaw', 1.)):
            term = getattr(reward, f'track_body_{name}_exp')
            term.weight = weight
            term.params['std'] = .3
        reward.track_body_yaw_exp.func = recovery_rewards.track_torso_yaw


@configclass
class Elf3PelvisRecoveryAgentCfg(Elf3WBCAgentCfg):
    # Kept outside algorithm kwargs: the standard runner creates PPO, then the
    # train entrypoint installs the project-local update guard after loading.
    ppo_stability: dict = {
        'initial_learning_rate': 1e-4, 'min_learning_rate': 1e-5,
        'max_learning_rate': 3e-4, 'max_mean_kl': .02,
    }

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1e-4
        self.algorithm.num_learning_epochs = 3
        self.algorithm.desired_kl = .01
