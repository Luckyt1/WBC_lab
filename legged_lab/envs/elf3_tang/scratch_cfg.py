"""Fresh BXI training: staged commands/arms, proprioceptive arms, no teacher."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from .stand_cfg import Elf3StandWalkBxiEnvCfg, Elf3StandWalkBxiAgentCfg, StandingAlgorithmCfg, StandingRewardCfg
from .focus_cfg import GuardedArmMotionCfg
from .scratch_curriculum import configure_scratch_stage
from . import rewards


@configclass
class ScratchArmMotionCfg(GuardedArmMotionCfg):
    guard_fixed_targets = True
    amass_neutral_fraction = .30
    scale = .05
    amass_scale_end_iteration = None


@configclass
class ScratchAlgorithmCfg(StandingAlgorithmCfg):
    learning_rate_bounds: tuple = (1.e-5, 3.e-4)


@configclass
class ScratchRewardCfg(StandingRewardCfg):
    # Penalize the requested target BEFORE clipping/delay; physical limits stay unchanged.
    raw_action_target_limits = RewTerm(func=rewards.raw_action_target_limits, weight=-1.0)


@configclass
class Elf3ScratchBxiEnvCfg(Elf3StandWalkBxiEnvCfg):
    observe_arm_state = True
    arm_motion = ScratchArmMotionCfg()
    reward = ScratchRewardCfg()
    scratch_stage = 'locomotion'

    def __post_init__(self):
        super().__post_init__()
        configure_scratch_stage(self, self.scratch_stage)


@configclass
class Elf3ScratchBxiAgentCfg(Elf3StandWalkBxiAgentCfg):
    experiment_name = 'walk_tang_bxi31_scratch'
    max_iterations = 5000
    resume = False
    algorithm = ScratchAlgorithmCfg(**Elf3StandWalkBxiAgentCfg().algorithm.to_dict())

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = .5
        self.policy.noise_std_bounds = (.1, .6)
        self.algorithm.learning_rate = 3.e-4
        self.algorithm.entropy_coef = .003
        self.algorithm.desired_kl = .008
        self.algorithm.action_mean_bound_loss_coef = 1.0
