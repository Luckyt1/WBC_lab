"""Continue HOMIE squatting while correcting raw action-target saturation."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from legged_lab import mdp
from legged_lab.envs.elf3.elf3_homie_squat_config import Elf3HomieSquatEnvCfg, Elf3HomieSquatRewardCfg


@configclass
class Elf3HomieRecoveryRewardCfg(Elf3HomieSquatRewardCfg):
    # Linear excess avoids squaring already large inherited action magnitudes.
    # Use all policy joints: the diagnostic also found ankle-roll saturation.
    raw_action_target_limits = RewardTermCfg(func=mdp.raw_action_target_limits_l1, weight=-0.2)


@configclass
class Elf3HomieRecoveryEnvCfg(Elf3HomieSquatEnvCfg):
    reward = Elf3HomieRecoveryRewardCfg()
