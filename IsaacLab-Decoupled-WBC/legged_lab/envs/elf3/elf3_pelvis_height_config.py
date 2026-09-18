"""ELF3 full WBC with pelvis-height tracking and unconditional posture costs."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from legged_lab import mdp
from legged_lab.envs.elf3.elf3_wbc_config import Elf3WBCEnvCfg, Elf3WBCRewardCfg

# In the current URDF, the three waist joint origins coincide 0.2265 m below
# torso_link. This converts upright reference heights only; the reward measures
# waist_z_link directly, without adding an offset to its measured position.
UPRIGHT_TORSO_TO_PELVIS_OFFSET = 0.2265
PELVIS_HEIGHT_RANGE = (0.4735, 0.8235)
PELVIS_STANDING_HEIGHT = 0.8235


@configclass
class Elf3PelvisHeightRewardCfg(Elf3WBCRewardCfg):
    # Preserve the existing action-target cost independently of HOMIE.
    raw_action_target_limits = RewardTermCfg(func=mdp.raw_action_target_limits_l1, weight=-0.2)

    def __post_init__(self):
        super().__post_init__()
        self.track_body_height_exp.params["asset_cfg"].body_names = ["waist_z_link"]


@configclass
class Elf3PelvisHeightEnvCfg(Elf3WBCEnvCfg):
    reward = Elf3PelvisHeightRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.robot.nominal_height = PELVIS_STANDING_HEIGHT
        self.commands.ranges.body_height = PELVIS_HEIGHT_RANGE
        self.commands.curriculum.body_height.start_range = PELVIS_HEIGHT_RANGE
        self.commands.curriculum.body_height.end_range = PELVIS_HEIGHT_RANGE
