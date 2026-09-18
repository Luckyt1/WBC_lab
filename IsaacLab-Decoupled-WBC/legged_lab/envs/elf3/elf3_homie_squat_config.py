"""ELF3 WBC with HOMIE's knee cost and command-gated posture penalties."""

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab import mdp
from legged_lab.envs.elf3.elf3_wbc_config import Elf3WBCEnvCfg, Elf3WBCRewardCfg


@configclass
class Elf3HomieSquatRewardCfg(Elf3WBCRewardCfg):
    homie_knee_height = RewardTermCfg(
        func=mdp.homie_knee_height,
        weight=-0.75,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=["l_knee_y_joint", "r_knee_y_joint"],
            body_names=["torso_link"], preserve_order=True,
        )},
    )

    def __post_init__(self):
        super().__post_init__()
        # ELF3 uses torso height (nominal 1.05 m), not G1's pelvis height.
        # Preserve existing L1 costs above 1.00 m; release the hip/leg posture
        # bias below it, including knees grouped with hip pitch and ankles.
        for name in ("joint_deviation_hip", "joint_deviation_legs"):
            term = getattr(self, name)
            term.func = mdp.joint_deviation_l1_standing
            term.params["standing_height"] = 1.0


@configclass
class Elf3HomieSquatEnvCfg(Elf3WBCEnvCfg):
    reward = Elf3HomieSquatRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.body_height = (0.8, 1.05)
        self.commands.curriculum.body_height.start_range = (0.9, 1.05)
        self.commands.curriculum.body_height.end_range = (0.8, 1.05)
