"""4096-env specialist continuation with retained locomotion and AMASS guard."""
from isaaclab.utils import configclass
from .stand_cfg import Elf3StandWalkBxiEnvCfg, Elf3StandWalkBxiAgentCfg
from .walk_cfg import ArmMotionCurriculumCfg
from .commands import TangCommandCfg


@configclass
class GuardedArmMotionCfg(ArmMotionCurriculumCfg):
    collision_guard = True
    ramp_s = 1.0
    # The guard enforces its own step limit and reports reference deviation.
    amass_max_velocity = None


@configclass
class Elf3FocusedBxiEnvCfg(Elf3StandWalkBxiEnvCfg):
    arm_motion = GuardedArmMotionCfg()
    commands = TangCommandCfg(rel_standing_envs=.25, rel_turning_envs=.15,
                              rel_lateral_envs=.15, rel_crouch_yaw_envs=.15,
                              rel_neutral_standing=.7, rel_nominal_height=.5)


@configclass
class Elf3FocusedBxiAgentCfg(Elf3StandWalkBxiAgentCfg):
    experiment_name = 'walk_tang_bxi31_focused'
    max_iterations = 500
