"""Fresh 50k locomotion/disturbance run; neutral height and waist throughout."""
from isaaclab.utils import configclass
from .scratch_cfg import Elf3ScratchBxiEnvCfg, Elf3ScratchBxiAgentCfg
from .walk_robust_curriculum import configure_walk_robust


@configclass
class Elf3WalkRobustBxiEnvCfg(Elf3ScratchBxiEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_walk_robust(self)


@configclass
class Elf3WalkRobustBxiAgentCfg(Elf3ScratchBxiAgentCfg):
    experiment_name = 'walk_tang_bxi31_robust'
    max_iterations = 50000
    save_interval = 500
