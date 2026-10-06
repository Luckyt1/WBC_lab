"""Checkpoint fine-tuning task for quiet standing without losing locomotion."""
from isaaclab.utils import configclass

from .walk_robust_cfg import Elf3WalkRobustBxiEnvCfg, Elf3WalkRobustBxiAgentCfg
from .standing_refine import configure_standing_refinement, configure_standing_refinement_agent


@configclass
class Elf3StandingRefineEnvCfg(Elf3WalkRobustBxiEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_standing_refinement(self)


@configclass
class Elf3StandingRefineAgentCfg(Elf3WalkRobustBxiAgentCfg):
    # Same experiment root permits explicitly choosing the original run to resume.
    # Each refinement still writes its own timestamped run directory.
    run_name = 'standing_refine'
    max_iterations = 5000
    save_interval = 500

    def __post_init__(self):
        super().__post_init__()
        configure_standing_refinement_agent(self)
