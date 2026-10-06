"""Explicit new-asset task; legacy checkpoints retain their original 29-joint task."""
from isaaclab.utils import configclass

from legged_lab.assets.elf3_bxi_31 import ELF3_BXI_CFG, PARAMETERS
from .walk_cfg import Elf3WalkTangFlatEnvCfg, Elf3WalkTangAgentCfg


@configclass
class Elf3WalkTangBxiEnvCfg(Elf3WalkTangFlatEnvCfg):
    def __post_init__(self):
        self.scene.robot = ELF3_BXI_CFG.copy()
        self.robot.action_scale = [PARAMETERS['joints'][name]['action_scale']
                                   for name in PARAMETERS['policy_joint_names']]
        assert abs(self.sim.dt * self.sim.decimation - PARAMETERS['control_period_s']) < 1e-9


@configclass
class Elf3WalkTangBxiAgentCfg(Elf3WalkTangAgentCfg):
    experiment_name = 'walk_tang_height_yaw_bxi31'
