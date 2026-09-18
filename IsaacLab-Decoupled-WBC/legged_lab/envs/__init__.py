# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.elf3.elf3_config import Elf3FlatAgentCfg, Elf3FlatEnvCfg
from legged_lab.envs.elf3.elf3_wbc_config import Elf3WBCAgentCfg, Elf3WBCEnvCfg
from legged_lab.envs.elf3.elf3_homie_squat_config import Elf3HomieSquatEnvCfg
from legged_lab.envs.elf3.elf3_homie_recovery_config import Elf3HomieRecoveryEnvCfg
from legged_lab.envs.elf3.elf3_pelvis_height_config import Elf3PelvisHeightEnvCfg
from legged_lab.envs.elf3.elf3_pelvis_recovery_config import Elf3PelvisRecoveryEnvCfg, Elf3PelvisRecoveryAgentCfg
from legged_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)
from legged_lab.utils.task_registry import task_registry

task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_rough", BaseEnv, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("elf3_flat", BaseEnv, Elf3FlatEnvCfg(), Elf3FlatAgentCfg())
task_registry.register("elf3_wbc", BaseEnv, Elf3WBCEnvCfg(), Elf3WBCAgentCfg())
task_registry.register("elf3_homie_squat", BaseEnv, Elf3HomieSquatEnvCfg(), Elf3WBCAgentCfg())
task_registry.register("elf3_homie_recovery", BaseEnv, Elf3HomieRecoveryEnvCfg(), Elf3WBCAgentCfg())
task_registry.register("elf3_pelvis_height", BaseEnv, Elf3PelvisHeightEnvCfg(), Elf3WBCAgentCfg())
task_registry.register("elf3_pelvis_recovery", BaseEnv, Elf3PelvisRecoveryEnvCfg(), Elf3PelvisRecoveryAgentCfg())
