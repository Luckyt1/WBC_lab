# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

# Register independent robot families; their external assets are loaded on use.
from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.elf3_tang.elf3_env import Elf3Env as Elf3TangEnv
from legged_lab.utils.task_registry import task_registry

from legged_lab.envs.elf3_tang.walk_cfg import (
    Elf3WalkTangFlatEnvCfg,
    Elf3WalkTangAgentCfg,
)
task_registry.register("walk_elf3_tang", Elf3TangEnv, Elf3WalkTangFlatEnvCfg(), Elf3WalkTangAgentCfg())

from legged_lab.envs.elf3_tang.walk_bxi_cfg import Elf3WalkTangBxiEnvCfg, Elf3WalkTangBxiAgentCfg
task_registry.register("walk_elf3_tang_bxi", Elf3TangEnv, Elf3WalkTangBxiEnvCfg(), Elf3WalkTangBxiAgentCfg())

from legged_lab.envs.elf3_tang.stand_cfg import Elf3StandWalkBxiEnvCfg, Elf3StandWalkBxiAgentCfg
task_registry.register("walk_elf3_tang_bxi_stand", Elf3TangEnv, Elf3StandWalkBxiEnvCfg(), Elf3StandWalkBxiAgentCfg())

from legged_lab.envs.elf3_tang.focus_cfg import Elf3FocusedBxiEnvCfg, Elf3FocusedBxiAgentCfg
task_registry.register("walk_elf3_tang_bxi_focus", Elf3TangEnv, Elf3FocusedBxiEnvCfg(), Elf3FocusedBxiAgentCfg())

from legged_lab.envs.elf3_tang.scratch_cfg import Elf3ScratchBxiEnvCfg, Elf3ScratchBxiAgentCfg
task_registry.register("walk_elf3_tang_bxi_scratch", Elf3TangEnv, Elf3ScratchBxiEnvCfg(), Elf3ScratchBxiAgentCfg())

from legged_lab.envs.elf3_tang.walk_robust_cfg import Elf3WalkRobustBxiEnvCfg, Elf3WalkRobustBxiAgentCfg
task_registry.register("walk_elf3_tang_bxi_robust", Elf3TangEnv, Elf3WalkRobustBxiEnvCfg(), Elf3WalkRobustBxiAgentCfg())

from legged_lab.envs.elf3_tang.standing_refine_cfg import Elf3StandingRefineEnvCfg, Elf3StandingRefineAgentCfg
task_registry.register("walk_elf3_tang_bxi_still", Elf3TangEnv, Elf3StandingRefineEnvCfg(), Elf3StandingRefineAgentCfg())

from legged_lab.envs.gl165.gl165_env import Gl165Env
from legged_lab.envs.gl165.wbc_cfg import Gl165WbcAgentCfg, Gl165WbcEnvCfg

task_registry.register("walk_gl165_wbc", Gl165Env, Gl165WbcEnvCfg(), Gl165WbcAgentCfg())
