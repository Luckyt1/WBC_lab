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

"""ELF3 31-DOF deployment-URDF asset with deployment PD gains.

Effort and velocity limits are inherited from the verified USD, so the old
29-DOF limits cannot override the current URDF. Head joints hold zero targets.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR


ELF3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/elf3_dof31_current/usd/elf3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "head_z_joint": 0.0,
            "head_y_joint": 0.0,
            "waist_y_joint": 0.0,
            "waist_x_joint": 0.0,
            "waist_z_joint": 0.0,
            "l_hip_y_joint": -0.3,
            "l_hip_x_joint": 0.0,
            "l_hip_z_joint": 0.0,
            "l_knee_y_joint": 0.6,
            "l_ankle_y_joint": -0.3,
            "l_ankle_x_joint": 0.0,
            "r_hip_y_joint": -0.3,
            "r_hip_x_joint": 0.0,
            "r_hip_z_joint": 0.0,
            "r_knee_y_joint": 0.6,
            "r_ankle_y_joint": -0.3,
            "r_ankle_x_joint": 0.0,
            "l_shoulder_y_joint": 0.2,
            "l_shoulder_x_joint": 0.2,
            "l_shoulder_z_joint": 0.0,
            "l_elbow_y_joint": 0.6,
            "l_wrist_x_joint": 0.0,
            "l_wrist_y_joint": 0.0,
            "l_wrist_z_joint": 0.0,
            "r_shoulder_y_joint": 0.2,
            "r_shoulder_x_joint": -0.2,
            "r_shoulder_z_joint": 0.0,
            "r_elbow_y_joint": 0.6,
            "r_wrist_x_joint": 0.0,
            "r_wrist_y_joint": 0.0,
            "r_wrist_z_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_z_joint", "head_y_joint"],
            # Same gains as the deployment hello state's head controller.
            stiffness=16.747,
            damping=1.066,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_y_joint", "waist_x_joint", "waist_z_joint"],
            stiffness={
                "waist_y_joint": 108.448,
                "waist_x_joint": 162.672,
                "waist_z_joint": 176.421,
            },
            damping={
                "waist_y_joint": 6.904,
                "waist_x_joint": 10.356,
                "waist_z_joint": 11.231,
            },
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_y_joint",
                ".*_hip_x_joint",
                ".*_hip_z_joint",
                ".*_knee_y_joint",
            ],
            stiffness={
                ".*_hip_y_joint": 176.421,
                ".*_hip_x_joint": 176.421,
                ".*_hip_z_joint": 54.224,
                ".*_knee_y_joint": 176.421,
            },
            damping={
                ".*_hip_y_joint": 11.231,
                ".*_hip_x_joint": 11.231,
                ".*_hip_z_joint": 3.452,
                ".*_knee_y_joint": 11.231,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_y_joint", ".*_ankle_x_joint"],
            stiffness={
                ".*_ankle_y_joint": 33.493,
                ".*_ankle_x_joint": 21.771,
            },
            damping={
                ".*_ankle_y_joint": 2.132,
                ".*_ankle_x_joint": 1.386,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_y_joint",
                ".*_shoulder_x_joint",
                ".*_shoulder_z_joint",
                ".*_elbow_y_joint",
            ],
            stiffness={
                ".*_shoulder_y_joint": 54.224,
                ".*_shoulder_x_joint": 54.224,
                ".*_shoulder_z_joint": 16.747,
                ".*_elbow_y_joint": 54.224,
            },
            damping={
                ".*_shoulder_y_joint": 3.452,
                ".*_shoulder_x_joint": 3.452,
                ".*_shoulder_z_joint": 1.066,
                ".*_elbow_y_joint": 3.452,
            },
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_x_joint", ".*_wrist_y_joint", ".*_wrist_z_joint"],
            stiffness={
                ".*_wrist_x_joint": 16.747,
                ".*_wrist_y_joint": 16.747,
                ".*_wrist_z_joint": 16.747,
            },
            damping={
                ".*_wrist_x_joint": 1.066,
                ".*_wrist_y_joint": 1.066,
                ".*_wrist_z_joint": 1.066,
            },
        ),
    },
)
