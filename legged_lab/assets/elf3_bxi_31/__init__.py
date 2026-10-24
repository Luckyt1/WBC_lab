"""BXI 31-joint asset and controller parameters, frozen from deployment sources."""
import json
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from legged_lab.assets.elf3_lite import ELF3LITE_CFG
from legged_lab.paths import robot_asset_dir
from .ankle_limits import joint_velocity_limits

ASSET_DIR = robot_asset_dir()
PARAMETERS = json.loads((Path(__file__).resolve().parent / 'control_parameters.json').read_text())
JOINTS = PARAMETERS['joints']

ELF3_BXI_CFG = ELF3LITE_CFG.replace(
    spawn=ELF3LITE_CFG.spawn.replace(usd_path=str(ASSET_DIR / 'usd/elf3.usd')),
    init_state=ELF3LITE_CFG.init_state.replace(
        pos=tuple(PARAMETERS['source_root_position']),
        joint_pos={name: values['default_position'] for name, values in JOINTS.items()},
    ),
    actuators={
        'deployment': ImplicitActuatorCfg(
            joint_names_expr=PARAMETERS['deployment_joint_names'],
            effort_limit_sim={name: values['effort_limit'] for name, values in JOINTS.items()},
            velocity_limit_sim=joint_velocity_limits(JOINTS),
            stiffness={name: values['kp'] for name, values in JOINTS.items()},
            damping={name: values['kd'] for name, values in JOINTS.items()},
            armature={name: values['armature'] for name, values in JOINTS.items()},
        ),
    },
)
