"""Isaac articulation using the converted URDF's masses, joints and limits."""

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .model import ASSEMBLY_COLLISION_FILTERS, CONTROL_PARAMETERS, DEFAULT_POSITIONS, INITIAL_BASE_HEIGHT, JOINT_NAMES


def make_robot_cfg(asset, *, joint_gains=None):
    asset = Path(asset)
    if not (asset / "usd/gl165.usd").is_file():
        raise FileNotFoundError(f"GL-165 asset missing: {asset}. Run tools/convert_gl165.py first.")
    model = json.loads((asset / "model.json").read_text())
    if set(model['joints']) != set(JOINT_NAMES) or model.get('fix_base') is not False:
        raise ValueError("GL-165 asset must contain all 31 joints and a floating base")
    if model.get('assembly_collision_filters') != [list(pair) for pair in ASSEMBLY_COLLISION_FILTERS]:
        raise ValueError("GL-165 asset lacks the required nested-joint collision filters; reconvert it")
    joints = model['joints']
    # A checkpoint supplies its saved gains; fresh training uses the shared ELF3 table.
    gains = CONTROL_PARAMETERS if joint_gains is None else joint_gains
    if set(gains) != set(JOINT_NAMES):
        raise ValueError('Controller gains must identify all 31 GL-165 joints')
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset / "usd/gl165.usd"), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=100.0, max_angular_velocity=100.0, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, INITIAL_BASE_HEIGHT), joint_pos=DEFAULT_POSITIONS, joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=.95,
        actuators={"joints": ImplicitActuatorCfg(
            joint_names_expr=JOINT_NAMES,
            effort_limit_sim={n: j['limits']['effort'] for n, j in joints.items()},
            velocity_limit_sim={n: j['limits']['velocity'] for n, j in joints.items()},
            stiffness={n: j['kp'] for n, j in gains.items()},
            damping={n: j['kd'] for n, j in gains.items()},
            armature=.01)},
    )
