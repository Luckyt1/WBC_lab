"""Named GL-165 contract, usable without Isaac Sim.

PD gains and action scales share the ELF3 controller's named parameter table,
as requested by the user. Effort and speed limits come from the GL-165 URDF.
"""

import hashlib
import json
import math
import os
from pathlib import Path

POLICY_JOINT_NAMES = ["waist_z_joint", "waist_x_joint", "waist_y_joint"] + [
    f"{side}{joint}_joint"
    for side in ("l", "r")
    for joint in ("hip_y", "hip_x", "hip_z", "knee_y", "ankle_y", "ankle_x")
]
ARM_JOINT_NAMES = [
    f"{side}{joint}_joint"
    for side in ("l", "r")
    for joint in ("shouder_y", "shouder_x", "shouder_z", "elbow_y", "wrist_x", "wrist_z", "wrist_y")
]
HEAD_JOINT_NAMES = ["head_z_joint", "head_y_joint"]
JOINT_NAMES = POLICY_JOINT_NAMES + ARM_JOINT_NAMES + HEAD_JOINT_NAMES
ELF3_CONTROL_FILE = Path(__file__).resolve().parents[1] / 'elf3_bxi_31/control_parameters.json'
ELF3_JOINT_NAMES = {
    name: (name[0] + '_' + name[1:].replace('shouder', 'shoulder')
           if name[0] in ('l', 'r') else name)
    for name in JOINT_NAMES
}
_elf3_control_bytes = ELF3_CONTROL_FILE.read_bytes()
_elf3_controls = json.loads(_elf3_control_bytes)['joints']
if len(set(ELF3_JOINT_NAMES.values())) != len(JOINT_NAMES) or set(ELF3_JOINT_NAMES.values()) != set(_elf3_controls):
    raise ValueError('GL-165 / ELF3 controller joint mapping is incomplete or ambiguous')
CONTROL_PARAMETERS = {
    name: {field: float(_elf3_controls[source][field]) for field in ('kp', 'kd', 'action_scale')}
    for name, source in ELF3_JOINT_NAMES.items()
}
for _name, _values in CONTROL_PARAMETERS.items():
    if (not all(math.isfinite(v) for v in _values.values()) or _values['kp'] <= 0 or _values['kd'] < 0
            or _values['action_scale'] < 0 or (_name in POLICY_JOINT_NAMES and _values['action_scale'] == 0)):
        raise ValueError(f'Invalid ELF3 controller parameters for {_name}; provide verified values')
CONTROL_SOURCE = {
    'file': 'legged_lab/assets/elf3_bxi_31/control_parameters.json',
    'sha256': hashlib.sha256(_elf3_control_bytes).hexdigest(),
    'joint_mapping': ELF3_JOINT_NAMES,
}
FEET_BODY_NAMES = ["lfoot_x_link", "rfoot_x_link"]
# CAD housings overlap across these two-axis assemblies. Directly connected
# links are already filtered by PhysX; these pairs span the small intermediate link.
ASSEMBLY_COLLISION_FILTERS = [
    ("waist_z_link", "body_y_link"),
    ("lsmallleg_y_link", "lfoot_x_link"),
    ("rsmallleg_y_link", "rfoot_x_link"),
    ("lsmallarm_x_link", "lwrist_y_link"),
    ("rsmallarm_x_link", "rwrist_y_link"),
]
DEFAULT_POSITIONS = dict.fromkeys(JOINT_NAMES, 0.0)
# FK solution for horizontal feet at (0.01, +/-0.105, -0.948) m in base_link.
# The +/-20 degree hip mounting angles require coupled roll/yaw compensation.
for side, sign in (("l", 1), ("r", -1)):
    DEFAULT_POSITIONS.update({
        f"{side}hip_y_joint": -0.3771778,
        f"{side}hip_x_joint": sign * 0.0075206,
        f"{side}hip_z_joint": -sign * 0.1374135,
        f"{side}knee_y_joint": 0.6501734,
        f"{side}ankle_y_joint": -0.2977962,
        f"{side}ankle_x_joint": -sign * 0.0320223,
        f"{side}shouder_x_joint": sign * 0.15,
        f"{side}elbow_y_joint": 0.8,
    })
INITIAL_BASE_HEIGHT = 0.99  # Foot soles start 1 cm above the plane; no height command.


def pd_gains(name):
    values = CONTROL_PARAMETERS[name]
    return values['kp'], values['kd']


def policy_action_scales():
    return [CONTROL_PARAMETERS[name]['action_scale'] for name in POLICY_JOINT_NAMES]


def asset_dir():
    root = Path(__file__).resolve().parents[3]
    return Path(os.environ.get("GL165_ASSET_DIR", root / "resources/gl165")).expanduser().resolve()


def ik_bank_dir():
    if os.environ.get('GL165_IK_BANK'):
        return Path(os.environ['GL165_IK_BANK']).expanduser().resolve()
    asset = asset_dir()
    active = asset / 'active_ik_bank.json'
    if active.exists():
        return (asset / json.loads(active.read_text())['path']).resolve()
    return asset / 'ik_bank'
