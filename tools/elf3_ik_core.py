"""Load the pinned deployment IK without importing ROS or starting hardware."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legged_lab.paths import ik_source_dir

SOURCE = ik_source_dir()
PROVENANCE = ROOT / 'legged_lab/assets/elf3_ik_upstream/PROVENANCE.json'


def source_module():
    provenance = json.loads(PROVENANCE.read_text())
    for name, digest in provenance['sha256'].items():
        if not (SOURCE/name).is_file():
            raise FileNotFoundError(f'Missing {SOURCE/name}; run scripts/prepare_ik_source.py --checkout PATH')
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Pinned IK source changed: {name}')
    path = SOURCE/'arm_bridge.py'
    tree = ast.parse(path.read_text())
    names = {'ARM_JOINTS', 'DEFAULTS', '_rotation_from_wxyz', 'ArmIkResult', 'PinocchioArmIk'}
    selected = []
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module in {'__future__', 'dataclasses'}:
            selected.append(n)
        elif isinstance(n, ast.Import) and all(a.name in {'math', 'numpy', 'pinocchio'} for a in n.names):
            selected.append(n)
        elif isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names:
            selected.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            selected.append(n)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in names:
            selected.append(n)
        elif isinstance(n, ast.ClassDef) and n.name == 'ArmIkBridgeNode':
            method = next(m for m in n.body if isinstance(m, ast.FunctionDef) and m.name == '_rate_limited_arm_reference')
            selected.append(ast.ClassDef(name='OriginalRateLimiter', bases=[], keywords=[], body=[method], decorator_list=[]))
    module = types.ModuleType('pinned_elf3_ik')
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), 'exec'), module.__dict__)
    return module


def new_solver():
    import yaml
    module = source_module()
    manifest = yaml.safe_load((SOURCE/'mod.yaml').read_text())
    argv = next(v['arm_ik_bridge']['arguments'] for v in manifest.values()
                if isinstance(v, dict) and 'arm_ik_bridge' in v)
    p = dict(module.DEFAULTS)
    for key, value in zip(argv[::2], argv[1::2]):
        key = key[2:].replace('-', '_')
        if key in p and isinstance(p[key], (int, float)):
            p[key] = type(p[key])(value)
    solver = module.PinocchioArmIk(str(SOURCE/'elf3-dof31.urdf'),
        iterations=p['ik_iterations'], damping=p['ik_damping'], step_size=p['ik_step_size'],
        tolerance=p['ik_tolerance'], maximum_joint_step=p['maximum_joint_step_rad'],
        maximum_position_error=p['maximum_position_error_m'], maximum_orientation_error=p['maximum_orientation_error_rad'],
        joint_limit_margin=p['joint_limit_margin_rad'], joint_centering_gain=p['joint_centering_gain'],
        swivel_continuity_gain=p['swivel_continuity_gain'], swivel_min_radius=p['swivel_min_radius_m'])
    limiter = module.OriginalRateLimiter()
    limiter._solver = solver
    limiter._maximum_reference_step = solver.maximum_joint_step
    limiter._last_published_arm_positions = None
    return module, solver, limiter, p


def forward_targets(solver, arms):
    import numpy as np
    import pinocchio as pin
    from scipy.spatial.transform import Rotation
    q = pin.neutral(solver.model)
    q[solver._arm_q_indices] = arms
    pin.framesForwardKinematics(solver.model, solver.data, q)
    anchor = solver.data.oMf[solver._anchor_frame]
    frames = [anchor.inverse()*solver.data.oMf[solver._wrist_frames[s]] for s in ('left', 'right')]
    return (np.array([f.translation for f in frames]),
            np.array([Rotation.from_matrix(f.rotation).as_quat()[[3, 0, 1, 2]] for f in frames]))
