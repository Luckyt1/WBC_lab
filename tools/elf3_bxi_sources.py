"""Read the deployment's named parameters without importing ROS or running its nodes."""
import ast
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


def assignment(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise ValueError(f'{name} is missing from {path}')


def read_sources(repository):
    repository = Path(repository).resolve()
    package = repository / 'src/bxi_example_py_elf3'
    urdf_path = repository / 'resources/elf3_dof31/urdf/elf3.urdf'
    parameters_path = package / 'bxi_example_py_elf3/policies/joints.py'
    defaults_path = package / 'bxi_example_py_elf3/bxi_example_demo.py'
    mjcf_path = package / 'data/mujoco_simulation/elf3.xml'
    runtime_path = package / 'config/elf3_state_machine.yaml'
    order = list(ast.literal_eval(assignment(parameters_path, 'ELF3_POLICY_JOINTS').args[0]))
    rows = ast.literal_eval(assignment(parameters_path, 'ELF3_ISAAC_PARAMETERS').args[1])
    parameters = {n: dict(default_position=q, kp=kp, kd=kd, action_scale=scale) for n, q, kp, kd, scale in rows}
    defaults = assignment(defaults_path, 'ELF3_COMMAND_DEFAULTS').args[0]
    for name_node, call in zip(defaults.keys, defaults.values):
        name = ast.literal_eval(name_node)
        values = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        parameters[name] = dict(default_position=values['position'], kp=values['kp'], kd=values['kd'], action_scale=0.0)
    reset_order = list(ast.literal_eval(assignment(defaults_path, 'ELF3_RESET_JOINTS').args[0]))
    assert len(order) == 29 and reset_order == order + ['head_z_joint', 'head_y_joint']
    urdf = ET.parse(urdf_path).getroot()
    joints = {j.get('name'): j for j in urdf.findall('joint') if j.get('type') == 'revolute'}
    assert set(joints) == set(parameters) == set(reset_order)
    mjcf = ET.parse(mjcf_path).getroot()
    mj_joints = {j.get('name'): j for j in mjcf.findall('.//worldbody//joint') if j.get('name') in joints}
    motors = {j.get('joint'): j for j in mjcf.findall('actuator/*')}
    assert set(motors) == set(joints)
    effort_differences = []
    for name in reset_order:
        joint = joints[name]
        limits = {k: float(v) for k, v in joint.find('limit').attrib.items()}
        mj_joint, motor = mj_joints[name], motors[name]
        assert motor.tag == 'motor' and float(motor.get('gear', '1')) == 1
        motor_range = [float(v) for v in motor.get('ctrlrange').split()]
        body_range = [float(v) for v in mj_joint.get('actuatorfrcrange').split()]
        assert motor_range[0] == -motor_range[1] and body_range[0] == -body_range[1]
        deployment_effort = min(motor_range[1], body_range[1])
        assert [float(v) for v in mj_joint.get('range').split()] == [limits['lower'], limits['upper']], name
        assert [float(v) for v in mj_joint.get('axis').split()] == [float(v) for v in joint.find('axis').get('xyz').split()], name
        parameters[name].update(limits=limits, deployment_effort=deployment_effort,
                                armature=float(mj_joint.get('armature', 0)))
        if deployment_effort != limits['effort']:
            effort_differences.append(dict(joint=name, urdf=limits['effort'], deployment=deployment_effort))
    files = [urdf_path, parameters_path, defaults_path, mjcf_path, runtime_path]
    meshes = sorted({(urdf_path.parent / m.get('filename')).resolve() for m in urdf.findall('.//mesh')})
    assert all(p.is_file() and p.is_relative_to(repository) for p in meshes)
    files += meshes
    hashes = {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return dict(source_repository=str(repository), source_hashes=hashes,
                urdf_relative_path=str(urdf_path.relative_to(repository)),
                policy_joint_names=order[:15], arm_joint_names=order[15:],
                held_joint_names=reset_order[29:], deployment_joint_names=reset_order,
                joints=parameters, effort_differences=effort_differences,
                control_period_s=yaml.safe_load(runtime_path.read_text())['control_runtime']['period_sec'],
                source_mass_kg=sum(float(l.find('inertial/mass').get('value')) for l in urdf.findall('link') if l.find('inertial/mass') is not None),
                source_root_position=[float(v) for v in mjcf.find('worldbody/body').get('pos').split()])
