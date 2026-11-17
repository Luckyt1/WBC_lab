"""GL-165 arm disturbance preview using the training IK replay, without a policy."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from legged_lab.assets.gl165.model import (
    ARM_JOINT_NAMES, ASSEMBLY_COLLISION_FILTERS, DEFAULT_POSITIONS,
    INITIAL_BASE_HEIGHT, JOINT_NAMES,
)

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('prepare', 'reach', 'hold', 'return', 'rest')


def build_model(asset):
    """Keep URDF inertias/geometry and the training assembly collision filters."""
    import trimesh

    asset = Path(asset).resolve()
    urdf = asset / 'source/urdf/gl165.urdf'
    metadata = json.loads((asset / 'model.json').read_text())
    digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
    if digest != metadata['training_urdf_sha256']:
        raise ValueError('Training URDF does not match model.json')
    output = asset / 'mujoco'
    output.mkdir(exist_ok=True)
    identity = dict(urdf_sha256=digest, model_sha256=hashlib.sha256(
        (asset / 'model.json').read_bytes()).hexdigest(), mujoco_version=mujoco.__version__, adapter_version=3)
    manifest_path, xml_path = output / 'manifest.json', output / 'gl165_arms.xml'
    if manifest_path.exists() and xml_path.exists():
        previous = json.loads(manifest_path.read_text())
        if all(previous.get(k) == v for k, v in identity.items()):
            if previous['xml_sha256'] == hashlib.sha256(xml_path.read_bytes()).hexdigest():
                return xml_path
    (output / 'meshes').mkdir(exist_ok=True)
    robot = ET.parse(urdf).getroot()
    converted = set()
    for mesh in robot.findall('.//mesh'):
        source = (urdf.parent / mesh.get('filename')).resolve()
        destination = output / 'meshes' / (source.stem + '.obj')
        if source not in converted:
            # The supplied STL files include ASCII meshes; MuJoCo requires binary
            # STL. OBJ also avoids its STL face-count limit without decimation.
            trimesh.load_mesh(source, process=False).export(destination)
            converted.add(source)
        mesh.set('filename', str(destination))
    extension = ET.SubElement(robot, 'mujoco')
    ET.SubElement(extension, 'compiler', discardvisual='false', fusestatic='false',
                  strippath='false', inertiafromgeom='false')
    imported = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding='unicode'))
    with tempfile.TemporaryDirectory() as temp:
        intermediate = Path(temp) / 'imported.xml'
        mujoco.mj_saveLastXML(str(intermediate), imported)
        scene = ET.parse(intermediate).getroot()
    scene.set('model', 'GL165 training arm IK disturbance')
    option = scene.find('option')
    if option is None:
        option = ET.SubElement(scene, 'option')
    option.attrib.update(timestep='0.005', gravity='0 0 -9.81', integrator='implicitfast',
                         iterations='50', cone='elliptic')
    world = scene.find('worldbody')
    base = world.find("body[@name='base_link']")
    if base is None:
        raise ValueError('URDF importer did not preserve base_link')
    # Freeze the root, waist, legs and head at the training rest pose. Bake the
    # hinge angles into body transforms so only the 14 arm joints remain dynamic.
    base.set('pos', f'0 0 {INITIAL_BASE_HEIGHT - .01}')
    for body in world.findall('.//body'):
        for joint in list(body.findall('joint')):
            name = joint.get('name')
            if name in ARM_JOINT_NAMES:
                continue
            if name not in DEFAULT_POSITIONS or np.linalg.norm(np.fromstring(joint.get('pos', '0 0 0'), sep=' ')) > 1e-10:
                raise ValueError(f'Unexpected non-arm joint transform: {name}')
            axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
            half_angle = DEFAULT_POSITIONS[name] / 2
            rotation = np.r_[np.cos(half_angle), np.sin(half_angle) * axis]
            orientation = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
            combined = np.empty(4)
            mujoco.mju_mulQuat(combined, orientation, rotation)
            body.set('quat', ' '.join(map(str, combined)))
            body.remove(joint)
    for joint in scene.findall('.//joint'):
        if joint.get('name') in JOINT_NAMES:
            joint.set('armature', '0.01')
            joint.set('damping', '0')
    for geom in world.findall('.//geom'):
        if geom.get('contype') == '0' and geom.get('conaffinity') == '0':
            geom.set('group', '1')
        else:
            geom.set('group', '3')
            geom.set('friction', '1 0.005 0.0001')
    assets = scene.find('asset')
    ET.SubElement(assets, 'texture', name='ground_grid', type='2d', builtin='checker',
                  rgb1='.20 .23 .27', rgb2='.28 .31 .35', width='512', height='512')
    ET.SubElement(assets, 'material', name='ground_material', texture='ground_grid',
                  texrepeat='8 8', texuniform='true', reflectance='.1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .05',
                  material='ground_material', friction='1 0.005 0.0001', group='0')
    ET.SubElement(world, 'light', pos='1 -2 4', dir='-.2 .3 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'light', pos='-2 2 3', dir='.2 -.2 -1', diffuse='.4 .4 .4')
    visual = ET.SubElement(scene, 'visual')
    ET.SubElement(visual, 'global', azimuth='135', elevation='-15')
    contact = ET.SubElement(scene, 'contact')
    # PhysX filters directly connected URDF links. After fixing the torso, MuJoCo
    # welds it to the world, whose children are exempt from automatic parent
    # filtering. Preserve the original adjacency filters explicitly; otherwise
    # the shoulder motor housings collide with the torso and block arm motion.
    adjacent = [(joint.find('parent').get('link'), joint.find('child').get('link'))
                for joint in robot.findall('joint')]
    for left, right in sorted(set(adjacent + ASSEMBLY_COLLISION_FILTERS)):
        ET.SubElement(contact, 'exclude', body1=left, body2=right)
    actuators = ET.SubElement(scene, 'actuator')
    for name in ARM_JOINT_NAMES:
        joint = metadata['joints'][name]
        effort = joint['limits']['effort']
        ET.SubElement(actuators, 'position', name=name + '_pd', joint=name,
                      kp=str(joint['kp']), kv=str(joint['kd']),
                      forcelimited='true', forcerange=f'{-effort} {effort}')
    ET.indent(scene)
    xml_path.write_text(ET.tostring(scene, encoding='unicode') + '\n')
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if (model.nq, model.nv, model.nu) != (14, 14, 14):
        raise ValueError('Expected 14 dynamic arm joints and a fixed torso/leg pose')
    if abs(model.body_mass.sum() - metadata['total_mass_kg']) > 1e-4:
        raise ValueError('MuJoCo mass differs from source URDF')
    identity.update(total_mass_kg=float(model.body_mass.sum()),
                    xml_sha256=hashlib.sha256(xml_path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(identity, indent=2) + '\n')
    return xml_path


def load_arm_motion(bank):
    # Loading the pure replay file avoids legged_lab.envs.__init__, which starts
    # Isaac-dependent task registration. GL165 uses its ungated path generator.
    manifest = json.loads((Path(bank) / 'manifest.json').read_text())
    continuous = manifest['recipe'].get('motion_mode') == 'continuous_targets_v1'
    source = 'legged_lab/envs/gl165/arm_motion.py' if continuous else 'legged_lab/envs/elf3_tang/wbc/ik_arm_motion.py'
    spec = importlib.util.spec_from_file_location('gl165_ik_replay', ROOT / source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ContinuousIkArmMotion if continuous else module.IkArmMotion


class Gl165ArmPreview:
    def __init__(self, asset, bank, arm_group='both', seed=42, target_region='all'):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.xml_path = build_model(asset)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.dt, self.decimation = .02, 4
        self.joint_ids = np.array([self.model.joint(n).id for n in ARM_JOINT_NAMES])
        self.qids = self.model.jnt_qposadr[self.joint_ids]
        self.defaults = np.array([DEFAULT_POSITIONS[n] for n in ARM_JOINT_NAMES])
        self.torso = self.model.body('body_y_link').id
        self.wrists = [self.model.body(n).id for n in ('lwrist_y_link', 'rwrist_y_link')]
        defaults = torch.tensor(self.defaults[None], dtype=torch.float32)
        self.arms = load_arm_motion(bank)(bank, 'train', ARM_JOINT_NAMES, defaults, self.dt)
        self.continuous = self.arms.manifest['recipe'].get('motion_mode') == 'continuous_targets_v1'
        metadata = json.loads((Path(asset) / 'model.json').read_text())
        if self.arms.manifest['training_urdf_sha256'] != metadata['training_urdf_sha256']:
            raise ValueError('Arm bank and robot do not match')
        # Legacy previews skip neutral cycles; continuous banks already have none.
        self.arms.manifest['recipe']['neutral_fraction'] = 0.
        if arm_group != 'all':
            group = self.arms.group_names.index(arm_group)
            self.arms.weights *= (self.arms.groups == group)
        if target_region != 'all':
            regions = {t['id']: t.get('region', 'low') for t in self.arms.manifest['targets']}
            self.arms.weights *= torch.tensor([regions[i] == target_region for i in self.arms.ids])
        if self.arms.weights.sum() <= 0:
            raise ValueError('No targets match the selected arm group and height region')
        self.arms.weights /= self.arms.weights.sum()
        self.reset()

    @torch.inference_mode()
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = self.defaults
        self.data.ctrl[:] = self.defaults
        mujoco.mj_forward(self.model, self.data)
        self.arm_target = self.arms.defaults.clone()
        self.arms.reset(torch.tensor([0]), self.arm_target)
        self.elapsed = 0.
        self.max_penetration = 0.
        self.peak_torque_fraction = 0.

    @torch.inference_mode()
    def step(self):
        requested = self.arms.step()
        delta = self.dt * self.arms.manifest['recipe']['max_reference_velocity_rad_s']
        self.arm_target += (requested - self.arm_target).clamp(-delta, delta)
        bounds = torch.from_numpy(self.model.jnt_range[self.joint_ids])
        self.arm_target.clamp_(bounds[:, 0], bounds[:, 1])
        self.arms.target.copy_(self.arm_target)
        self.data.ctrl[:] = self.arm_target.numpy()[0]
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
            self.max_penetration = max(self.max_penetration,
                max((max(0., -c.dist) for c in self.data.contact), default=0.))
            self.peak_torque_fraction = max(self.peak_torque_fraction, float(np.max(
                np.abs(self.data.actuator_force) / self.model.actuator_forcerange[:, 1])))
        self.elapsed += self.dt
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Non-finite MuJoCo arm state')

    def target_positions(self):
        local = self.arms.goal_positions[self.arms.index].numpy()[0]
        return local @ self.data.xmat[self.torso].reshape(3, 3).T + self.data.xpos[self.torso]

    def metrics(self):
        phase = int(self.arms.phase[0])
        return dict(time_s=round(self.elapsed, 2), phase=(('endpoint' if phase == 2 else 'transition')
                    if self.continuous else PHASES[phase]),
                    target_id=self.arms.ids[int(self.arms.index[0])],
                    target_region=(self.arms.region_names[int(self.arms.regions[self.arms.index[0]])]
                                   if self.continuous else 'low'),
                    completed_cycles=int(self.arms.completed_cycles),
                    max_collision_penetration_m=self.max_penetration,
                    peak_torque_fraction=self.peak_torque_fraction,
                    arm_tracking_error_rad=float(np.abs(self.data.qpos[self.qids] - self.arm_target.numpy()[0]).mean()),
                    wrist_position_error_m=np.linalg.norm(self.data.xpos[self.wrists] - self.target_positions(), axis=1).tolist())
