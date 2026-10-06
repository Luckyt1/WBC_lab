"""Numerically check the generated USD against frozen URDF and deployment sources."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]
from isaaclab.app import AppLauncher
from legged_lab.paths import robot_asset_dir

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', default=str(robot_asset_dir()))
parser.add_argument('--legacy-usd', type=Path, help='Optional older USD for a mass/joint comparison')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysxSchema
from elf3_bxi_sources import read_sources


def vector(element, key):
    return np.fromstring(element.get(key, '0 0 0') if element is not None else '0 0 0', sep=' ')


def qmat(quaternion):
    return Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()


def transform(position, rotation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def rigid_bodies(stage):
    return {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}


def revolute_joints(stage):
    return {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}


def main():
    root = Path(args.asset).resolve()
    manifest = json.loads((root / 'conversion_manifest.json').read_text())
    parameters = json.loads((root / 'control_parameters.json').read_text())
    original_parameters = read_sources(root / 'source')
    assert original_parameters['source_hashes'] == parameters['source_hashes']
    for name, digest in parameters['source_hashes'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == digest, name
    urdf = ET.parse(root / 'source' / parameters['urdf_relative_path']).getroot()
    links = {l.get('name'): l for l in urdf.findall('link') if l.find('inertial') is not None}
    joints = {j.get('name'): j for j in urdf.findall('joint') if j.get('type') == 'revolute'}
    stage = Usd.Stage.Open(str(root / 'usd/elf3.usd'))
    bare = Usd.Stage.Open(str(root / 'usd/elf3_urdf.usd'))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1 and UsdGeom.GetStageUpAxis(stage) == 'Z'
    bodies, usd_joints = rigid_bodies(stage), revolute_joints(stage)
    assert set(bodies) == set(links) and len(bodies) == 32
    assert set(usd_joints) == set(joints) and len(usd_joints) == 31
    assert not any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())
    link_checks, joint_checks = [], []
    for name, link in links.items():
        inertial = link.find('inertial')
        values = {k: float(v) for k, v in inertial.find('inertia').attrib.items()}
        matrix = np.array([[values['ixx'], values['ixy'], values['ixz']],
                           [values['ixy'], values['iyy'], values['iyz']],
                           [values['ixz'], values['iyz'], values['izz']]])
        rotation = Rotation.from_euler('xyz', vector(inertial.find('origin'), 'rpy')).as_matrix()
        expected_inertia = rotation @ matrix @ rotation.T
        mass = UsdPhysics.MassAPI(bodies[name])
        principal = qmat(mass.GetPrincipalAxesAttr().Get())
        actual_inertia = principal @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ principal.T
        mass_error = abs(float(inertial.find('mass').get('value')) - mass.GetMassAttr().Get())
        com_error = np.max(np.abs(vector(inertial.find('origin'), 'xyz') - np.array(mass.GetCenterOfMassAttr().Get())))
        assert mass_error < 1e-6 and com_error < 1e-6, name
        assert np.allclose(expected_inertia, actual_inertia, atol=1e-8, rtol=1e-4), name
        assert np.linalg.eigvalsh(actual_inertia).min() > 0, name
        link_checks.append(dict(name=name, mass_error_kg=mass_error, com_error_m=float(com_error),
                                inertia_error=float(np.abs(expected_inertia - actual_inertia).max())))
    for name, joint in joints.items():
        prim = usd_joints[name]
        api = UsdPhysics.RevoluteJoint(prim)
        t0 = transform(api.GetLocalPos0Attr().Get(), qmat(api.GetLocalRot0Attr().Get()))
        t1 = transform(api.GetLocalPos1Attr().Get(), qmat(api.GetLocalRot1Attr().Get()))
        actual = t0 @ np.linalg.inv(t1)
        expected = transform(vector(joint.find('origin'), 'xyz'), Rotation.from_euler('xyz', vector(joint.find('origin'), 'rpy')).as_matrix())
        assert api.GetBody0Rel().GetTargets()[0].name == joint.find('parent').get('link'), name
        assert api.GetBody1Rel().GetTargets()[0].name == joint.find('child').get('link'), name
        assert np.allclose(actual, expected, atol=1e-6, rtol=0), name
        axis = t1[:3, :3] @ np.eye(3)['XYZ'.index(api.GetAxisAttr().Get())]
        assert np.allclose(axis, vector(joint.find('axis'), 'xyz'), atol=1e-6, rtol=0), name
        values = parameters['joints'][name]
        original = original_parameters['joints'][name]
        for key in ('default_position', 'kp', 'kd', 'action_scale', 'armature', 'deployment_effort', 'limits'):
            assert values[key] == original[key], (name, key)
        limits = np.radians([api.GetLowerLimitAttr().Get(), api.GetUpperLimitAttr().Get()])
        assert np.allclose(limits, [values['limits']['lower'], values['limits']['upper']], atol=1e-5), name
        drive = UsdPhysics.DriveAPI.Get(prim, 'angular')
        assert abs(drive.GetMaxForceAttr().Get() - values['effort_limit']) < 1e-6, name
        assert math.isclose(drive.GetStiffnessAttr().Get() * 180 / math.pi, values['kp'], abs_tol=1e-5), name
        assert math.isclose(drive.GetDampingAttr().Get() * 180 / math.pi, values['kd'], abs_tol=1e-5), name
        assert math.isclose(math.radians(drive.GetTargetPositionAttr().Get()), values['default_position'], abs_tol=1e-6), name
        physx = PhysxSchema.PhysxJointAPI(prim)
        assert math.isclose(physx.GetArmatureAttr().Get(), values['armature'], abs_tol=1e-8), name
        assert math.isclose(math.radians(physx.GetMaxJointVelocityAttr().Get()), values['limits']['velocity'], abs_tol=1e-5), name
        bare_prim = revolute_joints(bare)[name]
        bare_drive = UsdPhysics.DriveAPI.Get(bare_prim, 'angular')
        assert bare_drive.GetMaxForceAttr().Get() == values['limits']['effort'], name
        assert bare_drive.GetStiffnessAttr().Get() == bare_drive.GetDampingAttr().Get() == 0, name
        joint_checks.append(dict(name=name, lower=float(limits[0]), upper=float(limits[1]),
                                 urdf_effort=values['limits']['effort'], selected_effort=values['effort_limit'],
                                 kp=values['kp'], kd=values['kd'], armature=values['armature']))
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(root / 'usd/elf3.usd'))
    assert set(unresolved) <= {'OmniPBR.mdl'}, unresolved
    assert all(Path(layer.realPath).resolve().is_relative_to(root) for layer in layers)
    # Check which rigid bodies own collisions, including instance-proxy meshes.
    collision_bodies = set()
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            parent = prim
            while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                parent = parent.GetParent()
            assert parent, prim.GetPath()
            collision_bodies.add(parent.GetName())
    expected_collisions = {l.get('name') for l in urdf.findall('link') if l.find('collision') is not None}
    assert collision_bodies == expected_collisions, (collision_bodies, expected_collisions)
    legacy = None
    if args.legacy_usd:
        old = Usd.Stage.Open(str(args.legacy_usd))
        old_bodies, old_joints = rigid_bodies(old), revolute_joints(old)
        differences = []
        for name in sorted(set(old_bodies) & set(bodies)):
            old_mass, new_mass = UsdPhysics.MassAPI(old_bodies[name]), UsdPhysics.MassAPI(bodies[name])
            if old_mass.GetMassAttr().Get() != new_mass.GetMassAttr().Get():
                differences.append(dict(body=name, old_mass=old_mass.GetMassAttr().Get(), new_mass=new_mass.GetMassAttr().Get()))
        legacy = dict(joints=len(old_joints), bodies=len(old_bodies),
                      mass_kg=sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in old_bodies.values()),
                      added_joints=sorted(set(usd_joints) - set(old_joints)), mass_differences=differences)
    summary = dict(status='PASS', joints=31, bodies=32, mass_kg=sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in bodies.values()),
                   max_mass_error_kg=max(x['mass_error_kg'] for x in link_checks),
                   max_com_error_m=max(x['com_error_m'] for x in link_checks),
                   max_inertia_error=max(x['inertia_error'] for x in link_checks),
                   collision_bodies=sorted(collision_bodies), usd_layers=len(layers), builtin_dependencies=unresolved,
                   effort_source=parameters['effort_source'],
                   legacy=legacy)
    (root / 'verification.json').write_text(json.dumps(dict(summary=summary, links=link_checks, joints=joint_checks), indent=2) + '\n')
    print('ELF3_BXI_VERIFICATION', json.dumps(summary), flush=True)


from legged_lab.utils.app_lifecycle import run_simulation
run_simulation(main, app, args.headless)
