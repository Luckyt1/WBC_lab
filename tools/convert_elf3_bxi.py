"""Snapshot BXI sources, import their URDF inertia, and author a deployment control layer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]
from elf3_bxi_sources import read_sources
from legged_lab.paths import robot_asset_dir
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', required=True, help='Local bxi_controller_ros2 checkout')
parser.add_argument('--output', default=str(robot_asset_dir()))
parser.add_argument('--effort-source', choices=('deployment', 'urdf'), default='deployment')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output).resolve()
if (output / 'usd/elf3.usd').exists() or (output / 'conversion_manifest.json').exists():
    parser.error('Output asset already exists; use a new output directory to preserve its version.')
parameters = read_sources(args.source)
launcher = AppLauncher(args)
app = launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


class InertiaConverter(UrdfConverter):
    def _get_urdf_import_config(self):
        cfg = super()._get_urdf_import_config()
        cfg.set_import_inertia_tensor(True)
        return cfg


def main():
    source = Path(parameters['source_repository'])
    # Keep a portable byte-identical input snapshot, including every referenced mesh.
    for relative, digest in parameters['source_hashes'].items():
        original = source / relative
        assert hashlib.sha256(original.read_bytes()).hexdigest() == digest, relative
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    urdf = output / 'source' / parameters['urdf_relative_path']
    cfg = UrdfConverterCfg(
        asset_path=str(urdf), usd_dir=str(output / 'usd'), usd_file_name='elf3_urdf.usd',
        force_usd_conversion=True, make_instanceable=True, fix_base=False,
        merge_fixed_joints=True, link_density=0.0, collider_type='convex_hull', self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type='force', target_type='position',
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
    )
    converter = InertiaConverter(cfg)
    stage = Usd.Stage.CreateNew(str(output / 'usd/elf3.usd'))
    stage.GetRootLayer().subLayerPaths = ['./elf3_urdf.usd']
    source_stage = Usd.Stage.Open(converter.usd_path)
    # Stage units/up-axis must be authored on this root layer, not just a sublayer.
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    stage.SetDefaultPrim(stage.GetPrimAtPath(source_stage.GetDefaultPrim().GetPath()))
    parameters['effort_source'] = args.effort_source
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        values = parameters['joints'][prim.GetName()]
        effort = values['deployment_effort'] if args.effort_source == 'deployment' else values['limits']['effort']
        values['effort_limit'] = effort
        drive = UsdPhysics.DriveAPI.Get(prim, 'angular')
        # USD angular drives use degrees; the Python actuator config uses radians.
        drive.CreateStiffnessAttr(values['kp'] * math.pi / 180)
        drive.CreateDampingAttr(values['kd'] * math.pi / 180)
        drive.CreateMaxForceAttr(effort)
        drive.CreateTargetPositionAttr(math.degrees(values['default_position']))
        PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr(values['armature'])
    stage.GetRootLayer().Save()
    (output / 'control_parameters.json').write_text(json.dumps(parameters, indent=2, ensure_ascii=False) + '\n')
    manifest = dict(converted_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_repository=str(source), source_hashes=parameters['source_hashes'],
                    import_inertia_tensor=True, merge_fixed_joints=True, collider_type='convex_hull',
                    bare_urdf_usd='usd/elf3_urdf.usd', deployment_usd='usd/elf3.usd',
                    effort_source=args.effort_source, effort_differences=parameters['effort_differences'],
                    note='Geometry, mass, inertia, joint axes/origins/position and velocity limits from URDF. Named PD/defaults/action scale from ISAAC deployment policy and head defaults. Armature from active MuJoCo joint definitions. Control-period metadata from deployed runtime. Existing task and USD remain unchanged.',
                    usd_sha256={str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (output / 'usd').rglob('*.usd')})
    (output / 'conversion_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    for relative, digest in parameters['source_hashes'].items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == digest, relative
    print('ELF3_BXI_CONVERSION_COMPLETE', output / 'usd/elf3.usd', flush=True)


from legged_lab.utils.app_lifecycle import run_simulation
run_simulation(main, app, args.headless)
