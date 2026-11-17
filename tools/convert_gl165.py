"""Convert the supplied GL-165 URDF to an independent floating-base Isaac asset."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "rsl_rl")]

from legged_lab.assets.gl165.model import ASSEMBLY_COLLISION_FILTERS, asset_dir
from tools.gl165_sources import prepare_source, sha256


def filter_assembly_collisions(usd_path):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(usd_path))
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    for left, right in ASSEMBLY_COLLISION_FILTERS:
        if left not in bodies or right not in bodies:
            raise ValueError(f"Missing collision-filter body: {left}, {right}")
        UsdPhysics.FilteredPairsAPI.Apply(bodies[left]).CreateFilteredPairsRel().AddTarget(bodies[right].GetPath())
    stage.GetRootLayer().Save()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=asset_dir())
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    urdf, manifest = prepare_source(args.source, args.output.resolve())
    app = AppLauncher(args).app

    def convert():
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from pxr import Usd, UsdPhysics

        class InertiaConverter(UrdfConverter):
            def _get_urdf_import_config(self):
                config = super()._get_urdf_import_config()
                config.set_import_inertia_tensor(True)
                return config

        converter = InertiaConverter(UrdfConverterCfg(
            asset_path=str(urdf), usd_dir=str(args.output.resolve() / "usd"), usd_file_name="gl165.usd",
            force_usd_conversion=True, make_instanceable=True, fix_base=False,
            merge_fixed_joints=True, link_density=0.0, collider_type="convex_hull", self_collision=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force", target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
        ))
        filter_assembly_collisions(converter.usd_path)
        stage = Usd.Stage.Open(converter.usd_path)
        joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
        if {p.GetName() for p in joints} != set(manifest['joints']):
            raise RuntimeError("Converted USD joint set differs from URDF")
        mass = sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0 for p in stage.Traverse()
                   if p.HasAPI(UsdPhysics.MassAPI))
        if abs(mass - manifest['total_mass_kg']) > 1e-4:
            raise RuntimeError(f"USD mass mismatch: {mass}")
        manifest.update(import_inertia_tensor=True, fix_base=False, collider_type="convex_hull",
                        assembly_collision_filters=ASSEMBLY_COLLISION_FILTERS,
                        usd_total_mass_kg=mass, usd_sha256=sha256(converter.usd_path))
        (args.output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("GL165_CONVERSION_COMPLETE", converter.usd_path, flush=True)

    from legged_lab.utils.app_lifecycle import run_simulation
    run_simulation(convert, app, args.headless)


if __name__ == "__main__":
    main()
