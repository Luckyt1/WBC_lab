"""Prepare a portable floating-base GL-165 URDF without modifying its source."""

import hashlib
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from legged_lab.assets.gl165.model import CONTROL_PARAMETERS, CONTROL_SOURCE, DEFAULT_POSITIONS, JOINT_NAMES, pd_gains


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_source(source):
    source = Path(source).expanduser().resolve()
    urdf = source / "gl_165.urdf" if source.is_dir() else source
    root = ET.parse(urdf).getroot()
    joints = {j.attrib["name"]: j for j in root.findall("joint") if j.attrib["type"] != "fixed"}
    if set(joints) != set(JOINT_NAMES):
        raise ValueError(f"Unexpected GL-165 joint layout: {set(joints) ^ set(JOINT_NAMES)}")
    ground = root.find("link[@name='ground_link']")
    fixed = root.find("joint[@name='ground_joint']")
    if ground is None or fixed is None or fixed.get("type") != "fixed":
        raise ValueError("Expected the original ground_link / ground_joint export")
    if (fixed.find("parent").get("link"), fixed.find("child").get("link")) != ("ground_link", "base_link"):
        raise ValueError("ground_joint must connect ground_link to base_link")
    if len(list(ground)) or sum(j.find("parent").get("link") == "ground_link" for j in root.findall("joint")) != 1:
        raise ValueError("Only an empty display root can be removed")
    total_mass = 0.0
    for link in root.findall("link"):
        if link is ground:
            continue
        mass = float(link.find("inertial/mass").get("value"))
        v = {k: float(v) for k, v in link.find("inertial/inertia").attrib.items()}
        inertia = np.array([[v['ixx'], v['ixy'], v['ixz']], [v['ixy'], v['iyy'], v['iyz']],
                            [v['ixz'], v['iyz'], v['izz']]])
        eig = np.linalg.eigvalsh(inertia)
        if not np.isfinite(eig).all() or mass <= 0 or eig[0] <= 0 or eig[2] > eig[0] + eig[1] + 1e-9:
            raise ValueError(f"Invalid mass/inertia: {link.get('name')}")
        total_mass += mass
    parameters = {}
    for name, joint in joints.items():
        limits = {k: float(joint.find("limit").get(k)) for k in ("lower", "upper", "effort", "velocity")}
        if (not all(np.isfinite(list(limits.values()))) or limits['effort'] <= 0 or limits['velocity'] <= 0
                or not limits['lower'] < DEFAULT_POSITIONS[name] < limits['upper']):
            raise ValueError(f"Invalid limits/default pose: {name}")
        kp, kd = pd_gains(name)
        parameters[name] = dict(limits=limits, default_position=DEFAULT_POSITIONS[name], kp=kp, kd=kd,
                                action_scale=CONTROL_PARAMETERS[name]['action_scale'])
    meshes = {}
    for mesh in root.findall(".//mesh"):
        reference = mesh.get("filename")
        candidates = [urdf.parent / reference, urdf.parent / "meshes" / Path(reference).name]
        path = next((p.resolve() for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing URDF mesh: {reference}")
        name = path.name
        if name in meshes and meshes[name] != path:
            raise ValueError(f"Ambiguous mesh basename: {name}")
        meshes[name] = path
        mesh.set("filename", f"../meshes/{name}")
    root.remove(ground)
    root.remove(fixed)
    root.set("name", "gl165")
    manifest = dict(robot="gl165", source_urdf=str(urdf), source_urdf_sha256=sha256(urdf),
                    source_mesh_sha256={name: sha256(path) for name, path in meshes.items()},
                    root_link="base_link", joint_count=len(joints), total_mass_kg=total_mass,
                    removed_display_root=["ground_link", "ground_joint"], joints=parameters,
                    pd_gains_source="ELF3 controller parameters synchronized by joint name at user request",
                    control_parameters_source=CONTROL_SOURCE)
    return root, meshes, manifest


def prepare_source(source, output):
    root, meshes, manifest = inspect_source(source)
    output = Path(output)
    # Never replace an existing conversion, including a partially generated asset.
    output.mkdir(parents=True, exist_ok=False)
    (output / "source/urdf").mkdir(parents=True)
    (output / "source/meshes").mkdir()
    shutil.copy2(manifest['source_urdf'], output / "source/original.urdf")
    for name, path in meshes.items():
        shutil.copy2(path, output / "source/meshes" / name)
    urdf = output / "source/urdf/gl165.urdf"
    ET.indent(root)
    ET.ElementTree(root).write(urdf, encoding="utf-8", xml_declaration=True)
    manifest['training_urdf_sha256'] = sha256(urdf)
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return urdf, manifest
