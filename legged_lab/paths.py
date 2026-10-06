"""Portable paths for external datasets, robot assets and the BXI runtime."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def robot_asset_dir():
    return Path(os.environ.get('ELF3_ASSET_DIR', ROOT / 'resources/elf3_bxi_31')).expanduser().absolute()


def amass_root():
    return Path(os.environ.get('ELF3_AMASS_DIR', ROOT / 'resources/dataset')).expanduser().absolute()


def deployment_framework():
    value = os.environ.get('ELF3_BXI_FRAMEWORK_DIR')
    if not value:
        raise RuntimeError('Set ELF3_BXI_FRAMEWORK_DIR to bxi_controller_ros2/src/bxi_example_py_elf3; see docs/installation.md')
    path = Path(value).expanduser().absolute()
    if not (path / 'bxi_example_py_elf3/framework').is_dir():
        raise FileNotFoundError(f'BXI framework package is missing under {path}')
    return path


def deployment_mod():
    return ROOT / 'deployment/com.bxi.walk_robust'


def mujoco_xml():
    value = os.environ.get('ELF3_MUJOCO_XML')
    return Path(value).expanduser().absolute() if value else deployment_framework() / 'data/mujoco_simulation/elf3.xml'
