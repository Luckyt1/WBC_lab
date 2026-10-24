"""Portable paths for external datasets, robot assets and the BXI runtime."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(variable, default):
    path = Path(os.environ.get(variable, default)).expanduser()
    return path.absolute() if path.is_absolute() else (ROOT / path).absolute()


def robot_asset_dir():
    return _path('ELF3_ASSET_DIR', 'resources/elf3_bxi_31')


def amass_root():
    return _path('ELF3_AMASS_DIR', 'resources/dataset')


def ik_source_dir():
    """Verified external upstream files, flattened by prepare_ik_source.py."""
    return _path('ELF3_IK_SOURCE_DIR', 'resources/elf3_ik_upstream')


def ik_bank_dir():
    return _path('ELF3_IK_BANK', 'resources/ik_bank_v3')


def deployment_framework():
    path = _path('ELF3_BXI_FRAMEWORK_DIR', 'resources/bxi_example_py_elf3')
    if not (path / 'bxi_example_py_elf3/framework').is_dir():
        raise FileNotFoundError(f'BXI framework package is missing under {path}')
    return path


def deployment_mod():
    return ROOT / 'deployment/com.bxi.walk_robust'


def mujoco_xml():
    value = os.environ.get('ELF3_MUJOCO_XML')
    return _path('ELF3_MUJOCO_XML', value) if value else deployment_framework() / 'data/mujoco_simulation/elf3.xml'
