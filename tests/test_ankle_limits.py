"""Regression checks for the restored no-waist-pose training configuration."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import xml.etree.ElementTree as ET

import pytest
import torch

from test_tang_commands import commands
from tools.fixed_pelvis_height import configure_height_range

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'legged_lab/assets/elf3_bxi_31'
spec = importlib.util.spec_from_file_location('ankle_limits_cpu', ASSET / 'ankle_limits.py')
ankle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ankle)
walk = importlib.import_module('tang_cpu.walk_robust_curriculum')
teleop = importlib.import_module('tang_cpu.teleop_standing')


def test_asset_assigns_solver_limits_to_correct_urdf_axes_without_changing_other_parameters(monkeypatch):
    # Construct the actual asset config with lightweight config containers; no GPU.
    class Config(NS):
        def replace(self, **fields):
            result = deepcopy(self)
            result.__dict__.update(fields)
            return result

    actuator_module = ModuleType('isaaclab.actuators')
    actuator_module.ImplicitActuatorCfg = Config
    legacy_module = ModuleType('legged_lab.assets.elf3_lite')
    legacy_module.ELF3LITE_CFG = Config(spawn=Config(), init_state=Config())
    monkeypatch.setitem(sys.modules, actuator_module.__name__, actuator_module)
    monkeypatch.setitem(sys.modules, legacy_module.__name__, legacy_module)
    asset_spec = importlib.util.spec_from_file_location(
        'bxi_asset_cpu', ASSET / '__init__.py', submodule_search_locations=[str(ASSET)])
    asset = importlib.util.module_from_spec(asset_spec)
    monkeypatch.setitem(sys.modules, asset_spec.name, asset)
    asset_spec.loader.exec_module(asset)
    original = json.loads((ASSET / 'control_parameters.json').read_text())
    actuator = asset.ELF3_BXI_CFG.actuators['deployment']
    for name, values in original['joints'].items():
        expected = values['limits']['velocity']
        if name in ('l_ankle_x_joint', 'r_ankle_x_joint'):
            expected = 5.0
        elif name in ('l_ankle_y_joint', 'r_ankle_y_joint'):
            expected = 15.0
        assert actuator.velocity_limit_sim[name] == expected
        assert actuator.effort_limit_sim[name] == values['effort_limit']
        assert actuator.stiffness[name] == values['kp']
        assert actuator.damping[name] == values['kd']
        assert actuator.armature[name] == values['armature']
    assert asset.PARAMETERS == original


def test_ankle_axis_mapping_in_prepared_urdf():
    from legged_lab.paths import robot_asset_dir
    path = robot_asset_dir()/'source/resources/elf3_dof31/urdf/elf3.urdf'
    if not path.is_file():
        pytest.skip('Optional external robot resources not installed')
    urdf = ET.parse(path)
    axes = {j.attrib['name']: j.find('axis').attrib['xyz']
            for j in urdf.findall('joint') if j.find('axis') is not None}
    for side in ('l', 'r'):
        assert axes[f'{side}_ankle_x_joint'] == '1 0 0'
        assert axes[f'{side}_ankle_y_joint'] == '0 1 0'


@pytest.mark.parametrize('age', [0, 5000, 20000, 80000])
def test_restored_commands_keep_waist_neutral_but_retain_walking_turning_and_height(age):
    cfg = NS(commands=walk.walk_commands(), arm_motion=NS(amass_neutral_fraction=.15), reward={})
    teleop.configure_teleop_standing(cfg, .5)
    configure_height_range(cfg, (.4, .8), 0, 20000)
    ankle.verify_neutral_waist(cfg.commands)
    cmd = commands.TangCommands(cfg.commands, 4096, 'cpu')
    cmd.iteration = age
    torch.manual_seed(42)
    cmd.reset(torch.arange(cmd.num_envs))
    assert cmd.full_command.shape == (4096, 5)
    assert (cmd.target[:, 4] == 0).all()
    assert .45 < float(cmd.is_standing_env.float().mean()) < .55
    assert (cmd.target[:, 0].abs() > .1).any()
    assert (cmd.target[:, 2].abs() > .1).any()  # Locomotion yaw is still commanded.
    assert float(cmd.target[:, 3].min()) >= .4 - 1e-6
    assert float(cmd.target[:, 3].max()) <= .8 + 1e-6
    if age >= 20000:
        assert float(cmd.target[:, 3].min()) < .405


def test_waist_pose_reenable_is_rejected():
    cfg = walk.walk_commands()
    cfg.waist_yaw_range = (-.3, .3)
    with pytest.raises(ValueError, match='waist_yaw_range'):
        ankle.verify_neutral_waist(cfg)
    cfg = walk.walk_commands()
    cfg.waist_rpy = {'limits_ypr': [.5, .1, .1]}
    with pytest.raises(ValueError, match='excludes'):
        ankle.verify_neutral_waist(cfg)


def test_simulator_readback_checks_every_ankle_and_environment():
    names = ['r_ankle_y_joint', 'hip_joint', 'l_ankle_x_joint', 'r_ankle_x_joint', 'l_ankle_y_joint']
    limits = torch.tensor([[15., 20., 5., 5., 15.], [15., 20., 5., 5., 15.]])
    robot = NS(joint_names=names, root_physx_view=NS(get_dof_max_velocities=lambda: limits))
    assert ankle.verify_simulator_limits(robot) == ankle.ANKLE_VELOCITY_LIMITS_RAD_S
    limits[1, 3] = 20.
    with pytest.raises(ValueError, match='r_ankle_x_joint'):
        ankle.verify_simulator_limits(robot)
    limits[1, 3] = float('nan')
    with pytest.raises(ValueError):
        ankle.verify_simulator_limits(robot)


def test_missing_ankle_joint_is_rejected():
    joints = json.loads((ASSET / 'control_parameters.json').read_text())['joints']
    del joints['r_ankle_y_joint']
    with pytest.raises(ValueError, match='r_ankle_y_joint'):
        ankle.joint_velocity_limits(joints)
