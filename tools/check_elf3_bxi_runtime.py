"""Assert the live PhysX articulation uses the frozen deployment parameters."""
import xml.etree.ElementTree as ET

import torch
from legged_lab.assets.elf3_bxi_31 import PARAMETERS, ASSET_DIR


def check_runtime(env):
    p = PARAMETERS
    robot = env.robot
    assert robot.num_joints == 31 and robot.num_bodies == 32
    assert set(robot.joint_names) == set(p['deployment_joint_names'])
    assert [robot.joint_names[i] for i in env.policy_joint_ids] == p['policy_joint_names']
    assert [robot.joint_names[i] for i in env.arm_joint_ids] == p['arm_joint_names']
    for index, name in enumerate(robot.joint_names):
        values = p['joints'][name]
        for field, expected in (
            ('default_joint_pos', values['default_position']),
            ('joint_stiffness', values['kp']), ('joint_damping', values['kd']),
            ('joint_armature', values['armature']), ('joint_effort_limits', values['effort_limit']),
            ('joint_vel_limits', values['limits']['velocity']),
        ):
            actual = getattr(robot.data, field)[:, index]
            assert torch.allclose(actual, torch.full_like(actual, expected), atol=1e-5), (name, field, actual, expected)
        actual_limits = robot.data.joint_pos_limits[:, index]
        expected_limits = actual_limits.new_tensor([values['limits']['lower'], values['limits']['upper']])
        assert torch.allclose(actual_limits, expected_limits.expand_as(actual_limits), atol=1e-5), name
    expected_scale = env.action_scale.new_tensor([p['joints'][n]['action_scale'] for n in p['policy_joint_names']])
    torch.testing.assert_close(env.action_scale, expected_scale)
    urdf = ET.parse(ASSET_DIR / 'source' / p['urdf_relative_path']).getroot()
    masses = {l.get('name'): float(l.find('inertial/mass').get('value')) for l in urdf.findall('link') if l.find('inertial/mass') is not None}
    expected_mass = robot.data.default_mass.new_tensor([masses[n] for n in robot.body_names])
    torch.testing.assert_close(robot.data.default_mass, expected_mass.expand_as(robot.data.default_mass), atol=1e-6, rtol=1e-6)
    assert abs(env.step_dt - p['control_period_s']) < 1e-9
    print('ELF3_BXI_RUNTIME_PARAMETERS_PASS: 31 joints, 32 bodies; masses, hard limits, PD, armature, scales and control period match.', flush=True)
