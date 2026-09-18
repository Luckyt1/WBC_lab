"""Checks shared by the ELF3 flat and WBC simulation smoke tests."""
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


def check_elf3_asset(env):
    source = Path(env.cfg.scene.robot.spawn.usd_path).parents[1] / "source_elf3.urdf"
    urdf = ET.parse(source).getroot()
    joints = {j.get("name"): j for j in urdf.findall("joint") if j.get("type") == "revolute"}
    masses = {l.get("name"): float(l.find("inertial/mass").get("value"))
              for l in urdf.findall("link") if l.find("inertial/mass") is not None}
    assert env.robot.num_joints == 31 and env.robot.num_bodies == 32
    assert set(env.robot.joint_names) == set(joints)
    assert set(env.robot.body_names) == set(masses)
    for index, name in enumerate(env.robot.joint_names):
        limit = joints[name].find("limit")
        assert torch.allclose(env.robot.data.joint_effort_limits[:, index],
                              torch.full_like(env.robot.data.joint_effort_limits[:, index], float(limit.get("effort")))), name
        assert torch.allclose(env.robot.data.joint_vel_limits[:, index],
                              torch.full_like(env.robot.data.joint_vel_limits[:, index], float(limit.get("velocity")))), name
    expected_mass = torch.tensor([masses[n] for n in env.robot.body_names], device=env.device)
    assert torch.allclose(env.robot.data.default_mass.to(env.device), expected_mass.expand(env.num_envs, -1), atol=1e-6)
    assert env.num_actions == 15 and env.num_arm_joints == 14
    held = env.custom_joint_ids[env.num_actions + env.num_arm_joints:]
    assert {env.robot.joint_names[i] for i in held} == {"head_z_joint", "head_y_joint"}
    assert (env.robot.data.default_joint_pos[:, held] == 0).all()
    print("PASS: 31 joints / 32 bodies, URDF masses and actuator limits, 15 actions / 14 arms / 2 held head joints.")
    return held


def check_held_head(env, head_ids):
    assert (env.robot.data.joint_pos_target[:, head_ids] == 0).all(), "Head targets must remain at zero"
    assert torch.isfinite(env.robot.data.joint_pos[:, head_ids]).all()
