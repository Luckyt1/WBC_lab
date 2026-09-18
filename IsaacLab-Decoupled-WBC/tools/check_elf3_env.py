"""Headless GPU smoke check for ELF3 model/control/command integration."""

import argparse
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import torch

from legged_lab.envs import task_registry
from legged_lab.envs.elf3.elf3_config import ELF3_ACTION_JOINT_NAMES
from check_elf3_asset import check_elf3_asset, check_held_head


def check():
    cfg, _ = task_registry.get_cfgs("elf3_flat")
    cfg.scene.num_envs = args.num_envs
    cfg.scene.seed = 42
    cfg.device = args.device
    env = task_registry.get_task_class("elf3_flat")(cfg, headless=args.headless)
    assert env.num_actions == 15
    head_ids = check_elf3_asset(env)
    assert env.action_joint_names == ELF3_ACTION_JOINT_NAMES
    assert [env.robot.joint_names[i] for i in env.custom_joint_ids[:15]] == ELF3_ACTION_JOINT_NAMES
    assert env.robot.body_names[env.base_body_id] == "waist_z_link"
    assert env.amass is None

    # A resumed curriculum reports its effective ranges, not the final limits.
    env.command_generator.update_curriculum(17000)
    snapshot = env.command_generator.curriculum_log()
    assert abs(snapshot["Curriculum/lin_vel_x/progress_percent"] - 17.5) < 1e-6
    assert abs(snapshot["Curriculum/lin_vel_x/max"] - 0.105) < 1e-6
    assert snapshot["Curriculum/body_roll/progress_percent"] == 0.0
    env.command_generator.cfg.curriculum.enable = False
    disabled = env.command_generator.curriculum_log()
    assert disabled["Curriculum/enabled"] == 0.0
    assert "Curriculum/lin_vel_x/progress_percent" not in disabled
    assert disabled["Curriculum/lin_vel_x/max"] == 0.6
    env.command_generator.cfg.curriculum.enable = True

    # Verify that movement and turning are reachable after the standing curriculum.
    env.set_curriculum_start_iteration(60000)
    env.command_generator.update_curriculum(60000)
    env.command_generator.reset(torch.arange(env.num_envs, device=env.device))
    env.command_generator.compute(env.step_dt)
    commands = env.command_generator.command
    assert (commands[:, :2].abs() > 0.01).any(), "Late curriculum cannot sample locomotion"
    assert (commands[:, 2].abs() > 0.01).any(), "Late curriculum cannot sample turning"

    arm_ids = env.custom_joint_ids[15:29]
    for _ in range(32):
        obs, rewards, _, extras = env.step(torch.zeros(env.num_envs, 15, device=env.device))
        assert extras["log"]["Curriculum/lin_vel_x/progress_percent"] == 100.0
        assert extras["log"]["Curriculum/body_yaw/progress_percent"] == 100.0
        assert snapshot["Curriculum/lin_vel_x/progress_percent"] == 17.5
        assert obs.shape == (env.num_envs, 60)
        assert extras["observations"]["critic"].shape == (env.num_envs, 63)
        assert torch.isfinite(obs).all() and torch.isfinite(rewards).all()
        check_held_head(env, head_ids)
        assert torch.isfinite(extras["observations"]["critic"]).all()
        assert torch.allclose(env.base_quat_w, env.robot.data.body_quat_w[:, env.base_body_id])
        assert torch.allclose(
            env.robot.data.joint_pos_target[:, arm_ids], env.robot.data.default_joint_pos[:, arm_ids]
        ), "Arms must remain at ELF3 default targets"
        limits = env.robot.data.soft_joint_pos_limits
        assert (env.robot.data.joint_pos_target >= limits[..., 0] - 1e-6).all()
        assert (env.robot.data.joint_pos_target <= limits[..., 1] + 1e-6).all()
    print("ELF3 integration passed: 31 joints, 15 actions, pelvis frame, held head/arms, finite rollouts, moving curriculum.")


if __name__ == "__main__":
    status = 0
    try:
        check()
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        # Match the training entry points: Kit teardown may hang in headless runs.
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
