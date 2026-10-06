"""CPU checks for command coverage and physical frame semantics (no Simulator needed)."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch

package = ModuleType('tang_cpu')
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'legged_lab/envs/elf3_tang')]
sys.modules[package.__name__] = package
commands = importlib.import_module('tang_cpu.commands')
rewards = importlib.import_module('tang_cpu.rewards')


def yaw_quat(angle):
    angle = torch.as_tensor(angle, dtype=torch.float32).reshape(-1)
    q = torch.zeros(len(angle), 4)
    q[:, 0], q[:, 3] = (angle / 2).cos(), (angle / 2).sin()
    return q


class TangCommandsTest(unittest.TestCase):
    def test_focused_modes_progress_and_hold_time(self):
        cfg = commands.TangCommandCfg(rel_lateral_envs=.15, rel_crouch_yaw_envs=.15)
        cmd = commands.TangCommands(cfg, 20000, 'cpu')
        for iteration, speed_range, height_range, yaw_range in (
                (81598, (.1, .2), (.6, .65), (.15, .25)),
                (83598, (.15, .3), (.5, .6), (.4, .6))):
            torch.manual_seed(93)
            cmd.iteration = iteration
            cmd.reset(torch.arange(cmd.num_envs))
            for mode, fraction in enumerate((.25, .15, .30, .15, .15)):
                self.assertAlmostEqual((cmd.sampled_mode == mode).float().mean().item(), fraction, delta=.012)
            lateral, crouch = cmd.sampled_mode == 3, cmd.sampled_mode == 4
            lat, low = cmd.target[lateral], cmd.target[crouch]
            torch.testing.assert_close(lat[:, [0, 2, 4]], torch.zeros_like(lat[:, [0, 2, 4]]))
            self.assertTrue(((lat[:, 1].abs() >= speed_range[0]-1e-6) & (lat[:, 1].abs() <= speed_range[1]+1e-6)).all())
            self.assertAlmostEqual((lat[:, 1] > 0).float().mean().item(), .5, delta=.03)
            self.assertTrue(((low[:, 3] >= height_range[0]-1e-6) & (low[:, 3] <= height_range[1]+1e-6)).all())
            self.assertTrue(((low[:, 4].abs() >= yaw_range[0]-1e-6) & (low[:, 4].abs() <= yaw_range[1]+1e-6)).all())
            self.assertAlmostEqual((low[:, 4] > 0).float().mean().item(), .5, delta=.03)
            torch.testing.assert_close(low[:, 1:3], torch.zeros_like(low[:, 1:3]))
            self.assertTrue((cmd.time_left[lateral | crouch] >= 10.).all())
            self.assertTrue((cmd.time_left[lateral | crouch] <= 14.).all())
        cmd.set_manual_command([0., -.2, 0., .5, .6])
        torch.testing.assert_close(cmd.target, cmd.target.new_tensor([0., -.2, 0., .5, .6]).expand_as(cmd.target))

    def test_curriculum_and_combined_squat_walk_yaw_coverage(self):
        torch.manual_seed(42)
        cfg = commands.TangCommandCfg()
        cmd = commands.TangCommands(cfg, 12000, 'cpu')
        ids = torch.arange(cmd.num_envs)
        cmd.reset(ids)
        self.assertGreaterEqual(cmd.target[:, 3].min(), 0.78 - 1e-6)
        self.assertLessEqual(cmd.target[:, 4].abs().max(), 0.10 + 1e-6)
        self.assertLessEqual(cmd.command[:, 0].abs().max(), 0.18 + 1e-6)
        cmd.iteration = cfg.pose_end_iteration
        cmd.reset(ids)
        moving = cmd.command[:, :2].norm(dim=-1) > 0.05
        turning = (cmd.command[:, :2].norm(dim=-1) == 0) & (cmd.command[:, 2].abs() > 0)
        self.assertTrue(0.23 < cmd.is_standing_env.float().mean() < 0.27)
        self.assertTrue(0.13 < turning.float().mean() < 0.17)
        self.assertGreater(moving.float().mean(), 0.55)
        # The three skills must co-occur, rather than live in disjoint training groups.
        combined = moving & (cmd.target[:, 3] < 0.6) & (cmd.target[:, 4].abs() > 0.3)
        self.assertGreater(combined.sum(), 300)
        self.assertTrue(torch.all(cmd.target[:, 3] >= 0.5))
        self.assertTrue(torch.all(cmd.target[:, 3] <= 0.8235))
        self.assertTrue(torch.all(cmd.target[:, 4].abs() <= 0.6))

    def test_pose_slew_manual_persistence_and_reset(self):
        cfg = commands.TangCommandCfg(resampling_time_range=(0.1, 0.1))
        cmd = commands.TangCommands(cfg, 3, 'cpu')
        cmd.set_manual_command([0.3, 0, 0, 0.5, 0.6])
        for _ in range(200):
            before = cmd.full_command.clone()
            cmd.compute(0.02)
            self.assertLessEqual((cmd.full_command[:, 3] - before[:, 3]).abs().max(), 0.0024 + 1e-7)
            self.assertLessEqual((cmd.full_command[:, 4] - before[:, 4]).abs().max(), 0.008 + 1e-7)
            torch.testing.assert_close(cmd.command, torch.tensor([[0.3, 0, 0]]).expand(3, -1))
        torch.testing.assert_close(cmd.full_command, torch.tensor([[0.3, 0, 0, 0.5, 0.6]]).expand(3, -1))
        cmd.reset(torch.tensor([1]))
        self.assertAlmostEqual(cmd.full_command[1, 3].item(), cfg.nominal_height, places=6)
        self.assertAlmostEqual(cmd.target[1, 3].item(), 0.5)
        self.assertFalse(cmd.is_standing_env.any())
        cmd.set_manual_command([0, 0, 0, 0.5, -0.6])
        self.assertTrue(cmd.is_standing_env.all())

    def test_manual_bounds_and_invalid_command(self):
        cmd = commands.TangCommands(commands.TangCommandCfg(), 1, 'cpu')
        cmd.set_manual_command([99, -99, 99, 0.1, 99])
        torch.testing.assert_close(cmd.target[0], torch.tensor([0.6, -0.3, 0.8, 0.5, 0.6]))
        for bad in ([0, 0, 0], [0, 0, 0, float('nan'), 0]):
            with self.assertRaises(ValueError):
                cmd.set_manual_command(bad)

    def test_quaternion_order_and_heading_wrap(self):
        q = yaw_quat([torch.pi / 2])
        v = torch.tensor([[0., 1., 0.]])
        torch.testing.assert_close(commands.rotate_inverse(q, v), torch.tensor([[1., 0., 0.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(commands.yaw_frame_velocity(q, v), torch.tensor([[1., 0.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(commands.wrap_to_pi(torch.tensor([2 * torch.pi - 0.1])), torch.tensor([-0.1]), atol=1e-6, rtol=0)

    def test_tracking_does_not_confuse_waist_yaw_with_base_turn(self):
        cmd = commands.TangCommands(commands.TangCommandCfg(), 2, 'cpu')
        cmd.full_command[:] = torch.tensor([0.3, 0., 0., 0.6, 0.4])
        # Two copies of the same task at different world headings and floor elevations.
        headings = torch.tensor([0., 2.9])
        data = SimpleNamespace(
            body_quat_w=torch.stack((yaw_quat(headings), yaw_quat(headings + 0.4)), dim=1),
            body_pos_w=torch.tensor([[[0., 0., 0.6], [0., 0., 0.83]], [[0., 0., 1.6], [0., 0., 1.83]]]),
            body_lin_vel_w=torch.zeros(2, 2, 3), body_ang_vel_w=torch.zeros(2, 2, 3))
        data.body_lin_vel_w[:, 0, 0] = 0.3 * headings.cos()
        data.body_lin_vel_w[:, 0, 1] = 0.3 * headings.sin()
        env = SimpleNamespace(robot=SimpleNamespace(data=data), pelvis_body_id=0, torso_body_id=1,
                              command_generator=cmd, scene=SimpleNamespace(env_origins=torch.tensor([[0., 0., 0.], [0., 0., 1.]])))
        for fn in (rewards.track_pelvis_velocity, rewards.track_pelvis_turn_rate,
                   rewards.track_pelvis_height, rewards.track_waist_yaw):
            torch.testing.assert_close(fn(env, std=0.1), torch.ones(2))
        # Torso rotation and elevation must not masquerade as pelvis movement or height.
        data.body_quat_w[:, 1] = yaw_quat(headings + 0.1)
        data.body_ang_vel_w[:, 1, 2] = 2.
        data.body_pos_w[:, 1, 2] += 0.3
        torch.testing.assert_close(rewards.track_pelvis_turn_rate(env, 0.1), torch.ones(2))
        torch.testing.assert_close(rewards.track_pelvis_height(env, 0.1), torch.ones(2))
        self.assertTrue(torch.all(rewards.track_waist_yaw(env, 0.1) < 0.01))
        data.body_pos_w[:, 0, 2] += 0.2
        self.assertTrue(torch.all(rewards.track_pelvis_height(env, 0.1) < 0.02))


if __name__ == '__main__':
    unittest.main()
