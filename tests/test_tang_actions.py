"""CPU regression tests for physical bounds, scales, and action ownership."""
import importlib
import unittest

import torch

from test_tang_commands import commands  # initializes the simulator-free package

policy_action_limits = importlib.import_module('tang_cpu.actions').policy_action_limits


class TangActionBoundsTest(unittest.TestCase):
    def test_per_environment_offsets_and_nonuniform_scales(self):
        defaults = torch.tensor([[0.5, -0.25], [0., 0.25]])
        limits = torch.tensor([[[-1., 1.], [-0.5, 0.5]], [[-0.2, 0.3], [-1., 1.]]])
        scale = torch.tensor([0.5, 0.25])
        low, high = policy_action_limits(defaults, limits, scale, 100.)
        torch.testing.assert_close(low, torch.tensor([[-3., -1.], [-0.4, -5.]]))
        torch.testing.assert_close(high, torch.tensor([[1., 3.], [0.6, 3.]]))
        for sign in (-1, 1):
            sampled = torch.full_like(defaults, sign * 1e6)
            original = sampled.clone()
            bounded = sampled.clamp(low, high)
            targets = defaults + bounded * scale
            self.assertTrue(torch.all(targets >= limits[..., 0] - 1e-6))
            self.assertTrue(torch.all(targets <= limits[..., 1] + 1e-6))
            # Clipping must never overwrite PPO's original Gaussian sample.
            torch.testing.assert_close(sampled, original, rtol=0, atol=0)

    def test_negative_scale_and_configured_cap_intersection(self):
        defaults = torch.zeros(1, 2)
        limits = torch.tensor([[[-2., 1.], [-10., 20.]]])
        scale = torch.tensor([-0.5, 1.])
        low, high = policy_action_limits(defaults, limits, scale, 3.)
        torch.testing.assert_close(low, torch.tensor([[-2., -3.]]))
        torch.testing.assert_close(high, torch.tensor([[3., 3.]]))

    def test_feasible_actions_and_zero_history_are_unchanged(self):
        defaults = torch.tensor([[0.8, -0.3]])
        limits = torch.tensor([[[-1., 1.], [-1., 1.]]])
        scale = torch.tensor([0.3, 0.7])
        low, high = policy_action_limits(defaults, limits, scale, 100.)
        for actions in (torch.zeros_like(defaults), torch.tensor([[-2., 1.]])):
            torch.testing.assert_close(actions.clamp(low, high), actions, rtol=0, atol=0)

    def test_invalid_config_fails_before_training(self):
        defaults = torch.zeros(1, 2)
        limits = torch.tensor([-1., 1.]).expand(1, 2, 2)
        for scale in (torch.tensor([0., 1.]), torch.tensor([float('nan'), 1.])):
            with self.assertRaisesRegex(ValueError, 'action_scale'):
                policy_action_limits(defaults, limits, scale, 100.)
        for cap in (0., -1., float('inf')):
            with self.assertRaisesRegex(ValueError, 'clip_actions'):
                policy_action_limits(defaults, limits, torch.ones(2), cap)
        with self.assertRaisesRegex(ValueError, 'defaults'):
            policy_action_limits(defaults + 2, limits, torch.ones(2), 100.)


if __name__ == '__main__':
    unittest.main()
