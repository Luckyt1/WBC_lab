"""Regression checks for reward conflicts and joint/body indexing."""
from types import SimpleNamespace as NS
import unittest

import torch

from test_tang_commands import commands, rewards, yaw_quat


def fake_env(n=3):
    cmd = commands.TangCommands(commands.TangCommandCfg(), n, 'cpu')
    cmd.set_manual_command([0, 0, 0, 0.5, 0.6])
    data = NS(
        body_quat_w=yaw_quat(torch.zeros(n * 5)).reshape(n, 5, 4),
        body_pos_w=torch.zeros(n, 5, 3), body_com_pos_w=torch.zeros(n, 5, 3),
        GRAVITY_VEC_W=torch.tensor([0., 0., -1.]).expand(n, -1),
        default_joint_pos=torch.zeros(n, 4),
        soft_joint_pos_limits=torch.tensor([-1., 1.]).expand(n, 4, 2).clone(),
        computed_torque=torch.zeros(n, 4), joint_effort_limits=torch.ones(n, 4) * 10,
    )
    sensor = NS(current_contact_time=torch.ones(n, 5), current_air_time=torch.zeros(n, 5))
    masses = torch.tensor([1., 1., 1., 1., 6.]).expand(n, -1)
    return NS(
        robot=NS(data=data, root_physx_view=NS(get_masses=lambda: masses)), device='cpu',
        contact_sensor=NS(data=sensor), feet_cfg=NS(body_ids=[3, 1]), feet_body_ids=[1, 2],
        knee_body_ids=[3, 4], pelvis_body_id=0, command_generator=cmd,
        policy_joint_ids=[2, 0], action_scale=torch.tensor([0.5, 0.25]), action=torch.zeros(n, 2),
        raw_actions=torch.zeros(n, 2),
        action_buffer=NS(_circular_buffer=NS(buffer=torch.zeros(n, 1, 2))),
    )


class TangRewardsTest(unittest.TestCase):
    def test_small_motion_and_pure_turn_keep_airtime_reward(self):
        env = fake_env()
        env.command_generator.target[:, :3] = torch.tensor([[0., 0., 0.], [0.02, 0., 0.], [0., 0., 0.02]])
        env.command_generator.is_standing_env[:] = torch.tensor([True, False, False])
        env.contact_sensor.data.current_contact_time[:, 1] = 0
        env.contact_sensor.data.current_air_time[:, 1] = 0.2
        torch.testing.assert_close(rewards.feet_air_time(env, 0.4), torch.tensor([0., 0.2, 0.2]))
        # Airborne, double support and standing must not earn single-support time.
        env.contact_sensor.data.current_contact_time[:, 3] = 0
        torch.testing.assert_close(rewards.feet_air_time(env, 0.4), torch.zeros(3))

    def test_stable_contact_requires_both_feet_and_standing(self):
        env = fake_env(4)
        env.contact_sensor.data.current_contact_time[:, 1] = torch.tensor([0., 0.49, 0.75, 1.])
        env.command_generator.is_standing_env[3] = False
        reward = rewards.standing_feet_contact(env)
        self.assertEqual(reward[:2].sum(), 0)
        self.assertGreater(reward[2], 0.9)
        self.assertEqual(reward[3], 0)
        torch.testing.assert_close(rewards.standing_support_gate(env), torch.tensor([0., 0., 1., 0.]))

    def test_flat_feet_free_yaw_but_not_upside_down_or_swing(self):
        env = fake_env(4)
        env.robot.data.body_quat_w[1, env.feet_body_ids] = yaw_quat([1.5, -2.])
        env.robot.data.body_quat_w[2, env.feet_body_ids] = torch.tensor([0., 1., 0., 0.])
        env.contact_sensor.data.current_contact_time[3, 1] = 0
        torch.testing.assert_close(rewards.standing_feet_flat(env), torch.tensor([1., 1., 0., 0.]))
        env.command_generator.set_manual_command([0., 0., 0.02, 0.5, 0.6])
        torch.testing.assert_close(rewards.standing_feet_flat(env), torch.zeros(4))

    def test_center_of_mass_uses_actual_mass_and_com_and_standing_gate(self):
        env = fake_env(4)
        # Heavy external arm moves its CoM while link origins stay fixed.
        env.robot.data.body_com_pos_w[1:, 4, 0] = 0.5
        env.command_generator.is_standing_env[2] = False
        env.contact_sensor.data.current_contact_time[3, 1] = 0
        reward = rewards.standing_center_of_mass(env)
        self.assertEqual(reward[0], 1)
        self.assertTrue(0 < reward[1] < 0.2)
        self.assertEqual(reward[2:].sum(), 0)
        # A world translation and yaw rotation must not change this constraint.
        for name in ('body_pos_w', 'body_com_pos_w'):
            p = getattr(env.robot.data, name)
            p[..., :2] = torch.stack((-p[..., 1], p[..., 0]), -1) + torch.tensor([17., -20.])
        torch.testing.assert_close(rewards.standing_center_of_mass(env), reward, atol=1e-5, rtol=1e-5)

    def test_knees_use_signed_lateral_separation_not_step_length(self):
        env = fake_env(4)
        env.robot.data.body_pos_w[:, 3, :2] = torch.tensor([[0., 0.2], [0.8, 0.2], [0.8, 0.07], [0.8, -0.2]])
        reward = rewards.knee_inward(env)
        torch.testing.assert_close(reward[:3], torch.tensor([0., 0., 0.25]))
        self.assertGreater(reward[3], 1)
        env.robot.data.body_quat_w[:, 0] = yaw_quat(torch.full((4,), torch.pi / 2))
        p = env.robot.data.body_pos_w
        p[..., :2] = torch.stack((-p[..., 1], p[..., 0]), -1)
        torch.testing.assert_close(rewards.knee_inward(env), reward)

    def test_raw_targets_use_unclipped_action_and_policy_joint_order(self):
        env = fake_env()
        env.robot.data.default_joint_pos[:, 2] = 0.5
        env.robot.data.soft_joint_pos_limits[:, 2, 1] = 0.75
        # Executed/delayed action is zero; newly requested target still exceeds the limit.
        env.raw_actions[:] = torch.tensor([1., 0.])
        torch.testing.assert_close(rewards.raw_action_target_limits(env), torch.ones(3) * 0.25)
        env.raw_actions[:] = torch.tensor([1000., -8.])
        self.assertTrue(torch.all(rewards.raw_action_target_limits(env) > 500))
        # Changing only external arm defaults and limits must not affect policy cost.
        before = rewards.raw_action_target_limits(env)
        env.robot.data.default_joint_pos[:, 1] = 500
        torch.testing.assert_close(rewards.raw_action_target_limits(env), before)

    def test_torque_is_normalized_and_excludes_arms(self):
        env = fake_env()
        env.robot.data.computed_torque[:] = torch.tensor([5., 1000., -20., 1000.])
        env.robot.data.joint_effort_limits[:, 2] = 20
        # Selected ratios are 1.0 and 0.5, only first exceeds 0.9.
        torch.testing.assert_close(rewards.joint_torque_excess(env), torch.ones(3) * 0.05)


if __name__ == '__main__':
    unittest.main()
