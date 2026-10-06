import importlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch

from test_tang_rewards import fake_env
from test_tang_commands import commands
from test_tang_commands import yaw_quat

standing = importlib.import_module('tang_cpu.standing')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rsl_rl'))
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.algorithms.ppo import PPO


def test_contact_reward_is_dense_but_zero_for_moving_and_turning():
    env = fake_env(4)
    env.command_generator.is_standing_env[:] = torch.tensor([True, True, False, False])
    env.contact_sensor.data.current_contact_time.zero_()
    env.contact_sensor.data.current_contact_time[0, env.feet_cfg.body_ids] = 0.02
    env.contact_sensor.data.current_contact_time[1, env.feet_cfg.body_ids[0]] = 0.02
    env.contact_sensor.data.current_contact_time[2:, env.feet_cfg.body_ids] = 1.
    torch.testing.assert_close(standing.standing_contact(env), torch.tensor([1., .5, 0., 0.]))


def test_quiet_reward_detects_drift_and_foot_swing():
    env = fake_env(3)
    env.robot.data.body_lin_vel_w = torch.zeros(3, 5, 3)
    env.robot.data.body_ang_vel_w = torch.zeros(3, 5, 3)
    env.robot.data.body_lin_vel_w[1:, 0, 0] = .2
    env.command_generator.is_standing_env[2] = False
    reward = standing.standing_quiet_base(env)
    assert reward[0] == 1 and 0 < reward[1] < .01 and reward[2] == 0
    cost = standing.standing_base_motion(env)
    assert cost[0] == 0 and cost[1] > 1 and cost[2] == 0
    env.robot.data.body_lin_vel_w[:, env.feet_body_ids, 2] = .3
    torch.testing.assert_close(standing.standing_foot_motion(env), torch.tensor([.3, .3, 0.]))


def test_dense_tracking_penalizes_stalling_and_uses_heading_and_current_height():
    env = fake_env(3)
    env.scene = NS(env_origins=torch.tensor([[0., 0., 10.]]).expand(3, -1))
    env.robot.data.body_pos_w[:, 0, 2] = 10.8
    env.robot.data.body_lin_vel_w = torch.zeros(3, 5, 3)
    env.robot.data.body_ang_vel_w = torch.zeros(3, 5, 3)
    env.command_generator.full_command[:] = torch.tensor([.3, 0., .5, .8, 0.])
    # The final squat target must not replace the current, slew-limited height.
    env.command_generator.target[:, 3] = .5
    env.robot.data.body_quat_w[:, 0] = yaw_quat(torch.full((3,), torch.pi / 2))
    env.robot.data.body_lin_vel_w[:, 0, 1] = torch.tensor([0., .1, .3])
    env.robot.data.body_ang_vel_w[:, 0, 2] = torch.tensor([0., .2, .5])
    torch.testing.assert_close(standing.velocity_tracking_error(env), torch.tensor([.3, .2, 0.]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(standing.turn_tracking_error(env), torch.tensor([.5, .3, 0.]))
    torch.testing.assert_close(standing.height_tracking_error(env), torch.zeros(3), atol=1e-6, rtol=0)
    env.robot.data.body_pos_w[:, 0, 2] -= torch.tensor([0., .1, 4.])
    torch.testing.assert_close(standing.height_tracking_error(env), torch.tensor([0., .1, .5]), atol=1e-6, rtol=1e-5)
    env.robot.data.body_lin_vel_w.fill_(10000.)
    env.robot.data.body_ang_vel_w.fill_(10000.)
    assert (standing.velocity_tracking_error(env) == 2.).all()
    assert (standing.turn_tracking_error(env) == 3.).all()


def test_diagnostics_separate_standing_turning_and_moving_without_empty_nan():
    env = fake_env(3)
    env.torso_body_id = 1
    env.scene = NS(env_origins=torch.zeros(3, 3))
    env.robot.data.body_lin_vel_w = torch.zeros(3, 5, 3)
    env.robot.data.body_ang_vel_w = torch.zeros(3, 5, 3)
    env.command_generator.full_command[:, :3] = torch.tensor([[0., 0., 0.], [0., 0., .5], [.3, 0., 0.]])
    env.command_generator.is_standing_env[:] = torch.tensor([True, False, False])
    env.robot.data.body_lin_vel_w[:, 0, 0] = torch.tensor([.1, 5., .2])
    result = standing.transition_metrics(env)
    assert abs(result['Standing/horizontal_speed'] - .1) < 1e-6
    assert abs(result['Turning/turn_rate_mae'] - .5) < 1e-6
    assert abs(result['Moving/velocity_xy_error'] - .1) < 1e-6
    assert result['Standing/sustained_double_support_fraction'] == 1
    env.command_generator.set_manual_command([0., 0., 0., .8235, 0.])
    result = standing.transition_metrics(env)
    assert result['Turning/turn_rate_mae'] == result['Moving/velocity_xy_error'] == 0
    assert all(torch.isfinite(v) for v in result.values())


def test_neutral_sampling_retains_other_commands():
    torch.manual_seed(42)
    cfg = commands.TangCommandCfg(rel_standing_envs=.5, rel_neutral_standing=.7)
    cmd = commands.TangCommands(cfg, 10000, 'cpu')
    cmd.iteration = 100000
    cmd.reset(torch.arange(10000))
    neutral = cmd.is_standing_env & (cmd.target[:, 3] == cfg.nominal_height) & (cmd.target[:, 4] == 0)
    assert .33 < neutral.float().mean() < .39
    assert ((~cmd.is_standing_env) & (cmd.command[:, :2].norm(dim=-1) > .1)).sum() > 2000
    assert ((cmd.target[:, 3] < .6) & (cmd.target[:, 4].abs() > .3)).sum() > 300


def test_noise_projection_preserves_unconfigured_policy_and_log_prob():
    policy = ActorCritic(3, 4, 2, actor_hidden_dims=[8], critic_hidden_dims=[8], noise_std_bounds=(.05, .8))
    with torch.no_grad():
        policy.std[:] = torch.tensor([-2., 37.])
    policy.constrain_parameters()
    torch.testing.assert_close(policy.std, torch.tensor([.05, .8]))
    obs = torch.zeros(5, 3)
    actions = policy.act(obs)
    torch.testing.assert_close(policy.get_actions_log_prob(actions), policy.distribution.log_prob(actions).sum(-1))
    legacy = ActorCritic(3, 4, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
    with torch.no_grad():
        legacy.std.fill_(37.)
    legacy.constrain_parameters()
    assert torch.all(legacy.std == 37.)
    policy.load_state_dict(legacy.state_dict())
    assert torch.all(policy.std == .8)


def test_mean_regularization_has_recovery_gradient_only_outside_bounds():
    policy = ActorCritic(3, 4, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
    alg = PPO(policy, device='cpu', action_mean_bound_loss_coef=.1)
    alg.set_action_bounds(torch.tensor([-1., -2.]), torch.tensor([2., 1.]))
    mean = torch.tensor([[-1000., 1000.], [0., 0.]], requires_grad=True)
    loss = alg.action_mean_bound_loss(mean)
    loss.backward()
    assert mean.grad[0, 0] < 0 and mean.grad[0, 1] > 0
    assert torch.equal(mean.grad[1], torch.zeros(2))
    assert torch.isfinite(loss) and mean.grad.abs().max() < 1


def test_ppo_update_with_bounds_retains_finite_model_and_valid_noise():
    torch.manual_seed(7)
    policy = ActorCritic(3, 4, 2, actor_hidden_dims=[8], critic_hidden_dims=[8],
                         init_noise_std=.2, noise_std_bounds=(.05, .4))
    alg = PPO(policy, device='cpu', action_mean_bound_loss_coef=.1, learning_rate=.1)
    alg.set_action_bounds(torch.tensor([-.1, -.2]), torch.tensor([.2, .1]))
    alg.init_storage('rl', 4, 2, [3], [4], [2])
    with torch.no_grad():
        for _ in range(2):
            alg.act(torch.randn(4, 3), torch.randn(4, 4))
            alg.process_env_step(torch.randn(4), torch.zeros(4), {})
        alg.compute_returns(torch.randn(4, 4))
    result = alg.update()
    assert result['action_mean_bounds'] > 0
    assert all(torch.isfinite(p).all() for p in policy.parameters())
    assert torch.all(policy.std >= .05) and torch.all(policy.std <= .4)
