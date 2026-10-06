import importlib
from types import SimpleNamespace as NS

import torch
from test_tang_commands import commands
from test_tang_rewards import fake_env

refine = importlib.import_module('tang_cpu.standing_refine')
walk = importlib.import_module('tang_cpu.walk_robust_curriculum')
standing = importlib.import_module('tang_cpu.standing')


def config():
    cfg = NS(commands=walk.walk_commands(), arm_motion=NS(amass_neutral_fraction=.3),
             reward=NS(standing_quiet_base=NS(weight=2., params={}),
                       standing_base_motion=NS(weight=-.5, params={})))
    refine.configure_standing_refinement(cfg)
    return cfg


def test_sampling_keeps_locomotion_and_fixed_pose_at_resumed_age():
    torch.manual_seed(27)
    cmd = commands.TangCommands(config().commands, 20000, 'cpu')
    for age in (22000, 38000, 100000):
        cmd.iteration = age
        cmd.reset(torch.arange(cmd.num_envs))
        for mode, fraction in [(0, .35), (1, .15), (2, .35), (3, .15)]:
            assert abs(float((cmd.sampled_mode == mode).float().mean()) - fraction) < .02
        assert (cmd.target[:, 3] == .8235).all()
        assert (cmd.target[:, 4] == 0).all()
        cmd.compute(15.)
        assert (cmd.is_standing_env & (cmd.command[:, :3].abs().sum(-1) == 0)).any()


def test_small_drift_cost_increases_and_is_zero_while_walking():
    cfg = config()
    env = fake_env(4)
    env.robot.data.body_lin_vel_w = torch.zeros(4, 5, 3)
    env.robot.data.body_ang_vel_w = torch.zeros(4, 5, 3)
    env.command_generator.is_standing_env[:] = torch.tensor([True, True, True, False])
    env.robot.data.body_lin_vel_w[:, 0, 0] = torch.tensor([0., .02, .07, .07])
    old_reward = 2 * standing.standing_quiet_base(env) - .5 * standing.standing_base_motion(env)
    new_reward = (cfg.reward.standing_quiet_base.weight * standing.standing_quiet_base(env, **cfg.reward.standing_quiet_base.params)
                  + cfg.reward.standing_base_motion.weight * standing.standing_base_motion(env, **cfg.reward.standing_base_motion.params))
    assert new_reward[0] > new_reward[1] > new_reward[2]
    assert new_reward[0] - new_reward[1] > 5 * (old_reward[0] - old_reward[1])
    assert new_reward[2] < 0
    assert new_reward[3] == old_reward[3] == 0
    assert torch.isfinite(new_reward).all()


def test_refinement_does_not_modify_original_sampling_and_supports_low_noise():
    original = walk.walk_commands()
    cfg = config()
    assert original.rel_standing_envs == .15 and cfg.commands.rel_standing_envs == .35
    agent = NS(policy=NS(), algorithm=NS())
    refine.configure_standing_refinement_agent(agent)
    assert agent.policy.noise_std_bounds == (.05, .30)
    assert agent.algorithm.learning_rate_bounds[0] <= agent.algorithm.learning_rate <= agent.algorithm.learning_rate_bounds[1]
