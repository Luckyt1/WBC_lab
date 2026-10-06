import importlib
import torch
import pytest
from test_tang_commands import commands
from test_standing_repair import PPO, ActorCritic

motion = importlib.import_module('tang_cpu.wbc.elf3_arm_motion')


def test_fresh_mode_curriculum_preserves_standing_and_reaches_specialists():
    cfg = commands.TangCommandCfg(
        rel_standing_envs=.35, rel_turning_envs=.15, rel_lateral_envs=.15, rel_crouch_yaw_envs=.15,
        initial_mode_fractions=(.60, .20, 0., 0.), mode_start_iteration=2000, mode_end_iteration=8000,
        focus_start_iteration=4000, focus_end_iteration=20000,
        focus_height_initial_range=(.74, .80), focus_yaw_initial_range=(.05, .15))
    c = commands.TangCommands(cfg, 16000, 'cpu')
    ids = torch.arange(c.num_envs)
    c.reset(ids)
    assert not (c.sampled_mode >= 3).any()
    assert .58 < c.is_standing_env.float().mean() < .62
    c.iteration = 4000
    c.reset(ids)
    crouch = c.sampled_mode == 4
    assert crouch.any() and (c.target[crouch, 3] >= .74).all()
    assert (c.target[crouch, 4].abs() <= .15 + 1e-6).all()
    c.iteration = 20000
    c.reset(ids)
    for mode, p in enumerate((.35, .15, .20, .15, .15)):
        assert abs(float((c.sampled_mode == mode).float().mean()) - p) < .015
    crouch = c.sampled_mode == 4
    assert c.target[crouch, 3].min() < .51
    assert c.target[crouch, 4].abs().max() > .59
    assert (c.target[crouch, 4] > 0).any() and (c.target[crouch, 4] < 0).any()
    c.set_manual_command([0, 0, 0, .8235, 0])
    assert c.is_standing_env.all() and not (c.sampled_mode >= 3).any()


def test_neutral_reference_mask_and_zero_scale_do_not_retarget_default_pose():
    defaults = torch.full((3, 14), .12)
    limits = torch.tensor([-2., 2.]).expand(3, 14, 2)
    m = motion.Elf3ArmMotion(motion.ELF3_AMASS_JOINT_NAMES, defaults, limits, .02,
                           ramp_s=0., max_velocity=None, scale=1.)
    pose = torch.ones(3, 14) * .5
    result = m.step(pose, neutral_mask=torch.tensor([True, False, True])).clone()
    torch.testing.assert_close(result[[0, 2]], defaults[[0, 2]])
    assert (result[1] - defaults[1]).abs().max() > .1
    m.scale = 0.
    torch.testing.assert_close(m.step(pose), defaults)


def test_adaptive_learning_rate_stays_in_explicit_fresh_bounds():
    torch.manual_seed(9)
    p = ActorCritic(3, 4, 2, actor_hidden_dims=[8], critic_hidden_dims=[8],
                    init_noise_std=.5, noise_std_bounds=(.1, .6))
    alg = PPO(p, device='cpu', schedule='adaptive', learning_rate=.01,
              learning_rate_bounds=(1e-5, 3e-4), num_learning_epochs=3, num_mini_batches=2)
    assert alg.learning_rate == alg.optimizer.param_groups[0]['lr'] == 3e-4
    alg.init_storage('rl', 4, 4, [3], [4], [2])
    for _ in range(3):
        with torch.no_grad():
            for _ in range(4):
                alg.act(torch.randn(4, 3), torch.randn(4, 4))
                alg.process_env_step(torch.randn(4), torch.zeros(4), {})
            alg.compute_returns(torch.randn(4, 4))
        alg.update()
        assert 1e-5 <= alg.learning_rate <= 3e-4
        assert alg.learning_rate == alg.optimizer.param_groups[0]['lr']
    assert all(torch.isfinite(v).all() for v in p.parameters())
    with pytest.raises(ValueError):
        PPO(p, learning_rate_bounds=(.1, .01))
