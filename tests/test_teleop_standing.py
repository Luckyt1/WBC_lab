"""Standing-first sampling coverage and arm-reference diagnostics."""
import copy
import importlib
from types import SimpleNamespace as NS

import pytest
import torch

from test_tang_commands import commands
from test_tang_rewards import fake_env

teleop = importlib.import_module('tang_cpu.teleop_standing')
walk = importlib.import_module('tang_cpu.walk_robust_curriculum')
standing = importlib.import_module('tang_cpu.standing')


def config():
    cfg = NS(arm_motion=NS(), noise=NS(), push_velocity_curriculum=NS(), reset_velocity_curriculum=NS())
    walk.configure_walk_robust(cfg)
    return cfg


@pytest.mark.parametrize('age', [0, 5000, 20000, 50000])
@pytest.mark.parametrize('fraction,probabilities', [(.8, (.8, .05, .10, .05)), (.5, (.5, .125, .25, .125))])
def test_command_mixture_at_all_curriculum_ages_with_locomotion_coverage(age, fraction, probabilities):
    cfg = config()
    teleop.configure_teleop_standing(cfg, fraction)
    torch.manual_seed(42)
    cmd = commands.TangCommands(cfg.commands, 30000, 'cpu')
    cmd.iteration = age
    ids = torch.arange(cmd.num_envs)
    cmd.reset(ids)
    for _ in range(3):
        cmd.compute(15.)  # force another command draw, as in long rollouts
        for mode, expected in enumerate(probabilities):
            assert abs((cmd.sampled_mode == mode).float().mean().item() - expected) < .012
        assert not (cmd.sampled_mode == 4).any()
        assert (cmd.full_command[cmd.is_standing_env, :3] == 0).all()
        assert (cmd.full_command[:, 3] == .8235).all()
        assert (cmd.full_command[:, 4] == 0).all()
        for mode, axis in [(1, 2), (2, 0), (3, 1)]:
            values = cmd.command[cmd.sampled_mode == mode, axis]
            assert .42 < (values > 0).float().mean().item() < .58
            assert (values.abs() >= .1).all()
        assert cmd.time_left.min() >= 10 and cmd.time_left.max() <= 14


@pytest.mark.parametrize('fraction', [.8, .5])
def test_profile_preserves_arm_and_disturbance_courses_and_baseline_instances(fraction):
    original = config()
    cfg = copy.deepcopy(original)
    metadata = teleop.configure_teleop_standing(cfg, fraction)
    # The higher standing probability must not turn the arms off or accelerate
    # their amplitude curriculum, or change unrelated command ranges/durations.
    for key in ('arm_motion', 'noise', 'push_velocity_curriculum', 'reset_velocity_curriculum'):
        assert vars(getattr(cfg, key)) == vars(getattr(original, key))
    allowed = {'rel_standing_envs', 'rel_turning_envs', 'rel_lateral_envs', 'rel_crouch_yaw_envs', 'initial_mode_fractions'}
    for key, value in vars(original.commands).items():
        if key not in allowed:
            assert getattr(cfg.commands, key) == value
    assert original.commands.rel_standing_envs == .15
    assert cfg.arm_motion.amass_neutral_fraction == .3
    assert metadata['arm_amass_probability'] == .7
    assert metadata['standing_position_penalty']


def test_joint_arm_standing_metrics_are_conditional_and_safe_with_no_standing():
    env = fake_env(4)
    env.torso_body_id = 1
    env.scene = NS(env_origins=torch.zeros(4, 3))
    env.robot.data.body_lin_vel_w = torch.zeros(4, 5, 3)
    env.robot.data.body_ang_vel_w = torch.zeros(4, 5, 3)
    env.command_generator.is_standing_env[:] = torch.tensor([True, True, True, False])
    env.use_amass = True
    env.amass_neutral_mask = torch.tensor([False, False, True, False])
    metrics = standing.transition_metrics(env)
    assert metrics['Standing/amass_reference_fraction'].item() == pytest.approx(2/3)
    assert metrics['Commands/standing_with_amass_reference_fraction'].item() == .5
    env.command_generator.is_standing_env[:] = False
    metrics = standing.transition_metrics(env)
    assert metrics['Standing/amass_reference_fraction'] == 0
    assert metrics['Commands/standing_with_amass_reference_fraction'] == 0
    assert all(torch.isfinite(v) for v in metrics.values())
