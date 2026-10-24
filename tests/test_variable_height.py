"""Variable height must reach every movement mode without changing rewards."""
from copy import deepcopy
import importlib
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from test_tang_commands import commands
from test_fixed_pelvis_height import ROOT
from tools.fixed_pelvis_height import (configure_fixed_height, configure_height_range,
    expected_height_range_recipe, range_height_scenarios, checkpoint_height_range)
from tools.audit_auxiliary_resume import verify_recipe

walk = importlib.import_module('tang_cpu.walk_robust_curriculum')


def config():
    cfg = NS(commands=walk.walk_commands(), reward={'feet_air_time': .15,
        'liftoff_load_transfer': .2, 'height_tracking_error': -10.})
    configure_fixed_height(cfg, .8)
    return cfg


@pytest.mark.parametrize('iteration,minimum', [(56001,.75),(61001,.65),(66001,.55)])
def test_height_curriculum_covers_standing_turning_sagittal_and_lateral(iteration, minimum):
    cfg = config()
    original = deepcopy(cfg.reward)
    configure_height_range(cfg, (.55,.8), 56001)
    cmd = commands.TangCommands(cfg.commands, 12000, 'cpu')
    cmd.iteration = iteration
    torch.manual_seed(173)
    cmd.reset(torch.arange(cmd.num_envs))
    assert cfg.reward == original
    assert cmd.full_command.shape == (12000,5)
    assert torch.all(cmd.full_command[:,3] == .8)  # no reset-time crouch jump
    for mode in (0,1,2,3):
        h = cmd.target[cmd.sampled_mode == mode,3]
        assert len(h) > 500
        assert minimum-1e-6 <= float(h.min()) < minimum+.003
        assert float(h.max()) <= .800001
        assert .20 < float((h == .8).float().mean()) < .30
    before = cmd.full_command[:,3].clone()
    cmd.compute(.02)
    assert (torch.abs(cmd.full_command[:,3]-before) <= .12*.02+1e-6).all()


def test_manual_height_is_clamped_and_ramped_while_walking():
    cfg = config(); configure_height_range(cfg, (.55,.8), 0)
    cmd = commands.TangCommands(cfg.commands, 4, 'cpu')
    cmd.set_manual_command([.3,0,0,.4,0])
    assert (cmd.target[:,3] == .55).all()
    for _ in range(150): cmd.compute(.02)
    torch.testing.assert_close(cmd.full_command[:,3], torch.full((4,),.55))
    assert (cmd.full_command[:,0] == .3).all()
    cmd.set_manual_command([.3,0,0,1.,0])
    assert (cmd.target[:,3] == .8).all()
    assert (cmd.full_command[:,3] == .55).all()


def test_resume_allows_only_declared_height_fields(tmp_path):
    cfg = config()
    source_env = dict(commands=asdict(cfg.commands), reward=cfg.reward, scene={'num_envs':4096})
    source_env['commands'].pop('height_sampling_all_modes')  # old checkpoint
    source = tmp_path/'source'; target = tmp_path/'target'
    for folder in (source,target):
        (folder/'params').mkdir(parents=True)
        (folder/'params/agent.yaml').write_text('policy: {}\nalgorithm: {}\nseed: 42\nnum_steps_per_env: 24\nempirical_normalization: false\n')
    (source/'params/env.yaml').write_text(yaml.safe_dump(source_env))
    before = yaml.load((source/'params/env.yaml').read_text(),Loader=yaml.BaseLoader)
    expected = expected_height_range_recipe(before,(.55,.8),56001)
    (target/'params/env.yaml').write_text(yaml.safe_dump(expected))
    with pytest.raises(ValueError): verify_recipe(source,target)
    assert verify_recipe(source,target,height_range=(.55,.8),height_curriculum_start=56001)
    assert checkpoint_height_range(target/'model.pt') == (.55,.8)
    expected['reward']['feet_air_time'] = '0.0'
    (target/'params/env.yaml').write_text(yaml.safe_dump(expected))
    with pytest.raises(ValueError):
        verify_recipe(source,target,height_range=(.55,.8),height_curriculum_start=56001)


def test_evaluation_preserves_nominal_cases_and_covers_low_mid_heights():
    rows = range_height_scenarios(walk.SCENARIOS,(.55,.8))
    original_names = {name for name,_ in walk.SCENARIOS}
    assert original_names.issubset(dict(rows))
    assert all(cmd[3] == .8 for name,cmd in rows if name in original_names)
    for h in (.55,.675):
        assert len([cmd for name,cmd in rows if name.startswith(f'height_{h:.3f}_')]) == 7


@pytest.mark.parametrize('age,minimum', [(0,.75), (10000,.575), (20000,.4), (80000,.4)])
def test_deep_crouch_course_preserves_zero_waist_yaw_and_all_velocity_modes(age,minimum):
    cfg = config()
    original_reward = deepcopy(cfg.reward)
    configure_height_range(cfg, (.4,.8), 0, 20000)
    cmd = commands.TangCommands(cfg.commands, 12000, 'cpu')
    cmd.iteration = age
    torch.manual_seed(473)
    cmd.reset(torch.arange(cmd.num_envs))
    assert cfg.reward == original_reward
    assert cmd.full_command.shape == (12000,5)
    assert (cmd.full_command[:,3] == .8).all()
    assert (cmd.target[:,4] == 0).all()
    for mode in (0,1,2,3):
        mask = cmd.sampled_mode == mode
        heights = cmd.target[mask,3]
        assert len(heights) > 500
        assert minimum-1e-6 <= heights.min() < minimum+.003
        assert heights.max() <= .800001
        assert .20 < float((heights == .8).float().mean()) < .30
    # Removing torso-relative yaw must preserve the separate base-turn command.
    assert (cmd.target[cmd.sampled_mode == 1,2].abs() >= .3).all()
    previous = cmd.full_command[:,3].clone()
    cmd.compute(.02)
    assert (abs(cmd.full_command[:,3]-previous) <= .002401).all()


def test_deep_height_manual_slew_and_lower_bound():
    cfg = config(); configure_height_range(cfg, (.4,.8), 0, 20000)
    cmd = commands.TangCommands(cfg.commands, 4, 'cpu')
    cmd.set_manual_command([0,0,0,.35,.6])
    assert (cmd.target[:,3] == .4).all()
    assert (cmd.target[:,4] == 0).all()
    for _ in range(180):
        before = cmd.full_command.clone()
        cmd.compute(.02)
        assert (abs(cmd.full_command[:,3]-before[:,3]) <= .002401).all()
    torch.testing.assert_close(cmd.full_command[:,3], torch.full((4,),.4))
    cmd.set_manual_command([0,0,0,.8,0])
    assert (cmd.full_command[:,3] == .4).all()
    for _ in range(180): cmd.compute(.02)
    torch.testing.assert_close(cmd.full_command[:,3], torch.full((4,),.8))


def test_deep_crouch_evaluation_covers_lower_and_middle_height_with_zero_yaw():
    rows = range_height_scenarios(walk.SCENARIOS,(.4,.8))
    assert len(rows) == len(walk.SCENARIOS)+14
    for h in (.4,.6):
        cases = [cmd for name,cmd in rows if name.startswith(f'height_{h:.3f}_')]
        assert len(cases) == 7
        assert all(cmd[3] == pytest.approx(h) and cmd[4] == 0 for cmd in cases)


@pytest.mark.parametrize('bounds',[(.8,.55),(.55,.9),(.55,float('nan')),(.8,.8),(.399,.8)])
def test_invalid_range_rejected(bounds):
    with pytest.raises(ValueError): configure_height_range(config(),bounds,0)
