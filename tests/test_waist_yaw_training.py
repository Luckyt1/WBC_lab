"""Yaw coverage, independent curricula, strict continuation and acceptance."""
from copy import deepcopy
from dataclasses import asdict
import importlib
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from test_tang_commands import commands
from test_tang_rewards import fake_env
from tools.fixed_pelvis_height import configure_height_range
from tools.waist_yaw_training import (configure_waist_yaw, waist_yaw_fields,
    expected_waist_yaw_recipe, checkpoint_waist_yaw, waist_yaw_scenarios, evaluate_waist_yaw)
from tools.audit_auxiliary_resume import verify_recipe

walk = importlib.import_module('tang_cpu.walk_robust_curriculum')
teleop = importlib.import_module('tang_cpu.teleop_standing')
standing = importlib.import_module('tang_cpu.standing')


def config():
    cfg = NS(commands=walk.walk_commands(), arm_motion=NS(amass_neutral_fraction=.15),
             reward={'waist_yaw_tracking_error': -2., 'track_body_yaw_exp': 1.})
    teleop.configure_teleop_standing(cfg, .5)
    configure_height_range(cfg, (.55, .8), 0)
    return cfg


@pytest.mark.parametrize('age,height_min,yaw_max', [(0,.75,.05),(5000,.65,.05),
                                                 (10000,.55,.05+.55/3),(20000,.55,.6)])
def test_all_modes_cover_both_signs_without_changing_height_or_velocity(age,height_min,yaw_max):
    cfg = config()
    before = deepcopy(cfg)
    configure_waist_yaw(cfg, (-.6,.6))
    assert cfg.reward == before.reward and cfg.arm_motion == before.arm_motion
    cmd = commands.TangCommands(cfg.commands, 20000, 'cpu')
    cmd.iteration = age
    torch.manual_seed(129)
    cmd.reset(torch.arange(cmd.num_envs))
    assert cmd.full_command.shape == (20000, 5)
    assert (cmd.full_command[:,4] == 0).all()  # no reset-time yaw step
    assert cmd.pose_ranges()[0] == pytest.approx((height_min,.8))
    assert cmd.pose_ranges()[1] == pytest.approx((-yaw_max,yaw_max))
    for mode, probability in enumerate((.5,.125,.25,.125)):
        mask = cmd.sampled_mode == mode
        assert abs(float(mask.float().mean())-probability) < .02
        h, yaw = cmd.target[mask,3], cmd.target[mask,4]
        assert float(h.min()) == pytest.approx(height_min, abs=.003)
        assert .20 < float((h == .8).float().mean()) < .30
        assert .20 < float((yaw == 0).float().mean()) < .30
        nonzero = yaw[yaw != 0]
        assert .45 < float((nonzero > 0).float().mean()) < .55
        assert float(yaw.max()) == pytest.approx(yaw_max, abs=.003)
        assert float(yaw.min()) == pytest.approx(-yaw_max, abs=.003)
    assert (cmd.command[cmd.is_standing_env] == 0).all()
    assert (cmd.command[cmd.sampled_mode == 1,:2] == 0).all()
    assert (cmd.command[cmd.sampled_mode == 3,2] == 0).all()
    for _ in range(200):
        previous = cmd.full_command.clone()
        cmd.compute(.02)
        assert (abs(cmd.full_command[:,4]-previous[:,4]) <= .008001).all()
        assert (abs(cmd.full_command[:,3]-previous[:,3]) <= .002401).all()
    torch.testing.assert_close(cmd.full_command[:,4], cmd.target[:,4])


def test_manual_waist_twist_changes_no_velocity_and_resets_smoothly():
    cfg = config(); configure_waist_yaw(cfg, (-.6,.6))
    cmd = commands.TangCommands(cfg.commands, 3, 'cpu')
    for request, expected in [(9.,.6),(-9.,-.6),(0.,0.)]:
        cmd.set_manual_command([0,0,0,.55,request])
        for _ in range(200): cmd.compute(.02)
        assert (cmd.target[:,4] == expected).all()
        assert (cmd.command == 0).all() and cmd.is_standing_env.all()
    cmd.set_manual_command([0,0,0,.55,.6])
    cmd.reset(torch.tensor([1]))
    assert cmd.full_command[1,4] == 0 and cmd.target[1,4] == .6


def test_waist_curriculum_opt_in_leaves_legacy_yaw_zero():
    cfg = config()
    cmd = commands.TangCommands(cfg.commands, 4000, 'cpu')
    cmd.iteration = 80000
    cmd.reset(torch.arange(cmd.num_envs))
    assert (cmd.target[:,4] == 0).all()


@pytest.mark.parametrize('bounds,start,updates', [((0,.6),0,10), ((-.6,0),0,10),
    ((-.7,.6),0,10), ((-.6,float('nan')),0,10), ((-.6,.6),-1,10), ((-.6,.6),0,0)])
def test_invalid_curriculum_rejected(bounds,start,updates):
    with pytest.raises(ValueError): waist_yaw_fields(bounds,start,updates)


def test_resume_allows_only_declared_yaw_fields_and_preserves_saved_timing(tmp_path):
    cfg = config()
    before = dict(commands=asdict(cfg.commands), reward=cfg.reward)
    for key in ('waist_yaw_sampling_all_modes','waist_yaw_start_iteration','waist_yaw_end_iteration'):
        before['commands'].pop(key)  # old checkpoint defaults are backward compatible
    source, target = tmp_path/'source', tmp_path/'target'
    for folder in (source,target):
        (folder/'params').mkdir(parents=True)
        (folder/'params/agent.yaml').write_text('policy: {}\nalgorithm: {}\nseed: 42\nnum_steps_per_env: 24\nempirical_normalization: false\n')
    (source/'params/env.yaml').write_text(yaml.safe_dump(before))
    before = yaml.load((source/'params/env.yaml').read_text(),Loader=yaml.BaseLoader)
    after = expected_waist_yaw_recipe(before,(-.6,.6))
    (target/'params/env.yaml').write_text(yaml.safe_dump(after))
    with pytest.raises(ValueError): verify_recipe(source,target)
    assert verify_recipe(source,target,waist_yaw_range=(-.6,.6))
    recipe = checkpoint_waist_yaw(target/'model.pt')
    assert recipe['bounds'] == (-.6,.6) and recipe['start_iteration'] == 5000
    assert recipe['ramp_updates'] == 15000 and checkpoint_waist_yaw(source/'model.pt') is None
    after['commands']['height_range'] = ['0.6','0.8']
    (target/'params/env.yaml').write_text(yaml.safe_dump(after))
    with pytest.raises(ValueError): verify_recipe(source,target,waist_yaw_range=(-.6,.6))


def test_acceptance_rejects_survival_without_yaw_tracking_or_stationary_feet():
    rows = waist_yaw_scenarios((.55,.8),(-.6,.6))
    assert len(rows) == len(dict(rows)) == 24
    report = dict(waist_yaw_command_range=(-.6,.6), height_command_range=(.55,.8),
                  repeats=32,warmup_s=4,measure_s=12,arm_guard_enabled=True,profiles=[])
    for name in ('bootstrap','controlled_arms_on','training_disturbances'):
        report['profiles'].append(dict(settings={'name':name}, scenarios=[dict(
            scenario=n, target=t, trials=32, survived=32, valid_samples=16000,
            mae=[0.,0.,0.,.01,.04], knee_clipped_fraction={'left':0.,'right':0.},
            single_support_fraction=0.) for n,t in rows]))
    assert evaluate_waist_yaw(report)['passed']
    report['profiles'][0]['scenarios'][0]['mae'][4] = .6
    assert not evaluate_waist_yaw(report)['passed']
    report['profiles'][0]['scenarios'][0]['mae'][4] = .04
    report['profiles'][0]['scenarios'][0]['mae'][0] = .06
    assert not evaluate_waist_yaw(report)['passed']
    report['profiles'][0]['scenarios'][0]['mae'][0] = 0.
    report['profiles'][0]['scenarios'][0]['single_support_fraction'] = .5
    assert not evaluate_waist_yaw(report)['passed']


def test_yaw_metrics_are_finite_without_active_yaw():
    env = fake_env(4)
    env.torso_body_id = 1
    env.scene = NS(env_origins=torch.zeros(4,3))
    env.robot.data.body_lin_vel_w = torch.zeros(4,5,3)
    env.robot.data.body_ang_vel_w = torch.zeros(4,5,3)
    env.command_generator.full_command[:,4] = 0
    result = standing.transition_metrics(env)
    assert result['Commands/waist_yaw_active_fraction'] == 0
    assert all(torch.isfinite(v) for v in result.values())
