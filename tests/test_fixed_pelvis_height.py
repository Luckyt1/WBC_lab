"""Height-only adaptation must reach commands, audits and simulator inference."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from test_tang_commands import commands
from test_tang_rewards import fake_env

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from fixed_pelvis_height import (configure_fixed_height, expected_height_recipe, height_scenarios,
                                checkpoint_height, apply_onnx_height, validate_height)
from audit_auxiliary_resume import verify_recipe
walk = importlib.import_module('tang_cpu.walk_robust_curriculum')
standing = importlib.import_module('tang_cpu.standing')


def test_all_modes_resamples_and_manual_commands_use_lower_height():
    cfg = NS(commands=walk.walk_commands())
    original = deepcopy(cfg)
    configure_fixed_height(cfg, .78)
    cmd = commands.TangCommands(cfg.commands, 10000, 'cpu')
    cmd.iteration = 62000
    cmd.reset(torch.arange(cmd.num_envs))
    for _ in range(3):
        cmd.compute(15.)
        assert (cmd.full_command[:, 3] == .78).all()
        assert (cmd.target[:, 3] == .78).all()
        assert set(cmd.sampled_mode.tolist()) == {0, 1, 2, 3}
    cmd.set_manual_command([0., 0., 0., .8235, 0.])
    assert (cmd.full_command[:, 3] == .78).all()
    assert original.commands.nominal_height == .8235
    scenarios = height_scenarios(walk.SCENARIOS, .78)
    assert all(command[3] == .78 for _, command in scenarios)
    assert all(command[3] == .8235 for _, command in walk.SCENARIOS)


def test_height_error_penalty_now_prefers_078():
    env = fake_env(2)
    env.scene = NS(env_origins=torch.zeros(2, 3))
    env.robot.data.body_pos_w[:, env.pelvis_body_id, 2] = torch.tensor([.78, .8235])
    env.command_generator.full_command[:, 3] = .78
    torch.testing.assert_close(standing.height_tracking_error(env), torch.tensor([0., .0435]))


def recipes(tmp_path):
    source, target = tmp_path/'source', tmp_path/'target'
    env = dict(commands=dict(nominal_height='.8235', height_initial_range=['.8235']*2,
        height_range=['.8235']*2, sampling_height_range=['.8235']*2, rel_standing_envs='.8'),
        scene=dict(num_envs='4096'), reward=dict(height_tracking_error=dict(weight='-10.0')))
    for folder, cfg in ((source,env), (target,expected_height_recipe(env,.78))):
        (folder/'params').mkdir(parents=True)
        (folder/'params/env.yaml').write_text(yaml.safe_dump(cfg))
        (folder/'params/agent.yaml').write_text('policy: {}\nalgorithm: {}\nseed: 42\nnum_steps_per_env: 24\nempirical_normalization: false\n')
    return source, target


def test_only_explicit_height_change_is_accepted_by_resume(tmp_path):
    source,target=recipes(tmp_path)
    with pytest.raises(ValueError):
        verify_recipe(source,target)
    assert verify_recipe(source,target,target_height=.78)
    assert checkpoint_height(target/'model.pt') == .78


@pytest.mark.parametrize('extra', ['reward','sampling','num_envs','seed','wrong_height'])
def test_height_override_cannot_hide_other_recipe_changes(tmp_path,extra):
    source,target=recipes(tmp_path)
    path=target/'params/env.yaml'
    cfg=yaml.safe_load(path.read_text())
    if extra=='reward': cfg['reward']['height_tracking_error']['weight']='-20.0'
    elif extra=='sampling': cfg['commands']['rel_standing_envs']='.9'
    elif extra=='num_envs': cfg['scene']['num_envs']='32'
    elif extra=='wrong_height': cfg['commands']['sampling_height_range']=['.80']*2
    else: (target/'params/agent.yaml').write_text('policy: {}\nalgorithm: {}\nseed: 7\nnum_steps_per_env: 24\nempirical_normalization: false\n')
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError): verify_recipe(source,target,target_height=.78)


def test_onnx_height_reaches_simulator_observation_config_and_legacy_retains_default(tmp_path):
    import onnx
    model=onnx.helper.make_model(onnx.helper.make_graph([], 'test', [], []))
    path=tmp_path/'model.onnx'
    onnx.save(model,path)
    policy=NS(info=dict(height_command=.8235,waist_yaw_command=0.))
    assert apply_onnx_height(policy,path)==.8235
    entry=model.metadata_props.add();entry.key='height_command';entry.value='0.78'
    onnx.save(model,path)
    assert apply_onnx_height(policy,path)==.78
    assert policy.info==dict(height_command=.78,waist_yaw_command=0.)


@pytest.mark.parametrize('height',[0.,-.78,.9,float('nan'),float('inf')])
def test_invalid_height_rejected(height):
    with pytest.raises(ValueError): validate_height(height)
