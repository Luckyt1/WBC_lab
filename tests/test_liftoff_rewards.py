"""Physical reward semantics and persistence of the explicit B+C+D recipe."""
import ast
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from test_tang_commands import commands, yaw_quat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
reward = importlib.import_module('tang_cpu.liftoff_rewards')
from tools.run_teleop_curriculum import training_command


def wbc_functions():
    # Exercise the production reward functions without starting Isaac/PhysX.
    path = ROOT/'legged_lab/envs/elf3_tang/wbc/rewards.py'
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ('knee_stance_width', 'center_of_gravity_tracking')]
    namespace = dict(torch=torch, BaseEnv=object,
                     SceneEntityCfg=lambda name, **kwargs: NS(name=name, **kwargs))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_midpoint_reward_retains_standing_value_and_disables_moving_value():
    data = NS(body_pos_w=torch.tensor([[[0., .15, 0.], [0., -.15, 0.]]]).repeat(2, 1, 1),
              body_com_pos_w=torch.tensor([[[.1, .15, .4], [.1, -.15, .4]]]).repeat(2, 1, 1))
    env = NS(scene={'robot': NS(data=data)}, _cached_body_masses=torch.ones(2, 2),
             command_generator=NS(is_standing_env=torch.tensor([True, False])))
    fn = wbc_functions()['center_of_gravity_tracking']
    selector = NS(name='robot', body_ids=[0, 1])
    original = fn(env, asset_cfg=selector)
    assert original[0].item() == pytest.approx(.7788008)
    torch.testing.assert_close(fn(env, asset_cfg=selector, standing_only=True),
                               torch.tensor([original[0], 0.]))


def test_lateral_knee_width_ignores_step_length_and_world_heading_but_not_crossing():
    data = NS(body_pos_w=torch.tensor([[[0., .27, 0.], [0., 0., 0.]],
                                     [[.3, .27, 0.], [0., 0., 0.]]]))
    env = NS(scene={'robot': NS(data=data)}, base_quat_w=yaw_quat([0., 0.]))
    fn = wbc_functions()['knee_stance_width']
    selector = NS(name='robot', body_ids=[0, 1])
    assert fn(env, selector, ideal_width=.27)[1] < .2
    torch.testing.assert_close(fn(env, selector, ideal_width=.27, lateral_only=True), torch.ones(2))
    p = data.body_pos_w
    p[..., :2] = torch.stack((-p[..., 1], p[..., 0]), -1)
    env.base_quat_w = yaw_quat([torch.pi/2, torch.pi/2])
    torch.testing.assert_close(fn(env, selector, ideal_width=.27, lateral_only=True),
                               torch.ones(2), atol=1e-6, rtol=0)
    data.body_pos_w = data.body_pos_w.flip(1)
    assert (fn(env, selector, ideal_width=.27, lateral_only=True) < .001).all()


def test_load_transfer_is_mirror_invariant_and_cannot_reward_airborne_or_standing():
    force = torch.zeros(7, 2, 3)
    force[..., 2] = torch.tensor([[200., 200.], [320., 80.], [80., 320.],
                                  [400., 0.], [400., 0.], [0., 0.], [30., 0.]])
    force[5, :, 0] = 500.  # Horizontal contact is not vertical support.
    env = NS(contact_sensor=NS(data=NS(net_forces_w=force)), feet_cfg=NS(body_ids=[0, 1]),
             command_generator=NS(is_standing_env=torch.tensor([False]*3+[True]+[False]*3)),
             base_projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(7, 1))
    env.base_projected_gravity_b[4, 2] = -.5  # Falling/strongly tilted.
    torch.testing.assert_close(reward.load_transfer(env), torch.tensor([0., .36, .36, 0., 0., 0., 0.]))


def test_recipe_changes_only_bcd_and_keeps_airtime_and_other_rewards(monkeypatch):
    manager = ModuleType('isaaclab.managers')
    manager.RewardTermCfg = lambda **kwargs: NS(**kwargs)
    monkeypatch.setitem(sys.modules, 'isaaclab.managers', manager)
    cfg = NS(reward=NS(center_of_gravity_tracking=NS(weight=.5, params={'sigma': .2}),
                       knee_stance_width=NS(weight=.5, params={'ideal_width': .27, 'std': .1}),
                       feet_air_time=NS(weight=.15), height_tracking_error=NS(weight=-10.)))
    metadata = reward.configure_liftoff_rewards(cfg)
    assert cfg.reward.center_of_gravity_tracking.params == {'sigma': .2, 'standing_only': True}
    assert cfg.reward.knee_stance_width.params == {'ideal_width': .27, 'std': .1, 'lateral_only': True}
    assert cfg.reward.center_of_gravity_tracking.weight == cfg.reward.knee_stance_width.weight == .5
    assert cfg.reward.feet_air_time.weight == .15 and cfg.reward.height_tracking_error.weight == -10.
    assert cfg.reward.liftoff_load_transfer.weight == .2
    assert metadata['recipe'] == cfg.liftoff_reward_recipe == 'bcd_v1'


@pytest.mark.parametrize('checkpoint', [None, Path('/run/model_4999.pt')])
def test_recipe_flag_survives_fresh_and_resumed_segments(checkpoint):
    argv = training_command('python', '/deploy/python', '/out', 'quiet', 1000, 4096, .8,
                            checkpoint=checkpoint, symmetry=True, liftoff_rewards=True)
    assert '--liftoff_rewards' in argv and '--elf3_symmetry' in argv
    assert ('--resume_checkpoint' in argv) == (checkpoint is not None)


def test_checkpoint_recipe_identifies_legacy_and_bcd(tmp_path):
    params = tmp_path/'params'
    params.mkdir()
    path = params/'env.yaml'
    path.write_text('pelvis_height: 0.8\n')
    assert reward.checkpoint_recipe(tmp_path/'model.pt') is None
    path.write_text('liftoff_reward_recipe: bcd_v1\n')
    assert reward.checkpoint_recipe(tmp_path/'model.pt') == 'bcd_v1'
    path.write_text('liftoff_reward_recipe: unknown\n')
    with pytest.raises(ValueError):
        reward.checkpoint_recipe(tmp_path/'model.pt')
