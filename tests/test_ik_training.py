"""Split isolation, physical execution parity and reach/hold/return semantics."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from test_tang_commands import commands

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT/'tests/fixtures/ik_banks/v1'
motion = importlib.import_module('tang_cpu.wbc.ik_arm_motion')
recipe = importlib.import_module('tang_cpu.ik_training')
gravity = importlib.import_module('tang_cpu.wbc.arm_gravity')


def instance(n=64, split='train'):
    m = json.loads((BANK/'manifest.json').read_text())
    x = motion.IkArmMotion(BANK,split,m['arm_joint_names'],torch.tensor(m['default_arm_q']).repeat(n,1),.02)
    x.reset(torch.arange(n), x.defaults)
    return x


def test_split_precedes_sampling_and_mirrors_are_disjoint():
    x, y = instance(), instance(split='test')
    m = x.manifest
    train = {t['pose_key'] for t in m['targets'] if t['split']=='train'}
    test = {t['pose_key'] for t in m['targets'] if t['split']=='test'}
    assert not train & test and not set(x.ids)&set(y.ids)
    assert not y.hard.any()
    assert x.weights[x.hard].sum().item() == pytest.approx(.3)
    torch.manual_seed(132)
    samples = torch.multinomial(x.weights,100000,replacement=True)
    assert x.hard[samples].float().mean().item() == pytest.approx(.3,abs=.01)
    limits = torch.tensor(m['joint_limits'])
    assert torch.isfinite(x.paths).all()
    assert (x.paths >= limits[:,0]-1e-6).all() and (x.paths <= limits[:,1]+1e-6).all()
    assert x.paths.diff(dim=2).abs().max() <= .120001


def test_phase_boundary_holds_and_returns_in_seconds_without_random_clip_start():
    x = instance(1)
    x.neutral[:] = False
    x.index[:] = x.ids.index('regression_002_both_forward_long')
    x.durations[:] = torch.tensor([[1.,2.,3.,2.,1.]])
    observed = {}
    # Fixed clocks exercise exact boundaries without cumulative roundoff.
    for t in (0., .5, 1., 2., 3., 4., 5., 6., 7., 8., 8.999):
        x.elapsed[:] = t
        observed[t] = (int(x.phase[0]), x.step().clone(), int(x.phase[0]))
    assert [observed[t][2] for t in (0.,1.,3.,6.,8.)] == [0,1,2,3,4]
    torch.testing.assert_close(observed[0.][1], x.defaults)
    torch.testing.assert_close(observed[1.][1], x.paths[x.index,0,0])
    for t in (3.,4.,5.,6.):
        torch.testing.assert_close(observed[t][1], x.paths[x.index,0,-1])
    torch.testing.assert_close(observed[8.][1], x.paths[x.index,1,-1])
    torch.testing.assert_close(observed[8.999][1], x.defaults,atol=1e-5,rtol=0)
    x.elapsed[:] = 9.; x.step()
    assert x.completed_cycles.item() == 1 and x.phase.item()==0


def test_partial_resets_and_live_rate_collision_supervision():
    from tang_cpu.wbc.arm_guard import ArmCollisionGuard
    x = instance(4)
    ids = torch.arange(4)
    limits = torch.tensor(x.manifest['joint_limits']).expand(4,-1,-1)
    guard = ArmCollisionGuard(x.manifest['arm_joint_names'],['torso_link'],limits,'cpu',max_velocity=6.)
    body_pos = torch.zeros(4,1,3)
    body_quat = torch.tensor([1.,0.,0.,0.]).expand(4,1,4)
    x.neutral[:] = False
    x.durations[:] = torch.tensor([.1,.5,.5,.5,.1])
    phases = set()
    for step in range(90):
        previous = x.target.clone()
        q = guard.project(x.step(),previous,previous,body_pos,body_quat,0)
        assert (q-previous).abs().max() <= .120001
        assert torch.isfinite(q).all()
        assert ((q >= limits[...,0]-1e-6)&(q<=limits[...,1]+1e-6)).all()
        x.target.copy_(q)
        phases.add(int(x.phase[0]))
    assert phases=={0,1,2,3,4}
    unchanged = x.elapsed[[0,2,3]].clone()
    x.reset(torch.tensor([1]),x.defaults[[1]])
    torch.testing.assert_close(x.elapsed[[0,2,3]],unchanged)


def test_gravity_matches_original_deployment_with_tilted_gravity_and_effort_clip():
    from legged_lab.paths import ik_source_dir
    p = ik_source_dir()/'gravity.py'
    if not p.is_file():
        pytest.skip('External upstream gravity.py not prepared')
    spec = importlib.util.spec_from_file_location('original_arm_gravity',p)
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    source = original.ArmGravityModel()
    batch = gravity.ArmGravity('cpu')
    rng = np.random.default_rng(2931)
    q = rng.uniform(-2.,2.,(100,14)).astype(np.float32)
    g = rng.normal(size=(100,3));g *= 9.81/np.linalg.norm(g,axis=1,keepdims=True)
    expected = np.stack([source.compute(a,b,np.empty(14)) for a,b in zip(q,g)])
    expected = np.clip(expected,-batch.limits.numpy(),batch.limits.numpy())
    actual = batch.compute(torch.from_numpy(q),torch.tensor(g,dtype=torch.float32))
    np.testing.assert_allclose(actual.numpy(),expected,atol=1e-5,rtol=2e-5)


def test_recipe_keeps_height_commands_and_reward_configuration():
    cfg = NS(arm_motion=NS(),scene=NS(max_episode_length_s=20),commands=object(),reward=object())
    before = (cfg.commands,cfg.reward)
    r = recipe.configure_ik_training(cfg,BANK)
    assert (cfg.commands,cfg.reward)==before
    assert cfg.arm_motion.source=='ik_elf3' and cfg.arm_motion.ik_split=='train'
    assert cfg.arm_motion.guard_max_velocity==6. and cfg.arm_motion.guard_margin==.012
    assert cfg.scene.max_episode_length_s >= 1.2+6.+8.+4.+1.5
    assert r['split']=='train' and r['sha256']
