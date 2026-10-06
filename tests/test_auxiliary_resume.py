"""Resume must retain learned state and count updates without repeating labels."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'rsl_rl')]
from audit_auxiliary_resume import audit_restored, completed_updates, record_resume, verify_recipe
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


def trained_state():
    return dict(iter=29999, model_state_dict={'weight': torch.tensor([.3])},
                optimizer_state_dict={'state': {0: {'step': torch.tensor(600000.), 'exp_avg': torch.tensor([.2])}},
                                      'param_groups': [{'lr': 3.375e-5}]},
                auxiliary_cfg={'mode': 'dynamics'},
                auxiliary_state_dict={'updates': torch.tensor(30000), 'weight': torch.tensor([.1])},
                auxiliary_optimizer_state_dict={'state': {0: {'step': torch.tensor(600000.)}},
                                                'param_groups': [{'lr': 3e-4}]})


@pytest.mark.parametrize('changed', ['policy', 'auxiliary', 'ppo_optimizer', 'aux_optimizer', 'learning_rate', 'iteration'])
def test_resume_rejects_partial_restoration(changed):
    source = trained_state()
    restored = deepcopy(source)
    lr = 3.375e-5
    assert audit_restored(source, restored, lr) == 30000
    if changed == 'policy':
        restored['model_state_dict']['weight'].add_(1)
    elif changed == 'auxiliary':
        restored['auxiliary_state_dict']['weight'].zero_()
    elif changed == 'ppo_optimizer':
        restored['optimizer_state_dict']['state'] = {}
    elif changed == 'aux_optimizer':
        restored['auxiliary_optimizer_state_dict']['state'] = {}
    elif changed == 'learning_rate':
        lr = 3e-4
    else:
        restored['iter'] = 30000
    with pytest.raises(ValueError):
        audit_restored(source, restored, lr)


def test_rejects_inconsistent_source_progress():
    saved = trained_state()
    saved['auxiliary_state_dict']['updates'].fill_(29999)
    with pytest.raises(ValueError, match='disagree'):
        completed_updates(saved)


@pytest.mark.parametrize('mode', ['dynamics', 'control'])
def test_real_runner_load_preserves_learned_state_and_starts_next_iteration(tmp_path, mode):
    def make_runner():
        policy = ActorCritic(12, 14, 2, actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8],
                             init_noise_std=.2, noise_std_bounds=(.1, .6))
        cfg = dict(mode='dynamics', feature_dim=8, joint_dim=2, action_dim=4) if mode == 'dynamics' else None
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.alg = PPO(policy, device='cpu', auxiliary_cfg=cfg, learning_rate_bounds=(1e-5, 3e-4))
        runner.empirical_normalization = False
        runner.env = SimpleNamespace()
        runner.current_learning_iteration = 29999
        return runner
    original = make_runner()
    for opt in [original.alg.optimizer, original.alg.auxiliary_optimizer]:
        if opt is None:
            continue
        for group in opt.param_groups:
            for param in group['params']:
                param.grad = torch.ones_like(param)*.01
        opt.step()
    if mode == 'dynamics':
        original.alg.auxiliary.updates.fill_(30000)
    original.alg.optimizer.param_groups[0]['lr'] = original.alg.learning_rate = 3.375e-5
    saved = dict(iter=29999, infos=None, model_state_dict=original.alg.policy.state_dict(),
                 optimizer_state_dict=original.alg.optimizer.state_dict())
    if mode == 'dynamics':
        saved.update(auxiliary_state_dict=original.alg.auxiliary.state_dict(),
                     auxiliary_optimizer_state_dict=original.alg.auxiliary_optimizer.state_dict(),
                     auxiliary_cfg=original.alg.auxiliary_cfg)
    source, resumed = tmp_path/'source', tmp_path/'resumed'
    for run in [source, resumed]:
        (run/'params').mkdir(parents=True)
        (run/'params/env.yaml').write_text('scene: {num_envs: 64}\n')
        (run/'params/agent.yaml').write_text('policy: {}\nalgorithm: {}\nseed: 42\nnum_steps_per_env: 24\nempirical_normalization: false\n')
    checkpoint = source/'model_29999.pt'
    torch.save(saved, checkpoint)
    runner = make_runner()
    runner.load(checkpoint)
    evidence = record_resume(runner, checkpoint, resumed, mode=mode)
    assert runner.current_learning_iteration == 30000
    assert evidence['exact_learned_state_restored']
    if mode == 'dynamics':
        assert int(runner.alg.auxiliary.updates) == 30000
    else:
        assert runner.alg.auxiliary is runner.alg.auxiliary_optimizer is runner.alg.auxiliary_cfg is None
        assert evidence['auxiliary_optimizer_state_entries'] == 0
        snapshot = torch.load(resumed/'resume_initial_policy.pt', weights_only=False)
        assert not any(k.startswith('auxiliary_') for k in snapshot)
    assert runner.alg.learning_rate == 3.375e-5
    assert list(range(runner.current_learning_iteration, runner.current_learning_iteration+20000))[-1] == 49999


@pytest.mark.parametrize('changed', ['auxiliary_checkpoint', 'optimizer', 'policy', 'learning_rate', 'initial_snapshot'])
def test_plain_ppo_resume_rejects_wrong_or_partial_state(changed):
    source = trained_state()
    if changed == 'auxiliary_checkpoint':
        with pytest.raises(ValueError, match='auxiliary'):
            completed_updates(source, mode='control')
        return
    for key in ('auxiliary_cfg', 'auxiliary_state_dict', 'auxiliary_optimizer_state_dict'):
        del source[key]
    restored = deepcopy(source)
    assert completed_updates(source, mode='control') == 30000
    with pytest.raises(ValueError):
        completed_updates(source)  # Dynamics must not silently accept a headless checkpoint.
    lr = 3.375e-5
    if changed == 'optimizer':
        restored['optimizer_state_dict']['state'] = {}
    elif changed == 'policy':
        restored['model_state_dict']['weight'].add_(1)
    elif changed == 'learning_rate':
        lr = 3e-4
    else:
        source['infos'] = {'before_first_update': True}
    with pytest.raises(ValueError):
        audit_restored(source, restored, lr, mode='control')


def test_explicit_symmetry_removal_keeps_strict_audit_for_every_other_setting(tmp_path):
    old = dict(policy={'actor_hidden_dims': [512, 256, 128]},
               algorithm={'learning_rate': .0003, 'symmetry_cfg': {
                   'use_data_augmentation': True, 'use_mirror_loss': True, 'mirror_loss_coeff': .1}},
               seed=42, num_steps_per_env=24, empirical_normalization=False)
    new = deepcopy(old)
    new['algorithm']['symmetry_cfg'] = None
    source, resumed = tmp_path/'source', tmp_path/'resumed'
    for run, agent in [(source, old), (resumed, new)]:
        (run/'params').mkdir(parents=True)
        (run/'params/env.yaml').write_text('reward: {liftoff_load_transfer: {weight: 0.2}}\n')
        (run/'params/agent.yaml').write_text(yaml.safe_dump(agent))
    with pytest.raises(ValueError):
        verify_recipe(source, resumed)
    assert verify_recipe(source, resumed, disable_symmetry=True)
    new['algorithm']['learning_rate'] = .001
    (resumed/'params/agent.yaml').write_text(yaml.safe_dump(new))
    with pytest.raises(ValueError):
        verify_recipe(source, resumed, disable_symmetry=True)
    new['algorithm']['learning_rate'] = .0003
    new['algorithm']['symmetry_cfg'] = {'use_data_augmentation': True, 'use_mirror_loss': False}
    (resumed/'params/agent.yaml').write_text(yaml.safe_dump(new))
    with pytest.raises(ValueError):
        verify_recipe(source, resumed, disable_symmetry=True)
    new['algorithm']['symmetry_cfg'] = None
    (resumed/'params/agent.yaml').write_text(yaml.safe_dump(new))
    (resumed/'params/env.yaml').write_text('reward: {liftoff_load_transfer: {weight: 1.0}}\n')
    with pytest.raises(ValueError):
        verify_recipe(source, resumed, disable_symmetry=True)
