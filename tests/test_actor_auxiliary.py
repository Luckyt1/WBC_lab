"""Check actor gradient flow, transition alignment and unchanged PPO baseline."""
from copy import deepcopy
from pathlib import Path
import sys
import importlib.util
import types

import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rsl_rl'))
from rsl_rl.modules import ActorCritic
from rsl_rl.modules.actor_auxiliary import ActorAuxiliary
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage


def policy():
    return ActorCritic(12,14,2,actor_hidden_dims=[16,8],critic_hidden_dims=[16,8],init_noise_std=.2)


def cfg(mode='dynamics'):
    return dict(mode=mode,feature_dim=8,joint_dim=2,action_dim=4,warmup_updates=0,ramp_updates=1)


def targets(n=6):
    return dict(velocity=torch.randn(n,3),contact=torch.randint(0,2,(n,2)).float(),
                applied_action=torch.randn(n,4),next_joint_velocity=torch.randn(n,2),
                next_angular_velocity=torch.randn(n,3),valid=torch.ones(n,dtype=torch.bool))


def test_head_initialization_preserves_rng_and_action_network():
    torch.manual_seed(99);p=policy();obs=torch.randn(6,12)
    before=p.actor(obs).detach().clone();state=torch.get_rng_state().clone()
    aux=ActorAuxiliary(**cfg())
    assert torch.equal(state,torch.get_rng_state())
    assert torch.equal(before,p.actor(obs))
    assert set(p.state_dict()).isdisjoint(set('auxiliary.'+k for k in aux.state_dict()))


@pytest.mark.parametrize('warmed',[False,True])
def test_aux_gradients_reach_actor_trunk_only_after_warmup(warmed):
    p=policy();aux=ActorAuxiliary(**cfg());aux.updates.fill_(int(warmed))
    loss,_=aux.loss(p,torch.randn(6,12),targets());loss.backward()
    assert (p.actor[0].weight.grad.abs().sum().item()>0)==warmed
    assert p.actor[-1].weight.grad is None
    assert all(param.grad is None for param in p.critic.parameters())
    assert aux.state_head[0].weight.grad.abs().sum()>0


def test_action_conditioning_and_done_mask():
    p=policy();aux=ActorAuxiliary(**cfg());obs=torch.randn(6,12);t=targets()
    a=aux.predict(p,obs,t['applied_action'])
    b=aux.predict(p,obs,t['applied_action']+1.)
    assert not torch.equal(a['joint_velocity'],b['joint_velocity'])
    assert torch.equal(a['velocity'],b['velocity'])
    t['valid'].zero_();_,m=aux.loss(p,obs,t)
    assert m['aux_joint_velocity_huber']==0 and m['aux_angular_velocity_huber']==0
    assert m['aux_contact_bce']>0


def test_physical_normalization_preserves_loss_under_unit_rescaling():
    p=policy();a=ActorAuxiliary(**cfg());b=deepcopy(a);obs=torch.randn(6,12);t=targets();u=deepcopy(t)
    b.velocity_scale.mul_(100);u['velocity'].mul_(100)
    b.joint_velocity_scale.mul_(10);u['next_joint_velocity'].mul_(10)
    b.angular_velocity_scale.mul_(20);u['next_angular_velocity'].mul_(20)
    torch.testing.assert_close(a.loss(p,obs,t)[0],b.loss(p,obs,u)[0])


def fill_and_update(ppo,with_aux):
    ppo.init_storage('rl',6,3,[12],[14],[2]);torch.manual_seed(847)
    for i in range(3):
        obs=torch.randn(6,12);critic=torch.randn(6,14)
        ppo.act(obs,critic)
        t=targets();infos={'auxiliary':t} if with_aux else {}
        ppo.process_env_step(torch.randn(6),torch.zeros(6,dtype=torch.bool),infos)
    ppo.compute_returns(torch.randn(6,14));return ppo.update()


def test_auxiliary_warmup_keeps_ppo_actions_and_optimizer_identical():
    torch.manual_seed(3);p=policy();q=deepcopy(p)
    plain=PPO(p,num_learning_epochs=2,num_mini_batches=3,device='cpu')
    augmented=PPO(q,num_learning_epochs=2,num_mini_batches=3,device='cpu',auxiliary_cfg=cfg())
    fill_and_update(plain,False);logs=fill_and_update(augmented,True)
    for name,value in p.state_dict().items(): torch.testing.assert_close(value,q.state_dict()[name],rtol=0,atol=0)
    assert logs['aux_actor_grad_ratio_first_batch']==0
    assert augmented.auxiliary.updates.item()==1
    assert len(augmented.auxiliary_optimizer.state)>0


def test_minibatch_auxiliary_labels_follow_observations():
    storage=RolloutStorage('rl',4,3,[1],[1],[1],device='cpu')
    for k in range(3):
        t=storage.Transition();x=torch.arange(4).float()[:,None]+k*4
        t.observations=t.privileged_observations=t.actions=t.values=t.action_mean=t.action_sigma=x
        t.rewards=t.dones=t.actions_log_prob=torch.zeros(4)
        t.auxiliary={'identity':x+100};storage.add_transitions(t)
    for batch in storage.mini_batch_generator(3,2):
        torch.testing.assert_close(storage.auxiliary_batch['identity'],batch[0]+100)


def test_active_auxiliary_balancing_and_head_checkpoint_roundtrip():
    p=policy();ppo=PPO(p,num_learning_epochs=1,num_mini_batches=2,device='cpu',auxiliary_cfg=cfg())
    ppo.auxiliary.updates.fill_(1)
    logs=fill_and_update(ppo,True)
    assert 0 < logs['aux_actor_grad_ratio_first_batch'] <= .10001
    copy=ActorAuxiliary(**cfg());copy.load_state_dict(ppo.auxiliary.state_dict())
    obs=torch.randn(6,12);action=torch.randn(6,4)
    for name,value in copy.predict(p,obs,action).items():
        torch.testing.assert_close(value,ppo.auxiliary.predict(p,obs,action)[name])


def test_disabled_ppo_matches_archived_implementation_exactly():
    source=(ROOT/'tests/fixtures/ppo_before_auxiliary.py').read_bytes()
    namespace={};exec(compile(source,'archived_ppo.py','exec'),namespace)
    torch.manual_seed(3);p=policy();q=deepcopy(p)
    old=namespace['PPO'](p,num_learning_epochs=2,num_mini_batches=3,device='cpu')
    new=PPO(q,num_learning_epochs=2,num_mini_batches=3,device='cpu')
    left,right=fill_and_update(old,False),fill_and_update(new,False)
    assert left==right
    for name,value in p.state_dict().items(): torch.testing.assert_close(value,q.state_dict()[name],rtol=0,atol=0)


def test_transition_targets_capture_before_reset_and_are_immutable():
    pkg=types.ModuleType('aux_cpu');pkg.__path__=[str(ROOT/'legged_lab/envs/elf3_tang')];sys.modules[pkg.__name__]=pkg
    import importlib
    capture=importlib.import_module('aux_cpu.auxiliary').transition_targets
    before=dict(velocity=torch.ones(2,3),contact=torch.ones(2,2),joint_velocity=torch.ones(2,2),angular_velocity=torch.ones(2,3))
    after={k:v*2 for k,v in before.items()};action=torch.ones(2,4);done=torch.tensor([False,True])
    target=capture(before,after,action,done);action.zero_();done.zero_()
    assert target['applied_action'].sum()==8
    assert target['valid'].tolist()==[True,False]
    assert target['velocity'].mean()==1 and target['next_joint_velocity'].mean()==2
