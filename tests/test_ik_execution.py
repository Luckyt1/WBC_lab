"""Regression checks for blocked reaches, live waist correction and arrival gates."""
import importlib
import json
from pathlib import Path
import torch
import pytest
from test_tang_commands import commands

guard=importlib.import_module('tang_cpu.wbc.arm_guard')
tracker=importlib.import_module('tang_cpu.wbc.arm_pose_tracker')
motion=importlib.import_module('tang_cpu.wbc.ik_arm_motion')
ROOT=Path(__file__).resolve().parents[1]
BANK=ROOT/'tests/fixtures/ik_banks/v2'


def test_obb_preserves_clear_neutral_pose_and_blocks_torso_penetration():
    old=json.loads((ROOT/'tests/fixtures/ik_banks/v1/manifest.json').read_text())
    g=guard.ArmCollisionGuard(old['arm_joint_names'],['torso_link'],torch.tensor(old['joint_limits']),
                            'cpu',margin=.003,max_velocity=6.,geometry_mode='obb_v2')
    pos=torch.zeros(1,1,3);rot=torch.tensor([[[1.,0.,0.,0.]]])
    q=torch.tensor(old['default_arm_q']).reshape(1,14)
    result=g.project(q,q,q,pos,rot,0)
    torch.testing.assert_close(result,q,atol=1e-7,rtol=0)
    bad=q.clone();bad[:,1]=-.6;bad[:,8]=.6
    for _ in range(100):
        result=g.project(bad,q,q,pos,rot,0)
        assert (result-q).abs().max()<=.120001
        q=result
    c,r=g.obstacle_frames(pos,rot,0)
    assert g.clearance(q.reshape(1,1,2,7),c,r).min()>=.003-1e-6
    assert g.clearance(bad.reshape(1,1,2,7),c,r).min()<0


def test_spatial_jacobian_matches_finite_differences():
    torch.manual_seed(3431)
    q=torch.randn(8,2,7,dtype=torch.float64)*.5
    p,r,J=tracker.kinematics(q)
    eps=1e-6
    for j in range(7):
        changed=q.clone();changed[...,j]+=eps
        pp,rr,_=tracker.kinematics(changed)
        numerical=torch.cat(((pp-p)/eps,tracker.rotation_error(rr,r)/eps),-1)
        torch.testing.assert_close(numerical,J[...,j],atol=5e-7,rtol=2e-5)


def test_guard_fast_path_matches_full_candidate_search():
    m=json.loads((BANK/'manifest.json').read_text());torch.manual_seed(992)
    n=48
    g=guard.ArmCollisionGuard(m['arm_joint_names'],['torso_link'],torch.tensor(m['joint_limits']).repeat(n,1,1),
                            'cpu',margin=.003,max_velocity=6.,geometry_mode='obb_v2')
    pos=torch.zeros(n,1,3);rot=torch.tensor([[[1.,0.,0.,0.]]]).repeat(n,1,1)
    previous=torch.tensor(m['default_arm_q']).repeat(n,1)+torch.randn(n,14)*.2
    desired=previous+torch.randn(n,14)*.3
    actual=previous+torch.randn(n,14)*.02
    expected=g._project_all(desired,previous,actual,pos,rot,0)[0]
    expected_metrics={k:v.clone() for k,v in g.metrics.items()}
    result=g.project(desired,previous,actual,pos,rot,0)
    torch.testing.assert_close(result,expected)
    for k in expected_metrics:torch.testing.assert_close(g.metrics[k],expected_metrics[k])


def test_correction_compensates_waist_without_changing_pelvis_goal():
    m=json.loads((BANK/'manifest.json').read_text());t=tracker.ArmPoseTracker(m,'cpu')
    guide=torch.tensor([[-1.,.5,0.,.8,0.,0.,0.,-1.,-.5,0.,.8,0.,0.,0.]]).repeat(3,1)
    pp=torch.zeros(3,3);pq=torch.tensor([[1.,0.,0.,0.]]).repeat(3,1)
    tp=torch.tensor(m['nominal_torso_in_pelvis']['position']).repeat(3,1)
    tq=guard.axis_quat(torch.tensor([-.08,0.,.08]),1)
    target_p,target_q=t.guide_pose(guide)
    result=t.correct(guide,pp,pq,tp,tq)
    p,r,_=tracker.kinematics(result.reshape(-1,2,7))
    p=tp[:,None]+guard.rotate(tq[:,None],p);r=guard.quat_mul(tq[:,None],r)
    assert (p-target_p).norm(dim=-1).max()<.003
    assert tracker.rotation_error(target_q,r).norm(dim=-1).max()<.01


def instance():
    m=json.loads((BANK/'manifest.json').read_text())
    x=motion.IkArmMotion(BANK,'train',m['arm_joint_names'],torch.tensor(m['default_arm_q']).reshape(1,14),.02)
    x.reset(torch.tensor([0]),x.defaults);x.neutral[:]=False
    x.durations[:]=torch.tensor([.1,.2,.3,.2,.1])
    return x


def test_not_at_goal_never_starts_hold_and_timeout_is_explicit():
    x=instance()
    for _ in range(25): x.step()
    assert x.phase.item()==1 and x.arrivals.item()==0
    for _ in range(150): x.step()
    assert x.phase.item()==3 and x.reach_timeouts.item()==1


def test_hold_starts_after_actual_arrival_dwell_and_only_one_cycle():
    x=instance()
    for _ in range(20): x.step()
    assert x.phase.item()==1
    for _ in range(4):
        x.tracking_metrics(x.goal_positions[x.index],x.goal_orientations[x.index]);x.step()
        assert x.phase.item()==1
    for _ in range(2):
        x.tracking_metrics(x.goal_positions[x.index],x.goal_orientations[x.index]);x.step()
    assert x.phase.item()==2 and x.arrivals.item()==1
    for _ in range(50):
        x.tracking_metrics(x.home_p,x.home_q);x.step()
    assert x.phase.item()==4 and x.completed_cycles.item()==1
    selected=x.index.clone()
    for _ in range(100): x.step(remaining_s=torch.tensor([.3]))
    assert x.completed_cycles.item()==1 and torch.equal(selected,x.index)


def test_validated_bank_has_endpoints_and_preserves_rejected_evaluation_goals():
    m=json.loads((BANK/'manifest.json').read_text())
    for t in m['targets']:
        for leg in ('outbound','inbound'):
            v=t['execution_validation'][leg]
            assert v['endpoint_error'][0]<=.005 and v['endpoint_error'][1]<=.03
            assert v['minimum_nominal_clearance_m']>=.003
    old=json.loads((ROOT/'tests/fixtures/ik_banks/v1/manifest.json').read_text())
    assert {t['id'] for t in m['targets']+m['rejected_targets']}=={t['id'] for t in old['targets']}
    assert {t['id'] for t in m['targets']}.isdisjoint({t['id'] for t in m['rejected_targets']})


def test_resume_allows_only_declared_arm_execution_changes(tmp_path):
    import sys,yaml
    sys.path.insert(0,str(ROOT/'tools'))
    from audit_auxiliary_resume import expected_ik_recipe,verify_recipe
    from copy import deepcopy
    old_bank=ROOT/'tests/fixtures/ik_banks/v1'
    before={'ik_recipe':'reach_hold_return_v1','scene':{'max_episode_length_s':'24.0','num_envs':'4096'},
            'arm_motion':{'ik_bank_path':str(old_bank),'ik_bank_sha256':'old','guard_margin':'0.012','guard_max_velocity':'6.0'},
            'reward':{'stand_weight':'4.0'}}
    after=expected_ik_recipe(before,BANK)
    a,b=tmp_path/'a',tmp_path/'b'
    agent=dict(policy={},algorithm={'symmetry':'enabled'},seed=42,num_steps_per_env=24,empirical_normalization=False)
    for path,env in ((a,before),(b,after)):
        (path/'params').mkdir(parents=True)
        (path/'params/env.yaml').write_text(yaml.safe_dump(env))
        (path/'params/agent.yaml').write_text(yaml.safe_dump(agent))
    assert verify_recipe(a,b,ik_bank=BANK)
    after['reward']['stand_weight']='0.0'
    (b/'params/env.yaml').write_text(yaml.safe_dump(after))
    with pytest.raises(ValueError,match='stand_weight'):verify_recipe(a,b,ik_bank=BANK)
