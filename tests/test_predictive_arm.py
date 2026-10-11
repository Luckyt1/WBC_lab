"""Constraints, lookahead geometry, planner detours, and v3 migration checks."""
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from test_tang_commands import commands

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
native=importlib.import_module('tang_cpu.wbc._arm_ruckig')
guard=importlib.import_module('tang_cpu.wbc.arm_guard')
executor=importlib.import_module('tang_cpu.wbc.predictive_arm')
BANK=ROOT/'tests/fixtures/ik_banks/v3'


@pytest.mark.parametrize('limits',[(2.,4.,20.),(6.,80.,1000.)])
def test_reference_bounds_survive_target_reversals_and_braking(limits):
    velocity,acceleration,jerk=limits
    rng=np.random.default_rng(932);state=np.zeros((8,3,14));times=np.array([.02,.08,.2])
    goal=np.ones((8,14))
    for i in range(400):
        if i%43==0:goal=rng.uniform(-2,2,(8,14))
        out,bounds,status=native.calculate(state,goal,times,velocity,acceleration,jerk,i%100>75,2)
        assert (status>=0).all()
        assert np.isfinite(out).all()
        assert np.abs(out[:,:,1]).max()<=velocity+1e-8
        assert np.abs(out[:,:,2]).max()<=acceleration+1e-8
        assert np.abs(out[:,0,2]-state[:,2]).max()<=jerk*.02+1e-8
        assert np.abs(out[:,0,0]-state[:,0]).max()<=velocity*.02+1e-8
        state=out[:,0].copy()


def fixture():
    m=json.loads((BANK/'manifest.json').read_text())
    names=['torso_link','l_hip_z_link']+[f'{s}_{p}_link' for s in ('l','r') for p in ('shoulder_z','wrist_x','wrist_y','wrist_z')]
    q=torch.tensor(m['default_arm_q']).repeat(2,1)
    g=guard.ArmCollisionGuard(m['arm_joint_names'],names,torch.tensor(m['joint_limits']),'cpu',
        margin=.003,geometry_mode='obb_v2')
    e=executor.PredictiveArmExecutor(g,q,m['recipe'],names)
    return g,e,q,names


def test_moving_leg_is_predicted_before_it_reaches_arm():
    g,e,q,names=fixture();n=2;h=len(e.times)
    centers=torch.full((n,len(g.body_ids),3),10.)
    rotations=torch.zeros(n,len(g.body_ids),4);rotations[...,0]=1
    p,r=guard.arm_frames(q.reshape(n,2,7));p=p+guard.rotate(r,g.arm_centers.expand_as(p))
    index=g.body_ids.index(names.index('l_hip_z_link'))
    centers[:,index]=p[:,0,3]+torch.tensor([0.,.3,0.])
    preview=torch.zeros(n,h,3,14);preview[:,:,0]=q[:,None]
    linear=torch.zeros_like(centers);angular=torch.zeros_like(centers)
    stationary=e._clearance(preview,centers,rotations,linear,angular,q)
    linear[:,index,1]=-1.5
    moving=e._clearance(preview,centers,rotations,linear,angular,q)
    assert (stationary>=g.margin).all()
    assert (moving<g.margin).all()


def test_vectorized_collision_check_matches_obstacle_loop():
    g,e,q,names=fixture();torch.manual_seed(78)
    q=torch.randn(20,3,2,7)*.8
    centers=torch.randn(20,len(g.body_ids),3)
    rotations=torch.randn(20,len(g.body_ids),4);rotations/=rotations.norm(dim=-1,keepdim=True)
    extra=torch.rand(20,len(g.body_ids))*.005
    expected=g.box_clearance(q,centers,rotations,extra)
    actual=g.box_clearance_batched(q,centers,rotations,extra)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_partial_reset_does_not_reset_other_reference_derivatives():
    g,e,q,_=fixture();e.state[:,1]=.5;e.state[:,2]=.25
    before=e.state[1].copy()
    e.reset(torch.tensor([0]),q[:1])
    assert np.array_equal(e.state[1],before)
    assert not e.state[0,1:].any()


def test_inside_margin_reset_can_retreat_without_reference_discontinuity():
    g,e,goal,names=fixture();q=goal.clone();q[:,1]=.08;q[:,8]=-.08
    e.reset(torch.arange(2),q)
    pos=torch.zeros(2,len(names),3);rot=torch.zeros(2,len(names),4);rot[...,0]=1
    pos[:,1]=10.
    def frames():
        p,r=guard.arm_frames(q.reshape(2,2,7));pos[:,2:]=p.reshape(2,8,3);rot[:,2:]=r.reshape(2,8,4)
        return g.obstacle_frames(pos,rot,0)
    c,r=frames();initial=g.box_clearance(q.reshape(2,1,2,7),c,r).min()
    assert initial<g.margin
    recovered=0.
    for _ in range(220):
        frames();previous=q.clone();q=e.project(goal,q,pos,rot,0)
        assert (q-previous).abs().max()<=e.vmax*e.dt+1e-5
        recovered+=float(e.metrics['ArmGuard/outward_recovery_fraction'])
    c,r=frames();final=g.box_clearance(q.reshape(2,1,2,7),c,r).min()
    assert recovered>0 and final>=g.margin-1e-5
    assert (q-goal).abs().max()<.03


def test_rrt_connect_routes_around_blocked_direct_edge():
    pytest.importorskip('pinocchio')
    from plan_ik_arm_bank_v3 import rrt_connect
    def edge(a,b):
        points=np.linspace(a,b,100)
        return not ((np.abs(points[:,0])<.25)&(np.abs(points[:,1])<.35)).any()
    start,goal=np.array([-.8,0.]),np.array([.8,0.])
    assert not edge(start,goal)
    route=rrt_connect(start,goal,np.full(2,-1.),np.ones(2),edge,np.random.default_rng(23))
    assert route is not None
    np.testing.assert_allclose(route[0],start);np.testing.assert_allclose(route[-1],goal)
    assert all(edge(a,b) for a,b in zip(route[:-1],route[1:]))
    assert rrt_connect(start,goal,-np.ones(2),np.ones(2),lambda a,b:False,np.random.default_rng(0),max_nodes=5) is None


def test_v3_preserves_full_goal_pool_split_and_episode_budget():
    old=json.loads((ROOT/'tests/fixtures/ik_banks/v2/manifest.json').read_text())
    new=json.loads((BANK/'manifest.json').read_text())
    pool=lambda m:{t['id']:t['split'] for t in m['targets']+m['rejected_targets']}
    assert pool(old)==pool(new)
    assert set(t['id'] for t in old['targets'])<=set(t['id'] for t in new['targets'])
    r=new['recipe']
    for split in ('train','test'):
        data=np.load(BANK/(split+'.npz'));d=data['min_durations']
        max_cycle=(r['prepare_s'][1]+np.maximum(d[:,0],r['reach_s'][1])+r['hold_s'][1]+
                   np.maximum(d[:,1],r['return_s'][1])+r['rest_s'][1]+2*r['arrival_timeout_s'])
        assert max_cycle.max()<r['max_episode_s']


def test_v3_arrival_requires_sustained_measured_pose_accuracy():
    motion=importlib.import_module('tang_cpu.wbc.ik_arm_motion')
    m=json.loads((BANK/'manifest.json').read_text());q=torch.tensor(m['default_arm_q']).reshape(1,14)
    x=motion.IkArmMotion(BANK,'train',m['arm_joint_names'],q,.02);x.reset(torch.tensor([0]),q)
    x.neutral[:]=False;x.phase[:]=1;x.phase_elapsed[:]=x.durations[:,1]
    for _ in range(20):
        # A transient crossing must not enter hold; normal joint motion from
        # body compensation does not by itself disqualify a stable wrist goal.
        offset=.03 if _%5==0 else 0.
        x.tracking_metrics(x.goal_positions[x.index]+offset,x.goal_orientations[x.index],torch.ones_like(q))
        x.step()
    assert x.phase.item()==1 and x.arrivals.item()==0
    for _ in range(20):
        x.tracking_metrics(x.goal_positions[x.index],x.goal_orientations[x.index],torch.zeros_like(q))
        x.step()
    assert x.phase.item()==2 and x.arrivals.item()==1


def test_resume_v2_to_v3_preserves_rewards_and_requires_parent():
    from audit_auxiliary_resume import expected_ik_recipe
    oldbank=ROOT/'tests/fixtures/ik_banks/v2'
    old=dict(ik_recipe='reach_hold_return_v2',arm_motion=dict(ik_bank_path=str(oldbank)),
             scene=dict(max_episode_length_s='28.0'),reward=dict(stand=-2))
    new=expected_ik_recipe(old,BANK)
    assert new['reward']==old['reward'] and new['ik_recipe']=='reach_hold_return_v3'
    old['arm_motion']['ik_bank_path']=str(ROOT/'tests/fixtures/ik_banks/v1')
    with pytest.raises(ValueError,match='descend'): expected_ik_recipe(old,BANK)
