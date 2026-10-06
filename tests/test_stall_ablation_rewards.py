from pathlib import Path
import sys
from types import SimpleNamespace as NS
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import stall_ablation_rewards as reward


def env():
    data=NS(body_pos_w=torch.zeros(3,2,3))
    sensor=NS(net_forces_w=torch.zeros(3,2,3),current_contact_time=torch.ones(3,2),last_air_time=torch.zeros(3,2))
    return NS(robot=NS(data=data),scene={'robot':NS(data=data)},base_quat_w=torch.tensor([[1.,0,0,0]]).repeat(3,1),
              base_projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(3,1),
              feet_body_ids=[0,1],feet_cfg=NS(body_ids=[0,1]),contact_sensor=NS(data=sensor),
              command_generator=NS(is_standing_env=torch.tensor([False,False,True])),
              episode_length_buf=torch.full((3,),10),step_dt=.02)


def test_knee_width_ignores_fore_aft_separation_and_world_heading():
    e=env();p=e.robot.data.body_pos_w
    p[:,0,:2]=torch.tensor([[0.,.27],[.3,.27],[-.3,.27]])
    cfg=NS(name='robot',body_ids=[0,1])
    torch.testing.assert_close(reward.lateral_knee_width(e,cfg),torch.ones(3))
    p[:,:,:2]=torch.stack((-p[:,:,1],p[:,:,0]),-1)
    e.base_quat_w[:]=torch.tensor([2**-.5,0.,0.,2**-.5])
    torch.testing.assert_close(reward.lateral_knee_width(e,cfg),torch.ones(3),atol=1e-6,rtol=0)
    p[:,0,0]=.27 # signed inversion is penalized
    assert (reward.lateral_knee_width(e,cfg)<.01).all()


def test_pre_liftoff_credit_requires_upright_loaded_moving_robot():
    e=env();e.contact_sensor.data.net_forces_w[:,:,2]=torch.tensor([[300.,300.],[550.,50.],[550.,50.]])
    r=reward.load_transfer(e)
    assert r[0]==r[2]==0 and 0<r[1]<1
    e.contact_sensor.data.net_forces_w.zero_()
    assert reward.load_transfer(e).sum()==0


def test_clearance_cannot_reward_both_airborne_or_standing():
    e=env();e.robot.data.body_pos_w[:,0,2]=.1
    e.contact_sensor.data.current_contact_time[:,0]=0
    torch.testing.assert_close(reward.swing_clearance(e),torch.tensor([1.,1.,0.]))
    e.contact_sensor.data.current_contact_time.zero_()
    assert reward.swing_clearance(e).sum()==0


def test_alternation_does_not_pay_repeated_same_foot_or_reset():
    e=env();s=e.contact_sensor.data
    s.current_contact_time[:]=torch.tensor([.02,1.]);s.last_air_time[:,0]=.2
    assert reward.alternating_touchdown(e).sum()==0
    assert reward.alternating_touchdown(e).sum()==0
    s.current_contact_time[:]=torch.tensor([1.,.02]);s.last_air_time[:,1]=.2
    torch.testing.assert_close(reward.alternating_touchdown(e),torch.tensor([50.,50.,0.]))
    e.episode_length_buf[:]=1
    assert reward.alternating_touchdown(e).sum()==0
