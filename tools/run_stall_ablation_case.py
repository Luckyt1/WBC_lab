"""One matched short continuation plus fixed-command evaluation; experimental only."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'rsl_rl')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--variant',choices=('baseline','knee','guidance','exploration','combined'),required=True)
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
p.add_argument('--updates',type=int,default=300)
p.add_argument('--num-envs',type=int,default=1024)
p.add_argument('--seed',type=int,default=42)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
launcher=AppLauncher(args)

import copy
import hashlib
import json
import math
from types import MethodType
import torch
from isaaclab.managers import RewardTermCfg
from isaaclab.utils.io import dump_yaml
from legged_lab.envs.elf3_tang.scratch_cfg import Elf3ScratchBxiEnvCfg,Elf3ScratchBxiAgentCfg
from legged_lab.envs.elf3_tang.elf3_env import Elf3Env
from legged_lab.envs.elf3_tang import rewards
from legged_lab.envs.elf3_tang.commands import yaw_frame_velocity,wrap_to_pi
from rsl_rl.runners import OnPolicyRunner
import stall_ablation_rewards as experimental


@torch.inference_mode()
def evaluate(env,runner):
    scenarios=[('stand',[0,0,0,.8235,0]),('forward',[.3,0,0,.8235,0]),
               ('backward',[-.3,0,0,.8235,0]),('turn_pos',[0,0,.5,.8235,0]),('turn_neg',[0,0,-.5,.8235,0])]
    repeats=32;n=5*repeats
    assert env.num_envs>=n
    ids=torch.arange(env.num_envs,device=env.device)
    target=torch.tensor([v for _,v in scenarios],device=env.device).repeat_interleave(repeats,0)
    all_target=target[:1].expand(env.num_envs,-1).clone();all_target[:n]=target
    def fixed(g,env_ids):
        g.target[env_ids]=all_target[env_ids]
        g.full_command[env_ids,:3]=all_target[env_ids,:3]
        g.is_standing_env[env_ids]=all_target[env_ids,:3].abs().sum(-1)<1e-6
        g.time_left[env_ids]=1.e9
    env.command_generator._resample=MethodType(fixed,env.command_generator)
    env.seed(20260921);env.reset(ids);env.event_manager.reset(ids)
    env.amass_neutral_mask.zero_() # all variants tested with 5% AMASS, no neutral-reference advantage
    policy=runner.get_inference_policy()
    original_check=env.check_reset;capture={}
    knees=[i for i,j in enumerate(env.policy_joint_ids) if 'knee' in env.robot.data.joint_names[j]]
    def checked():
        done,timeout=original_check()
        d=env.robot.data
        actual=torch.cat((yaw_frame_velocity(env.base_quat_w,env.base_lin_vel_w),env.base_ang_vel_w[:,2:3],
                          rewards.pelvis_height(env)[:,None],rewards.waist_yaw(env)[:,None]),-1)
        contact=env.contact_sensor.data.current_contact_time[:,env.feet_cfg.body_ids]>0
        capture.update(actual=actual[:n].clone(),xy=d.body_pos_w[:n,env.pelvis_body_id,:2].clone(),
                       contact=contact[:n].clone(),done=(done & ~timeout)[:n].clone(),
                       clip=(env.raw_actions[:n,knees]!=env.bounded_actions[:n,knees]).float())
        return done,timeout
    env.check_reset=checked
    obs,_=env.get_observations();alive=torch.ones(n,dtype=torch.bool,device=env.device)
    h={k:[] for k in ('actual','xy','contact','clip','valid')}
    for step in range(800):
        obs,_,_,_=env.step(policy(obs))
        if step>=200:
            for k in h:h[k].append(alive.clone() if k=='valid' else capture[k].clone())
        alive &= ~capture['done']
    arrays={k:torch.stack(v) for k,v in h.items()}
    error=arrays['actual']-target[None]
    error[:,:,4]=wrap_to_pi(error[:,:,4])
    rows=[]
    for i,(name,cmd) in enumerate(scenarios):
        sl=slice(i*repeats,(i+1)*repeats);mask=arrays['valid'][:,sl]
        row=dict(scenario=name,target=cmd,survived=int(alive[sl].sum()),trials=repeats,valid_samples=int(mask.sum()))
        if mask.any():
            row.update(actual_mean=arrays['actual'][:,sl][mask].mean(0).tolist(),
                       mae=error[:,sl][mask].abs().mean(0).tolist(),
                       single_support_fraction=float((arrays['contact'][:,sl][mask].sum(-1)==1).float().mean()),
                       knee_clipped_fraction=arrays['clip'][:,sl][mask].mean(0).tolist())
            delta=(arrays['xy'][-1,sl]-arrays['xy'][0,sl]).norm(dim=-1)[alive[sl]]
            row['drift_survivors_mean_m']=float(delta.mean()) if delta.numel() else None
        rows.append(row)
    return dict(eval_seed=20260921,repeats=repeats,warmup_s=4,measure_s=12,arm_scale=.05,
                method='Deterministic fixed commands; 5% AMASS+guard in every trial; no observation noise/push/reset velocity. Startup mass/friction from paired training seed retained. First trial only, including terminal transition; post-warmup tracking.',
                scenarios=rows)


def main():
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'started.json').exists():raise ValueError('Refusing to overwrite an existing experiment')
    assert args.updates>0 and args.num_envs>=160
    cfg,agent=Elf3ScratchBxiEnvCfg(),Elf3ScratchBxiAgentCfg()
    cfg.scene.num_envs=args.num_envs;cfg.scene.seed=agent.seed=args.seed
    cfg.device=agent.device=args.device
    guided=args.variant in ('guidance','combined')
    if args.variant in ('knee','combined'):
        cfg.reward.knee_stance_width.func=experimental.lateral_knee_width
    if guided:
        cfg.reward.ablation_load_transfer=RewardTermCfg(func=experimental.load_transfer,weight=.5)
        cfg.reward.ablation_swing_clearance=RewardTermCfg(func=experimental.swing_clearance,weight=.5)
        cfg.reward.ablation_alternating_touchdown=RewardTermCfg(func=experimental.alternating_touchdown,weight=.5)
        cfg.reward.feet_air_time.weight=1.5
    source=Path(args.checkpoint)
    manifest=dict(variant=args.variant,seed=args.seed,updates=args.updates,num_envs=args.num_envs,
                  checkpoint=str(source.resolve()),checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  pretrained_iteration=2000,changed_terms=[],status='initializing')
    (out/'started.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=Elf3Env(cfg,args.headless)
    if guided:
        for name,scale in (('track_body_height_exp',.5),('height_tracking_error',.5),
                           ('track_body_roll_exp',.5),('track_body_pitch_exp',.5),('center_of_gravity_tracking',.25)):
            term=copy.deepcopy(env.reward_manager.get_term_cfg(name))
            term.params=dict(original_func=term.func,original_params=term.params,moving_scale=scale)
            term.func=experimental.scaled_during_motion
            env.reward_manager.set_term_cfg(name,term)
            manifest['changed_terms'].append(dict(name=name,moving_scale=scale))
    agent.logger='tensorboard';agent.save_interval=100
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(out),device=env.device)
    runner.load(str(source),load_optimizer=True)
    saved=torch.load(source,map_location=env.device,weights_only=False)
    assert runner.current_learning_iteration==saved['iter']
    assert all(torch.equal(v,saved['model_state_dict'][k]) for k,v in runner.alg.policy.state_dict().items())
    assert len(runner.alg.optimizer.state)==len(saved['optimizer_state_dict']['state'])
    manifest['optimizer_entries']=len(runner.alg.optimizer.state)
    manifest['initial_std']=runner.alg.policy.std.detach().tolist()
    if args.variant in ('exploration','combined'):
        policy=runner.alg.policy
        original=policy.constrain_parameters
        minimum=torch.full_like(policy.std,.1)
        names=[env.robot.data.joint_names[j] for j in env.policy_joint_ids]
        for i,name in enumerate(names):
            if '_hip_' in name:minimum[i]=.25
        @torch.no_grad()
        def constrain(self):
            original();self.std.copy_(torch.maximum(self.std,minimum))
        policy.constrain_parameters=MethodType(constrain,policy)
        policy.constrain_parameters()
        manifest['hip_std_floor']=.25
    manifest.update(status='training',effective_initial_std=runner.alg.policy.std.detach().tolist())
    (out/'started.json').write_text(json.dumps(manifest,indent=2)+'\n')
    dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
    env.seed(args.seed);env.reset(torch.arange(env.num_envs,device=env.device))
    runner.learn(args.updates,init_at_random_ep_len=True)
    if runner.writer:runner.writer.close()
    runner.save(str(out/'final.pt'))
    assert all(torch.isfinite(v).all() for v in runner.alg.policy.state_dict().values())
    report=evaluate(env,runner)
    report.update(manifest, status='complete', final_iteration=runner.current_learning_iteration,
                  final_checkpoint=str((out/'final.pt').resolve()),final_std=runner.alg.policy.std.tolist())
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('ABLATION_COMPLETE',json.dumps(report),flush=True)


from legged_lab.utils.app_lifecycle import run_simulation
run_simulation(main,launcher.app,args.headless)
