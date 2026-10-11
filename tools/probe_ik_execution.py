"""Frozen-policy Isaac runtime probe for a proposed IK execution bank."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'rsl_rl')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--num_envs',type=int,default=64)
p.add_argument('--seconds',type=float,default=40.)
p.add_argument('--stochastic',action='store_true',help='Sample the saved PPO action distribution, as training does')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();launcher=AppLauncher(args)

import hashlib
import json
import time
import torch
from legged_lab.envs.elf3_tang.elf3_env import Elf3Env
from legged_lab.envs.elf3_tang.walk_robust_cfg import Elf3WalkRobustBxiEnvCfg,Elf3WalkRobustBxiAgentCfg
from legged_lab.envs.elf3_tang.ik_training import configure_ik_training
from legged_lab.envs.elf3_tang.teleop_standing import configure_teleop_standing
from legged_lab.envs.elf3_tang.standing_position import configure_standing_position_penalty
from legged_lab.envs.elf3_tang.liftoff_rewards import configure_liftoff_rewards
from tools.fixed_pelvis_height import configure_fixed_height
from rsl_rl.runners import OnPolicyRunner

@torch.inference_mode()
def main():
    cfg,agent=Elf3WalkRobustBxiEnvCfg(),Elf3WalkRobustBxiAgentCfg()
    configure_teleop_standing(cfg,.5);configure_standing_position_penalty(cfg)
    configure_liftoff_rewards(cfg);configure_fixed_height(cfg,.8);configure_ik_training(cfg,args.bank)
    cfg.scene.num_envs=args.num_envs;cfg.scene.seed=42
    cfg.sim.device=agent.device=args.device
    env=Elf3Env(cfg,args.headless)
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=env.device)
    runner.load(str(args.checkpoint),load_optimizer=False)
    env.set_training_iteration(runner.current_learning_iteration,runner.num_steps_per_env)
    policy=runner.get_inference_policy(device=env.device)
    obs,_=env.get_observations();start=time.monotonic();maxima={};terminated=0
    for i in range(round(args.seconds/env.step_dt)):
        action=runner.alg.policy.act(obs) if args.stochastic else policy(obs)
        obs,reward,done,info=env.step(action)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        for key,value in env.arm_guard.metrics.items():
            value=float(value)
            if not torch.isfinite(torch.tensor(value)):raise ValueError(key)
            maxima[key]=max(maxima.get(key,0.),value)
        terminated+=int((done&~info['time_outs']).sum())
        if i%250==0:
            m=env.ik_motion
            print(json.dumps(dict(step=i,arrivals=float(m.arrivals),terminated=terminated,
                reach_timeouts=float(m.reach_timeouts),in_reach=int((m.phase==1).sum()),
                position_error_median=float(m.position_error.median()),arm_speed_median=float(m.arm_speed.median()),
                brake_fraction=float(env.arm_guard.metrics.get('ArmGuard/predictive_brake_fraction',0.)))),flush=True)
    speeds=env.ik_motion.arm_speed[torch.isfinite(env.ik_motion.arm_speed)]
    report=dict(checkpoint=str(args.checkpoint.resolve()),bank=str(args.bank.resolve()),num_envs=args.num_envs,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),steps=i+1,
        runtime_s=time.monotonic()-start,finite=True,training_updates=0,terminated=terminated,stochastic=args.stochastic,
        maximum_metrics=maxima,ik_metrics={k:float(v) for k,v in env.ik_motion.metrics.items()},
        final_arm_speed_percentiles=(torch.quantile(speeds,torch.tensor([.1,.5,.9],device=env.device)).tolist() if len(speeds) else None))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)

from legged_lab.utils.app_lifecycle import run_simulation
run_simulation(main,launcher.app,args.headless)
