"""Frozen-policy stochastic rollouts with per-environment traces; no optimizer updates."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'rsl_rl')]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--num_envs', type=int, default=2048)
parser.add_argument('--steps', type=int, default=2400)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--threshold', type=float, default=40.)
parser.add_argument('--max_traces', type=int, default=8)
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--arms_fixed', action='store_true')
parser.add_argument('--bound_action_history', action='store_true',
                    help='Diagnostic ablation only: bound observed action history to feasible joint targets')
parser.add_argument('--bound_policy_actions', action='store_true',
                    help='Diagnostic ablation only: bound actions before the delay/history buffer and PD target')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

from collections import deque
import hashlib
import json
import time
import torch
from legged_lab.envs.elf3_tang.elf3_env import Elf3Env
from legged_lab.envs.elf3_tang.walk_bxi_cfg import Elf3WalkTangBxiEnvCfg, Elf3WalkTangBxiAgentCfg
from legged_lab.envs.elf3_tang.commands import rotate_inverse
from legged_lab.envs.elf3_tang import rewards as tang_rewards
from rsl_rl.runners import OnPolicyRunner


@torch.inference_mode()
def main():
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg, agent = Elf3WalkTangBxiEnvCfg(), Elf3WalkTangBxiAgentCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.seed = args.seed
    cfg.device = agent.device = args.device
    env = Elf3Env(cfg, args.headless)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=env.device)
    runner.load(args.checkpoint, load_optimizer=False)
    runner.eval_mode()
    policy = runner.alg.policy
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    low = ((env.robot.data.soft_joint_pos_limits[:, env.policy_joint_ids, 0]
            - env.robot.data.default_joint_pos[:, env.policy_joint_ids]) / env.action_scale)
    high = ((env.robot.data.soft_joint_pos_limits[:, env.policy_joint_ids, 1]
             - env.robot.data.default_joint_pos[:, env.policy_joint_ids]) / env.action_scale)
    if args.bound_action_history:
        original_observations = env.compute_current_observations
        def bounded_observations():
            actor, critic = original_observations()
            bounded = actor[:, 41:56].clamp(min=low, max=high)
            actor[:, 41:56] = bounded
            critic[:, 41:56] = bounded
            return actor, critic
        env.compute_current_observations = bounded_observations
    if args.arms_fixed:
        env.arm_targets = env.robot.data.default_joint_pos[:, env.arm_joint_ids].clone()
    env.seed(args.seed)
    env.reset(torch.arange(env.num_envs, device=env.device))
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    obs, extras = env.get_observations()
    critic_obs = extras['observations']['critic']
    captured = {}
    pushed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    push_cfg = env.event_manager.get_term_cfg('push_robot')
    original_push = push_cfg.func
    def logged_push(sim_env, env_ids, **kwargs):
        pushed[env_ids] = True
        return original_push(sim_env, env_ids, **kwargs)
    push_cfg.func = logged_push
    env.event_manager.set_term_cfg('push_robot', push_cfg)
    reward_compute = env.reward_manager.compute
    def capture_compute(dt):
        reward = reward_compute(dt)
        data = env.robot.data
        captured.update(
            reward=reward.clone(),
            reward_terms=env.reward_manager._step_reward.clone() * dt,
            executed_action=env.action.clone(),
            post_joint_pos=data.joint_pos.clone(),
            post_joint_vel=data.joint_vel.clone(),
            computed_torque=data.computed_torque.clone(),
            applied_torque=data.applied_torque.clone(),
            post_root_state=data.root_state_w[:, 7:13].clone(),
            height=tang_rewards.pelvis_height(env).clone(),
            contact_force=env.contact_sensor.data.net_forces_w.norm(dim=-1).amax(-1),
            terminated=(env.reset_buf & ~env.time_out_buf).clone(),
            timeout=env.time_out_buf.clone(),
            command=env.command_generator.full_command.clone(),
            arm_target_after=env.arm_position_target.clone(),
            pushed=pushed.clone(),
        )
        return reward
    env.reward_manager.compute = capture_compute
    ring = deque(maxlen=40)
    pending = []
    records = []
    cooldown = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    events = []
    stats = []
    maxima = {'action_abs':0., 'mean_abs':0., 'joint_velocity_abs':0., 'observation_abs':0., 'value_abs':0., 'sent_action_abs':0., 'executed_action_abs':0.}
    minimum_reward = 0.
    count_extreme = 0
    count_bad_reward = 0
    unique_extreme = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    started = time.time()

    def select(frame, env_id):
        return {key: (value[env_id].cpu().clone() if isinstance(value, torch.Tensor) else value)
                for key, value in frame.items()}

    for step in range(args.steps):
        pushed.zero_()
        means = policy.act_inference(obs)
        actions = means if args.deterministic else means + torch.randn_like(means) * policy.std
        data = env.robot.data
        frame = {'step':step,'obs':obs.clone(),'critic_obs':critic_obs.clone(),
                 'mean':means.clone(),'raw_action':actions.clone(),
                 'value':policy.evaluate(critic_obs).squeeze(-1),
                 'pre_joint_pos':data.joint_pos.clone(),'pre_joint_vel':data.joint_vel.clone(),
                 'pre_root_state':data.root_state_w[:, 7:13].clone(),
                 'episode_step':env.episode_length_buf.clone(),
                 'amass_index':env.amass_index.clone(),
                 'arm_target_before':env.arm_position_target.clone()}
        sent_actions = actions.clamp(min=low,max=high) if args.bound_policy_actions else actions
        frame['sent_action'] = sent_actions.clone()
        next_obs, reward, done, extras = env.step(sent_actions)
        frame.update({k:v.clone() for k,v in captured.items()})
        frame['done'] = done.clone()
        ring.append(frame)
        for trace in pending:
            trace['frames'].append(select(frame, trace['env_id']))
        completed = [tr for tr in pending if step >= tr['trigger_step'] + 24]
        for trace in completed:
            path = out/f"trace_{trace['trigger_step']:05d}_env_{trace['env_id']}.pt"
            torch.save(trace,path)
            records.append(str(path))
            pending.remove(trace)
        action_max_per_env = actions.abs().amax(-1)
        extreme = (action_max_per_env > args.threshold) | (reward < -100.)
        unique_extreme |= extreme
        count_extreme += int(extreme.sum())
        count_bad_reward += int((reward < -100.).sum())
        candidates = extreme & (cooldown < step)
        if candidates.any() and len(records) + len(pending) < args.max_traces:
            priority = torch.where(candidates, action_max_per_env, -torch.ones_like(action_max_per_env))
            env_id = int(priority.argmax())
            trace = {'env_id':env_id,'trigger_step':step,'frames':[select(f,env_id) for f in ring]}
            pending.append(trace)
            cooldown[env_id] = step + 500
            event = {'step':step,'env_id':env_id,'action_max':float(action_max_per_env[env_id]),
                     'reward':float(reward[env_id]),'mean_max':float(means[env_id].abs().max()),
                     'obs_max':float(obs[env_id].abs().max())}
            events.append(event)
            print('SPIKE_CAPTURE',json.dumps(event),flush=True)
        for key, value in [('action_abs',actions),('mean_abs',means),('joint_velocity_abs',frame['post_joint_vel']),('observation_abs',obs),('value_abs',frame['value']),('sent_action_abs',sent_actions),('executed_action_abs',frame['executed_action'])]:
            maxima[key] = max(maxima[key],float(value.abs().max()))
        minimum_reward = min(minimum_reward,float(reward.min()))
        if not (torch.isfinite(obs).all() and torch.isfinite(actions).all() and torch.isfinite(reward).all()):
            raise RuntimeError(f'Nonfinite tensor at step {step}')
        if step % 50 == 0:
            stats.append({'step':step,'reward_mean':float(reward.mean()),'reward_min':float(reward.min()),
                          'action_max':float(actions.abs().max()),'mean_max':float(means.abs().max()),
                          'value_max':float(frame['value'].abs().max()),'joint_vel_max':float(frame['post_joint_vel'].abs().max()),
                          'done_count':int(done.sum()),'extreme_count':int(extreme.sum())})
        if (step + 1) % 200 == 0:
            print('DIAG_PROGRESS',step+1,'/',args.steps,'maxima',maxima,'min_reward',minimum_reward,
                  'extreme_transitions',count_extreme,'traces',len(records)+len(pending),flush=True)
        obs = next_obs
        critic_obs = extras['observations']['critic']
    for trace in pending:
        path = out/f"trace_{trace['trigger_step']:05d}_env_{trace['env_id']}.pt"
        torch.save(trace,path)
        records.append(str(path))
    # Include an ordinary rollout sample for control comparisons even if no spike occurs.
    torch.save({'env_id':0,'trigger_step':args.steps-1,'frames':[select(f,0) for f in ring]},out/'ordinary_trace.pt')
    assert all(torch.equal(v, policy.state_dict()[k]) for k,v in before.items()), 'Policy must remain frozen'
    report = {'args':vars(args),'iteration':runner.current_learning_iteration,'elapsed_s':time.time()-started,
              'transitions':env.num_envs*args.steps,'maxima':maxima,'minimum_step_reward':minimum_reward,
              'extreme_transition_count':count_extreme,'reward_below_minus100_count':count_bad_reward,
              'unique_extreme_envs':int(unique_extreme.sum()),'events':events,'traces':records,'stats':stats,
              'reward_names':list(env.reward_manager._term_names),'policy_joint_names':[data.joint_names[i] for i in env.policy_joint_ids],
              'all_joint_names':list(data.joint_names),'std':policy.std.cpu().tolist(),
              'action_low':low[0].cpu().tolist(),'action_high':high[0].cpu().tolist(),
              'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'weights_unchanged':True,'dt':env.step_dt}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('DIAGNOSTIC_COMPLETE',str(out/'report.json'),flush=True)


from legged_lab.utils.app_lifecycle import run_simulation
run_simulation(main,app,args.headless)
