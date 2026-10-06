"""Exercise action ownership, delayed history, PD targets and PPO continuation in PhysX."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--num_envs', type=int, default=2048)
parser.add_argument('--train_iterations', type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import json
import torch
from legged_lab.envs.elf3_tang.elf3_env import Elf3Env
from legged_lab.envs.elf3_tang.walk_bxi_cfg import Elf3WalkTangBxiEnvCfg, Elf3WalkTangBxiAgentCfg
from legged_lab.envs.elf3_tang import rewards
from rsl_rl.runners import OnPolicyRunner


def main():
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg, agent = Elf3WalkTangBxiEnvCfg(), Elf3WalkTangBxiAgentCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.seed = agent.seed = 42
    cfg.sim.device = agent.device = args.device
    env = Elf3Env(cfg, args.headless)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(out / 'training'), device=env.device)
    runner.load(args.checkpoint)  # restore both model and optimizer
    start_iteration = runner.current_learning_iteration
    before = {key: value.clone() for key, value in runner.alg.policy.state_dict().items()}
    ids = torch.arange(env.num_envs, device=env.device)
    low, high = env.policy_action_low, env.policy_action_high
    physical = env.policy_target_limits
    captured = {}
    original_compute = env.reward_manager.compute

    def capture(dt):
        value = original_compute(dt)
        captured.update(raw=env.raw_actions.clone(), bounded=env.bounded_actions.clone(),
                        executed=env.action.clone(), history=env.action_buffer._circular_buffer.buffer.clone(),
                        targets=env.robot.data.joint_pos_target[:, env.policy_joint_ids].clone(),
                        raw_excess=rewards.raw_action_target_limits(env).clone())
        return value

    env.reward_manager.compute = capture
    with torch.no_grad():
        env.reset(ids)
        # Cover all configured lags; training below restores the production lag (zero).
        lags = ids.remainder(cfg.domain_rand.action_delay.params['max_delay'] + 1).to(torch.int)
        env.action_buffer.set_time_lag(lags, ids)
        for step in range(12):
            raw = torch.full_like(env.raw_actions, (-1 if step % 2 else 1) * (10000. + step))
            original = raw.clone()
            obs, reward, done, extras = env.step(raw)
            torch.testing.assert_close(raw, original, rtol=0, atol=0)
            torch.testing.assert_close(captured['raw'], original, rtol=0, atol=0)
            torch.testing.assert_close(captured['bounded'], original.clamp(low, high))
            history = captured['history']
            assert torch.all(history >= low[:, None] - 1e-6)
            assert torch.all(history <= high[:, None] + 1e-6)
            expected = history[ids, -1 - lags]
            torch.testing.assert_close(captured['executed'], expected)
            assert torch.all(captured['targets'] >= physical[..., 0])
            assert torch.all(captured['targets'] <= physical[..., 1])
            assert torch.all(captured['raw_excess'] > 1000), 'Raw diagnostics must survive clipping'
            assert torch.isfinite(reward).all()
            actor, critic = env.compute_current_observations()
            current_history = env.action_buffer._circular_buffer.buffer[:, -1]
            torch.testing.assert_close(actor[:, 41:56], current_history)
            torch.testing.assert_close(critic[:, 41:56], current_history)
        # Reset only selected environments; unrelated diagnostic state must survive.
        keep = env.raw_actions[1::2].clone()
        env.reset(ids[::2])
        assert torch.count_nonzero(env.raw_actions[::2]) == 0
        assert torch.count_nonzero(env.bounded_actions[::2]) == 0
        torch.testing.assert_close(env.raw_actions[1::2], keep)
        env.action_buffer.set_time_lag(torch.zeros_like(lags), ids)
        env.reset(ids)
        obs, extras = env.get_observations()
        raw = runner.alg.act(obs, extras['observations']['critic'])
        sampled, log_prob = raw.clone(), runner.alg.transition.actions_log_prob.clone()
        obs, reward, done, extras = env.step(raw)
        torch.testing.assert_close(runner.alg.transition.actions, sampled, rtol=0, atol=0)
        torch.testing.assert_close(runner.alg.transition.actions_log_prob, log_prob, rtol=0, atol=0)
        runner.alg.process_env_step(reward, done, extras)
        torch.testing.assert_close(runner.alg.storage.actions[0], sampled, rtol=0, atol=0)
        runner.alg.storage.clear()
        env.reset(ids)
    # Collect actual extrema across all transitions without per-step CPU syncs.
    statistics = torch.zeros(5, device=env.device)
    def monitor(dt):
        reward = original_compute(dt)
        statistics[0] = torch.minimum(statistics[0], reward.min())
        statistics[1] = torch.maximum(statistics[1], env.raw_actions.abs().max())
        statistics[2] += (reward < -100).sum()
        targets = env.robot.data.joint_pos_target[:, env.policy_joint_ids]
        excess = torch.maximum(physical[..., 0] - targets, targets - physical[..., 1]).clamp(min=0)
        statistics[3] = torch.maximum(statistics[3], excess.max())
        statistics[4] += (~torch.isfinite(reward)).sum()
        return reward
    env.reward_manager.compute = monitor
    print('ACTION_BOUNDS_RUNTIME_PASS: raw PPO samples, all delay lags, history, PD targets, diagnostics, selective reset', flush=True)
    runner.learn(num_learning_iterations=args.train_iterations, init_at_random_ep_len=True)
    after = runner.alg.policy.state_dict()
    assert all(torch.isfinite(value).all() for value in after.values()), 'Nonfinite model state'
    assert any(not torch.equal(value, before[key]) for key, value in after.items()), 'No policy update'
    for state in runner.alg.optimizer.state.values():
        assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor)), 'Nonfinite optimizer'
    assert statistics[3] == 0, 'PD targets escaped physical bounds'
    assert statistics[4] == 0, 'Nonfinite reward'
    runner.writer.close()
    report = {'checkpoint': str(Path(args.checkpoint).resolve()), 'runtime_checks_passed': True,
              'start_iteration': start_iteration, 'end_iteration': runner.current_learning_iteration,
              'train_iterations': args.train_iterations, 'num_envs': env.num_envs,
              'training_transitions': args.train_iterations * runner.num_steps_per_env * env.num_envs,
              'model_and_optimizer_finite': True, 'policy_updated': True,
              'minimum_step_reward': float(statistics[0]), 'raw_action_abs_max': float(statistics[1]),
              'reward_below_minus100_count': int(statistics[2]), 'max_target_excess_rad': float(statistics[3])}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('ACTION_BOUNDS_CONTINUATION_PASS', json.dumps(report), flush=True)


from legged_lab.utils.app_lifecycle import run_simulation
run_simulation(main, launcher.app, args.headless)
