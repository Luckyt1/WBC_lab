"""Fresh-task integration checks in simulation, before launching the long run."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'rsl_rl')]
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import json
import torch
from legged_lab.envs.elf3_tang.scratch_cfg import Elf3ScratchBxiEnvCfg, Elf3ScratchBxiAgentCfg
from legged_lab.envs.elf3_tang.elf3_env import Elf3Env
from legged_lab.utils.task_registry import task_registry
from rsl_rl.runners import OnPolicyRunner


def main():
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg, agent = Elf3ScratchBxiEnvCfg(), Elf3ScratchBxiAgentCfg()
    assert not agent.resume and cfg.scene.num_envs == 4096
    legacy, _ = task_registry.get_cfgs('walk_elf3_tang_bxi_focus')
    assert legacy.commands.rel_standing_envs == .25 and legacy.arm_motion.amass_scale_end_iteration is None
    cfg.scene.num_envs = 128
    cfg.scene.seed = 42
    assert cfg.scratch_stage == 'locomotion' and not cfg.noise.add_noise
    assert cfg.reward.feet_air_time.params['use_standing_mask']
    assert cfg.reward.feet_contact_standing_stable.params['use_standing_mask']
    assert cfg.reward.track_body_yaw_exp.params['wrap_error']
    assert cfg.reward.raw_action_target_limits.weight < 0
    assert agent.algorithm.action_mean_bound_loss_coef == 1.
    # Allocate noise for a benchmark toggle; training itself remains noise-free.
    cfg.noise.add_noise = True
    env = Elf3Env(cfg, args.headless)
    env.add_noise = False
    ids = torch.arange(env.num_envs, device=env.device)
    obs, extra = env.get_observations()
    assert obs.shape == (128, 840) and extra['observations']['critic'].shape == (128, 910)
    assert (env.noise_scale_vec[56:84] > 0).all()
    before, _ = env.compute_current_observations()
    env.robot.data.joint_pos[:, env.arm_joint_ids[0]] += .01
    after, _ = env.compute_current_observations()
    torch.testing.assert_close(after[:, :56], before[:, :56])
    assert (after[:, 56] - before[:, 56]).abs().mean() > .005
    env.reset(ids)
    curriculum = []
    for iteration in (0, 4000, 8000, 16000, 20000, 100000):
        env.set_training_iteration(iteration)
        env.command_generator.reset(ids)
        scale = float(env._get_arm_curriculum_progress())
        assert abs(scale-.05) < 1e-6 and not (env.command_generator.sampled_mode >= 3).any()
        assert env._get_push_velocity_curriculum_progress() == 0.
        assert env._get_reset_velocity_curriculum_progress() == 0.
        assert env.command_generator.mode_fractions() == (.25, .25, 0., 0.)
        curriculum.append(dict(iteration=iteration, scale=scale,
                               mode_fractions=env.command_generator.mode_fractions(),
                               pose_ranges=env.command_generator.pose_ranges()))
    env.set_training_iteration(0)
    # The frozen benchmark must reach full strength even at iteration zero.
    cfg.push_velocity_curriculum.initial_scale = cfg.push_velocity_curriculum.final_scale = 1.
    env._update_push_velocity_curriculum()
    assert env.event_manager.get_term_cfg('push_robot').params['velocity_range']['x'] == (-.8, .8)
    env.event_manager.reset(ids)
    cfg.push_velocity_curriculum.initial_scale = cfg.push_velocity_curriculum.final_scale = 0.
    env._update_push_velocity_curriculum()
    env.reset(ids)
    # Cover mixed references and the shared guard path for explicit neutral targets.
    count = 0
    original = env.arm_guard.project
    def project(*a, **kw):
        nonlocal count
        count += 1
        return original(*a, **kw)
    env.arm_guard.project = project
    for fixed in (False, True):
        env.arm_targets = env.robot.data.default_joint_pos[:, env.arm_joint_ids].clone() if fixed else None
        for _ in range(30):
            obs, reward, _, extra = env.step(torch.zeros(128, 15, device=env.device))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert all(torch.isfinite(torch.as_tensor(x)).all() for x in extra['log'].values())
    assert count == 60
    env.arm_targets = None
    env.reset(ids)
    agent.logger = 'tensorboard'
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(out), device=env.device)
    assert runner.current_learning_iteration == 0 and len(runner.alg.optimizer.state) == 0
    runner.learn(3, init_at_random_ep_len=True)
    assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
    assert 1.e-5 <= runner.alg.optimizer.param_groups[0]['lr'] <= 3.e-4
    assert .1 <= float(runner.alg.policy.std.min()) and float(runner.alg.policy.std.max()) <= .60001
    report = dict(status='PASS', num_envs=128, actor_dim=840, critic_dim=910,
                  actions=15, fresh_optimizer_verified=True, teacher_used=False,
                  ppo_updates=3, guard_calls_checked=60, curriculum=curriculum,
                  neutral_reference_fraction=float(env.amass_neutral_mask.float().mean()),
                  learning_rate=runner.alg.learning_rate)
    report['repaired_rewards'] = {name: env.reward_manager.get_term_cfg(name).weight for name in
        ('raw_action_target_limits', 'waist_yaw_tracking_error', 'feet_air_time', 'feet_contact_standing_stable')}
    report['knee_diagnostics'] = {k: float(v) for k, v in env.extras['log'].items()
                                if 'knee' in k and 'clipped_fraction' in k}
    assert len(report['knee_diagnostics']) == 2
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    runner.writer.close()
    print('SCRATCH_VALIDATION', json.dumps(report), flush=True)


from legged_lab.utils.app_lifecycle import run_simulation
run_simulation(main, launcher.app, args.headless)
