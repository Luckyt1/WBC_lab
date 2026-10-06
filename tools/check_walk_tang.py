"""Small simulation/PPO validation of walk_elf3_tang; never launches a full training run."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=8)
parser.add_argument('--steps', type=int, default=80)
parser.add_argument('--ppo_iterations', type=int, default=3)
parser.add_argument('--output', type=str, default=str(ROOT / 'logs/tang_validation'))
parser.add_argument('--task', choices=('walk_elf3_tang', 'walk_elf3_tang_bxi'), default='walk_elf3_tang')
parser.add_argument('--wbc_reference', type=Path, help='Optional original WBC project for reward parity checks')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import json
import torch
from isaaclab.utils.io import dump_yaml
from legged_lab.envs.elf3_tang.elf3_env import Elf3Env
from legged_lab.envs.elf3_tang.walk_cfg import Elf3WalkTangFlatEnvCfg, Elf3WalkTangAgentCfg
from legged_lab.envs.elf3_tang import rewards as tang_rewards
from rsl_rl.runners import OnPolicyRunner


def main():
    from legged_lab.utils.task_registry import task_registry
    cfg, agent = task_registry.get_cfgs(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.scene.seed = 42
    cfg.device = args.device
    cfg.domain_rand.events.push_robot = None
    env = Elf3Env(cfg, args.headless)
    env.set_training_iteration(cfg.commands.pose_end_iteration)
    env.command_generator.set_manual_command([0.3, 0, 0.2, 0.5, 0.6])
    obs, extra = env.get_observations()
    assert obs.shape == (args.num_envs, 560), obs.shape
    assert extra['observations']['critic'].shape == (args.num_envs, 630)
    assert env.num_actions == 15
    assert env.robot.data.body_names[env.pelvis_body_id] == 'waist_z_link'
    assert env.robot.data.body_names[env.torso_body_id] == 'torso_link'
    assert [env.robot.data.body_names[i] for i in env.knee_body_ids] == ['l_knee_y_link', 'r_knee_y_link']
    limits = env.robot.data.joint_effort_limits[:, env.policy_joint_ids]
    assert torch.isfinite(limits).all() and torch.all(limits > 0)
    if args.task == 'walk_elf3_tang_bxi':
        from check_elf3_bxi_runtime import check_runtime
        check_runtime(env)
    required_rewards = {'joint_torque_limits_waists', 'joint_torque_limits_ankles',
                        'joint_torque_limits_hip_pitch', 'feet_contact_standing_stable',
                        'feet_flat_standing', 'center_of_gravity_tracking', 'knee_stance_width',
                        'track_body_height_exp', 'track_body_yaw_exp', 'feet_air_time'}
    assert required_rewards.issubset(env.reward_manager.active_terms)
    nonzero_rewards = [name for name in env.reward_manager.active_terms
                       if env.reward_manager.get_term_cfg(name).weight != 0]
    assert len(nonzero_rewards) == 31
    assert env.use_amass and env.cfg.arm_motion.scale == 1.0
    assert env.amass_motion.ramp_s == 0 and env.amass_motion.max_delta is None
    assert env.command_generator.full_command.shape[-1] == 5
    assert (env.command_generator.wbc_command[:, 4:6] == 0).all()
    assert torch.all(env.noise_scale_vec[6:11] == 0)
    assert torch.all(env.noise_scale_vec[11:26] > 0)
    max_arm_delta = 0.
    reset_count = 0
    for _ in range(args.steps):
        arms = env.arm_position_target.clone()
        from legged_lab.envs.elf3_tang.wbc.elf3_arm_motion import map_g1_arms_to_elf3
        frames = env.amass_index % cfg.arm_motion.sequence_length
        poses = env.amass[torch.arange(env.num_envs, device=env.device), frames]
        expected_arms = map_g1_arms_to_elf3(poses, env.amass_motion.joint_names)
        arm_limits = env.robot.data.soft_joint_pos_limits[:, env.arm_joint_ids]
        expected_arms = expected_arms.clamp(arm_limits[..., 0], arm_limits[..., 1])
        obs, reward, done, extra = env.step(torch.zeros(args.num_envs, 15, device=env.device))
        live = ~done
        if live.any():
            delta = (env.arm_position_target[live] - arms[live]).abs().max().item()
            max_arm_delta = max(max_arm_delta, delta)
            torch.testing.assert_close(env.arm_position_target[live], expected_arms[live])
        reset_count += done.sum().item()
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        assert torch.isfinite(extra['observations']['critic']).all()
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in extra['log'].values())
        for name in required_rewards:
            term = env.reward_manager.get_term_cfg(name)
            value = term.func(env, **term.params)
            assert value.shape == (env.num_envs,) and torch.isfinite(value).all(), name
        if args.task == 'walk_elf3_tang_bxi':
            head_ids, _ = env.robot.find_joints(['head_z_joint', 'head_y_joint'], preserve_order=True)
            assert (env.robot.data.joint_pos_target[:, head_ids] == 0).all()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    parity = None
    if args.wbc_reference:
        from check_wbc_reward_parity import check_reward_parity
        parity = check_reward_parity(env, args.wbc_reference)
    env.command_generator.manual_command = None
    env.set_training_iteration(0)
    env.reset(torch.arange(env.num_envs, device=env.device))
    agent.device = env.device
    agent.logger = 'tensorboard'
    agent.save_interval = 1
    dump_yaml(str(output / 'params/env.yaml'), cfg)
    dump_yaml(str(output / 'params/agent.yaml'), agent)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(output), device=env.device)
    if args.ppo_iterations:
        runner.learn(args.ppo_iterations, init_at_random_ep_len=True)
        assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
        for state in runner.alg.optimizer.state.values():
            assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor))
    report = dict(status='PASS',num_envs=env.num_envs,sim_steps=args.steps,
                  actor_dim=560,critic_dim=630,actions=15,physical_joints=env.robot.num_joints,
                  zero_policy_resets=reset_count,max_arm_delta_per_control_step=max_arm_delta,
                  ppo_iterations=args.ppo_iterations,source=str(ROOT),task=args.task,
                  active_rewards=env.reward_manager.active_terms,
                  nonzero_rewards=nonzero_rewards,
                  arm_source=cfg.arm_motion.source,arm_scale=cfg.arm_motion.scale,
                  amass_sequences=len(env.amass_loader.sequences),reward_parity=parity,
                  peak_torch_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2)
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if runner.writer is not None:
        runner.writer.close()
    print('TANG_VALIDATION', json.dumps(report), flush=True)


from legged_lab.utils.app_lifecycle import run_simulation

run_simulation(main, app, args.headless)
