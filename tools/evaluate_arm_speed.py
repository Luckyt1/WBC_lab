"""Explicit, post-training Isaac evaluation of ONE frozen ELF3 checkpoint.

No training, automatic checkpoint selection, deployment, or background watcher.
Legacy playback is retained as a separate baseline. Speed factors refer to source
FPS, not the legacy one-source-frame-per-policy-step convention.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'rsl_rl')]
from arm_speed_metrics import resample_source, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speeds', nargs='+', type=float, default=[.5, .75, 1.])
    parser.add_argument('--arm-scale', type=float, default=1.)
    parser.add_argument('--repeats', type=int, default=32)
    parser.add_argument('--measure-s', type=float, default=60.)
    parser.add_argument('--warmup-s', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--cases', nargs='+', choices=('stand', 'walk_then_stop'),
                        default=['stand', 'walk_then_stop'])
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    import math
    if (args.repeats < 1 or not 0 <= args.arm_scale <= 1
            or not all(math.isfinite(x) and x > 0 for x in args.speeds)
            or not math.isfinite(args.measure_s) or args.measure_s <= 0
            or not math.isfinite(args.warmup_s) or args.warmup_s < .04):
        parser.error('invalid duration, speed, scale, or repeat count')
    if not args.checkpoint.is_file():
        parser.error('checkpoint does not exist')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new or empty directory')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    launcher = AppLauncher(args)

    def evaluate():
        import numpy as np
        import torch
        from isaaclab.utils.io import dump_yaml
        from legged_lab.envs.elf3_tang.elf3_env import Elf3Env
        from legged_lab.envs.elf3_tang.walk_robust_cfg import Elf3WalkRobustBxiEnvCfg, Elf3WalkRobustBxiAgentCfg
        from legged_lab.envs.elf3_tang.commands import quat_yaw
        from legged_lab.envs.elf3_tang.rewards import pelvis_height
        from rsl_rl.runners import OnPolicyRunner

        cfg, agent = Elf3WalkRobustBxiEnvCfg(), Elf3WalkRobustBxiAgentCfg()
        cfg.scene.num_envs, cfg.scene.seed = args.repeats, args.seed
        cfg.device = agent.device = args.device
        cfg.scene.max_episode_length_s = args.measure_s + args.warmup_s + 8. + 5.
        cfg.noise.add_noise = False
        cfg.domain_rand.events.push_robot = None
        cfg.reset_velocity_curriculum.initial_scale = cfg.reset_velocity_curriculum.final_scale = 0.
        cfg.arm_motion.amass_scale_end_iteration = None
        cfg.arm_motion.scale = args.arm_scale
        cfg.arm_motion.amass_neutral_fraction = 0.
        cfg.arm_motion.guard_fixed_targets = True
        env = Elf3Env(cfg, args.headless)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=env.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        ids = torch.arange(env.num_envs, device=env.device)
        height = cfg.commands.nominal_height
        legacy_length = cfg.arm_motion.sequence_length
        capture = {}
        check_reset = env.check_reset

        def capture_before_reset():
            done, timeout = check_reset()
            data = env.robot.data
            capture.update(pelvis_xy=data.body_pos_w[:, env.pelvis_body_id, :2].clone(),
                           pelvis_yaw=quat_yaw(data.body_quat_w[:, env.pelvis_body_id]).clone(),
                           height=pelvis_height(env).clone(),
                           horizontal_speed=data.body_lin_vel_w[:, env.pelvis_body_id, :2].norm(dim=-1),
                           terminated=(done & ~timeout).clone(), timeout=timeout.clone())
            return done, timeout

        env.check_reset = capture_before_reset
        schedule = dict(step=0, stop_step=None)
        compute_commands = env.command_generator.compute

        def scheduled_commands(dt):
            compute_commands(dt)
            schedule['step'] += 1
            if schedule['step'] == schedule['stop_step']:
                # Change commands at the transition boundary before constructing
                # the NEXT observation; never append a duplicate history frame.
                env.command_generator.set_manual_command([0., 0., 0., height, 0.])

        env.command_generator.compute = scheduled_commands
        report = dict(status='running', checkpoint=str(args.checkpoint.resolve()),
                      checkpoint_sha256=checkpoint_hash, iteration=runner.current_learning_iteration,
                      seed=args.seed, repeats=args.repeats, policy_dt=env.step_dt,
                      physics_dt=env.physics_dt, arm_scale=args.arm_scale,
                      moving_threshold_rad_s=env.arm_diagnostics.moving_threshold,
                      measure_s=args.measure_s, warmup_s=args.warmup_s,
                      source_fps_origin='loader filename convention; validate against dataset metadata',
                      method='Frozen deterministic actor, no noise/push/reset velocity; startup domain randomization retained. Paired initial states and source clips. Full first trial including warmup/terminal transition; post-failure samples excluded. Native clips hold at end; legacy baseline repeats original padded bank.',
                      auto_deploy=False, training_updates=0, profiles=[])
        dump_yaml(str(args.output/'env.yaml'), cfg)
        bank = None
        initial_hash = None
        sources = None
        source_data = None
        profiles = [('legacy', None), ('neutral', None)] + [
            (f'source_fps_{i}_{speed:g}x', speed) for i, speed in enumerate(args.speeds)]
        for case in args.cases:
            stop_step = round(8./env.step_dt) if case == 'walk_then_stop' else 0
            warmup_steps = stop_step + round(args.warmup_s/env.step_dt)
            steps = warmup_steps + round(args.measure_s/env.step_dt)
            for name, speed in profiles:
                print('ARM_SPEED_EVALUATING', case, name, flush=True)
                env.seed(args.seed)
                cfg.arm_motion.amass_neutral_fraction = 0.
                cfg.arm_motion.sequence_length = legacy_length
                env.amass = torch.empty(env.num_envs, legacy_length, 14, device=env.device)
                env.arm_targets = None
                env.command_generator.set_manual_command([.3 if stop_step else 0., 0., 0., height, 0.])
                env.reset(ids)
                schedule.update(step=0, stop_step=stop_step or None)
                initial = torch.cat((env.robot.data.root_state_w, env.robot.data.joint_pos,
                                     env.robot.data.joint_vel), -1).cpu().numpy()
                current_hash = hashlib.sha256(initial.tobytes()).hexdigest()
                if initial_hash is None:
                    initial_hash = current_hash
                if current_hash != initial_hash:
                    raise RuntimeError('Paired reset states differ; refusing an unpaired comparison')
                if bank is None:
                    bank = env.amass.cpu().numpy().copy()
                    bank_valid = env.amass_valid_length.cpu().numpy().copy()
                    sources, source_data = [], []
                    for seq_id, start in zip(env.amass_source_sequence.tolist(), env.amass_source_start.tolist()):
                        meta = env.amass_loader.sequences[seq_id]
                        path = meta['path']
                        raw = np.load(path)
                        source_data.append(raw[:, 22:36])
                        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                            sequence_id=seq_id, start_frame=start, fps=meta['fps'], frames=len(raw)))
                    np.savez_compressed(args.output/'paired_legacy_bank.npz', arm=bank, valid_length=bank_valid)
                    report.update(sources=sources, initial_state_sha256=initial_hash,
                                  legacy_bank_sha256=hashlib.sha256(bank.tobytes()).hexdigest())
                if speed is None:
                    index = np.arange(steps) % legacy_length
                    trajectory = bank[:, index].copy()
                    tails = index[None, :] >= bank_valid[:, None]
                else:
                    sampled = [resample_source(raw, meta['start_frame'], meta['fps'], env.step_dt, speed, steps)
                               for raw, meta in zip(source_data, sources)]
                    trajectory = np.stack([x[0] for x in sampled])
                    tails = np.stack([x[1] for x in sampled])
                cfg.arm_motion.sequence_length = steps
                env.amass = torch.as_tensor(trajectory, device=env.device)
                # The exact per-step tail mask below also handles legacy wraparound.
                env.amass_valid_length[:] = steps
                env.amass_neutral_mask.fill_(name == 'neutral')
                cfg.arm_motion.amass_neutral_fraction = float(name == 'neutral')
                if name == 'neutral':
                    tails[:] = False
                obs, _ = env.get_observations()
                alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                history = {}
                for step in range(steps):
                    valid = alive.clone()
                    obs, _, _, info = env.step(policy(obs))
                    if not torch.isfinite(obs).all() or capture['timeout'].any():
                        raise RuntimeError('Nonfinite observation or unexpected timeout')
                    sample = dict(info['arm_motion'])
                    sample.update(capture)
                    sample['valid'] = valid
                    sample['padded_tail'] = torch.as_tensor(tails[:, step], device=env.device)
                    for key, value in sample.items():
                        history.setdefault(key, []).append(value.cpu().numpy().copy())
                    alive &= ~capture['terminated']
                    if (step+1) % 500 == 0:
                        print('ARM_SPEED_PROGRESS', case, name, step+1, steps, int(alive.sum()), flush=True)
                arrays = {key: np.stack(values) for key, values in history.items()}
                filename = f'{case}_{name}.npz'
                np.savez_compressed(args.output/filename, **arrays)
                metrics = summarize(arrays, warmup_steps, env.step_dt, env.arm_diagnostics.moving_threshold)
                row = dict(case=case, profile=name, source_speed=speed, measurement_start_s=warmup_steps*env.step_dt,
                           initial_state_sha256=current_hash, trace=filename, **metrics)
                report['profiles'].append(row)
                print('ARM_SPEED_RESULT', json.dumps(row, allow_nan=False), flush=True)
                (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
            raise RuntimeError('Checkpoint changed during evaluation')
        report['status'] = 'complete'
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    import torch
    from legged_lab.utils.app_lifecycle import run_simulation
    run_simulation(torch.inference_mode()(evaluate), launcher.app, args.headless)


if __name__ == '__main__':
    main()
