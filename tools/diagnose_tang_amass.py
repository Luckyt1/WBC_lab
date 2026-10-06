"""Paired frozen-policy AMASS ablations, including pre-reset contact diagnostics."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--reference-report', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--seed', type=int, default=20260921)
parser.add_argument('--repeats', type=int, default=32)
parser.add_argument('--profiles', nargs='+', choices=('original', 'ramp_1s', 'ramp_1s_rate_2', 'fixed_arms', 'guarded_neutral'))
parser.add_argument('--record-pairs', action='store_true')
parser.add_argument('--arm-guard', action='store_true')
parser.add_argument('--disable-self-collisions', action='store_true',
                    help='Diagnostic counterfactual only; never a proposed acceptance configuration.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import hashlib
import json
from types import MethodType
import torch
from legged_lab.envs.elf3_tang.elf3_env import Elf3Env
from legged_lab.envs.elf3_tang.stand_cfg import Elf3StandWalkBxiEnvCfg, Elf3StandWalkBxiAgentCfg
from legged_lab.envs.elf3_tang.commands import yaw_frame_velocity, rotate_inverse
from legged_lab.envs.elf3_tang.rewards import pelvis_height
from legged_lab.envs.elf3_tang.wbc.arm_guard import arm_frames, inverse, rotate
from rsl_rl.runners import OnPolicyRunner


@torch.inference_mode()
def main():
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads(Path(args.reference_report).read_text())
    scenarios = reference['profiles'][0]['scenarios']
    cfg, agent = Elf3StandWalkBxiEnvCfg(), Elf3StandWalkBxiAgentCfg()
    cfg.scene.num_envs = args.repeats * len(scenarios)
    cfg.scene.seed = args.seed
    cfg.scene.max_episode_length_s = 21.
    cfg.noise.add_noise = False
    cfg.domain_rand.events.push_robot = None
    cfg.reset_velocity_curriculum.initial_scale = cfg.reset_velocity_curriculum.final_scale = 0.
    cfg.arm_motion.collision_guard = args.arm_guard
    if args.disable_self_collisions:
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    env = Elf3Env(cfg, args.headless)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=env.device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy()
    ids = torch.arange(env.num_envs, device=env.device)
    targets = torch.tensor([s['target'] for s in scenarios], device=env.device).repeat_interleave(args.repeats, 0)

    def resample_fixed(generator, env_ids):
        generator.target[env_ids] = targets[env_ids]
        generator.full_command[env_ids, :3] = targets[env_ids, :3]
        generator.is_standing_env[env_ids] = targets[env_ids, :3].abs().sum(-1) < 1e-6
        generator.time_left[env_ids] = 1e9

    env.command_generator._resample = MethodType(resample_fixed, env.command_generator)
    captured = {}
    previous = {}
    original_check = env.check_reset
    term_ids = env.termination_contact_cfg.body_ids
    term_names = [env.contact_sensor.body_names[i] for i in term_ids]
    pair_names = [f'{s}_{part}_link' for s in ('l', 'r') for part in ('shoulder_z', 'wrist_x', 'wrist_y', 'wrist_z')]
    pair_views = []
    if args.record_pairs:
        view = env.contact_sensor._physics_sim_view
        pair_views = [view.create_rigid_contact_view(f'/World/envs/env_*/Robot/{name}',
                      filter_patterns=[f'/World/envs/env_*/Robot/{other}' for other in pair_names])
                      for name in term_names]
        assert all(v.sensor_count == env.num_envs and v.filter_count == len(pair_names) for v in pair_views)
        original_step = env.sim.step

        def physics_step(*a, **kw):
            result = original_step(*a, **kw)
            forces = torch.stack([v.get_contact_force_matrix(env.physics_dt).norm(dim=-1) for v in pair_views], 1)
            previous['pairs'] = torch.maximum(previous.get('pairs', torch.zeros_like(forces)), forces)
            return result

        env.sim.step = physics_step

    def capture():
        done, timeout = original_check()
        data = env.robot.data
        quat = data.body_quat_w[:, env.pelvis_body_id]
        captured.update(
            failed=done & ~timeout,
            actual=torch.cat((yaw_frame_velocity(quat, data.body_lin_vel_w[:, env.pelvis_body_id]),
                              data.body_ang_vel_w[:, env.pelvis_body_id, 2:3], pelvis_height(env)[:, None]), -1),
            up_z=-rotate_inverse(quat, data.GRAVITY_VEC_W)[:, 2],
            forces=env.contact_sensor.data.net_forces_w_history[:, :, term_ids].norm(dim=-1).amax(1),
            arm_target_speed=(env.arm_position_target - previous['target']).abs().amax(-1) / env.step_dt,
            source_sequence=env.amass_source_sequence.clone(),
            source_start=env.amass_source_start.clone(),
            source_offset=(env.amass_index - 1) % cfg.arm_motion.sequence_length,
        )
        return done, timeout

    env.check_reset = capture
    profiles = [('original', 0., None, False), ('ramp_1s', 1., None, False),
                ('ramp_1s_rate_2', 1., 2., False), ('fixed_arms', 0., None, True)]
    if args.profiles and 'guarded_neutral' in args.profiles:
        if not args.arm_guard:
            raise ValueError('guarded_neutral requires --arm-guard')
        profiles.append(('guarded_neutral', 0., None, False))
    if args.profiles:
        profiles = [p for p in profiles if p[0] in args.profiles]
    report = {'checkpoint': str(Path(args.checkpoint).resolve()), 'seed': args.seed,
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'num_envs': env.num_envs, 'repeats': args.repeats, 'warmup_s': 4., 'measure_s': 12.,
              'self_collisions_enabled': not args.disable_self_collisions,
              'arm_guard_enabled': args.arm_guard,
              'method': 'Same initial arm sequences, reset seed and physics; no observation noise/push/reset velocity; scale=1; masked at first failure. Smoothing is a changed arm tracking task, not proof the original task is solved.',
              'termination_contact_names': term_names, 'profiles': []}
    baseline_hash = None
    original_motion_step = env.amass_motion.step
    for name, ramp, rate, fixed in profiles:
        print('DIAG_PROFILE', name, flush=True)
        env.amass_motion.ramp_s = ramp
        env.amass_motion.max_delta = None if rate is None else rate * env.step_dt
        env.amass_motion.step = original_motion_step
        env.arm_targets = None
        env.seed(args.seed)
        env.reset(ids)
        fk_pos, _ = arm_frames(env.robot.data.joint_pos[:, env.arm_joint_ids].reshape(-1, 2, 7))
        fk_body_ids = [env.robot.body_names.index(n) for n in pair_names]
        observed = rotate(inverse(env.robot.data.body_quat_w[:, env.torso_body_id])[:, None, :],
                          env.robot.data.body_pos_w[:, fk_body_ids]
                          - env.robot.data.body_pos_w[:, env.torso_body_id, None, :])
        fk_error = (observed - fk_pos.reshape(-1, 8, 3)).norm(dim=-1).max().item()
        assert fk_error < .001, f'Arm forward kinematics disagrees with physics: {fk_error} m'
        clip_hash = hashlib.sha256(env.amass.cpu().numpy().tobytes()).hexdigest()
        if baseline_hash is None:
            baseline_hash = clip_hash
        assert clip_hash == baseline_hash, 'AMASS clips differ between paired profiles'
        if fixed:
            env.arm_targets = env.robot.data.default_joint_pos[:, env.arm_joint_ids].clone()
        if name == 'guarded_neutral':
            # A neutral reference with dynamic collision correction is a separate
            # task from strict fixed arms. Keep both tests; never relabel them.
            def neutral_step(_poses):
                env.amass_motion.target.copy_(env.robot.data.default_joint_pos[:, env.arm_joint_ids])
                return env.amass_motion.target
            env.amass_motion.step = neutral_step
        obs, _ = env.get_observations()
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        count = torch.zeros(env.num_envs, device=env.device)
        actual_sum = torch.zeros(env.num_envs, 4, device=env.device)
        max_target_speed = torch.zeros(env.num_envs, device=env.device)
        failures = []
        guard_metrics = {}
        for step in range(800):
            previous['target'] = env.arm_position_target.clone()
            previous.pop('pairs', None)
            obs, _, _, _ = env.step(policy(obs))
            assert torch.isfinite(obs).all()
            max_target_speed = torch.maximum(max_target_speed, captured['arm_target_speed'] * alive)
            if step >= 200:
                count += alive
                actual_sum += captured['actual'] * alive[:, None]
            failed = alive & captured['failed']
            for i in failed.nonzero(as_tuple=False).flatten().tolist():
                force = captured['forces'][i].cpu().tolist()
                failures.append({'env_id': i, 'scenario': scenarios[i // args.repeats]['scenario'],
                                 'time_s': (step + 1) * env.step_dt,
                                 'height_m': captured['actual'][i, 3].item(), 'up_z': captured['up_z'][i].item(),
                                 'contacts_over_50N': {n: f for n, f in zip(term_names, force) if f > 50},
                                 'max_arm_target_speed_rad_s': max_target_speed[i].item()})
                if not fixed and name != 'guarded_neutral':
                    source = env.amass_loader.sequences[int(captured['source_sequence'][i])]
                    start_frame = int(captured['source_start'][i])
                    source_frame = min(start_frame + int(captured['source_offset'][i]), source['length'] - 1)
                    failures[-1].update(amass_path=str(source['path']), source_start_frame=start_frame,
                                        source_frame=source_frame, source_fps=source['fps'])
                if pair_views:
                    pairs = previous['pairs'][i].cpu().tolist()
                    failures[-1]['contact_pairs_over_1N'] = {
                        name + ' / ' + other: pairs[j][k] for j, name in enumerate(term_names)
                        for k, other in enumerate(pair_names) if pairs[j][k] > 1.}
            alive &= ~captured['failed']
            if args.arm_guard and not fixed:
                for key, value in env.arm_guard.metrics.items():
                    guard_metrics[key] = guard_metrics.get(key, 0.) + float(value) / 800
        rows = []
        for i, scenario in enumerate(scenarios):
            group = slice(i * args.repeats, (i + 1) * args.repeats)
            rows.append({'scenario': scenario['scenario'], 'target': scenario['target'],
                         'survived': int(alive[group].sum()), 'trials': args.repeats,
                         'actual_mean': (actual_sum[group].sum(0) / count[group].sum().clamp(min=1)).tolist()})
        row = {'name': name, 'ramp_s': ramp, 'max_velocity': rate, 'scale': 1.,
               'arm_reference_mode': ('strict_fixed' if fixed else
                                      'neutral_with_dynamic_guard' if name == 'guarded_neutral' else 'amass'),
               'initial_arm_fk_max_error_m': fk_error,
               'initial_clip_sha256': clip_hash, 'survived': int(alive.sum()),
               'max_arm_target_speed_rad_s': max_target_speed.max().item(),
               'median_max_arm_target_speed_rad_s': max_target_speed.median().item(),
               'failures': failures, 'scenarios': rows}
        row['guard_metrics_all_rollout_environments'] = guard_metrics
        report['profiles'].append(row)
        (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print('DIAG_RESULT', name, row['survived'], '/', env.num_envs, flush=True)


from legged_lab.utils.app_lifecycle import run_simulation
run_simulation(main, launcher.app, args.headless)
