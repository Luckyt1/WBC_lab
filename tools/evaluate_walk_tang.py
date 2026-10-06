"""Evaluate a frozen Tang checkpoint across fixed commands, without training updates."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', choices=('walk_elf3_tang', 'walk_elf3_tang_bxi', 'walk_elf3_tang_bxi_scratch', 'walk_elf3_tang_bxi_robust', 'walk_elf3_tang_bxi_still'),
                    default='walk_elf3_tang', help='Select the robot asset used during training')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--repeats', type=int, default=32)
parser.add_argument('--seed', type=int, default=20260921)
parser.add_argument('--arm-guard', action='store_true')
parser.add_argument('--include-guarded-neutral', action='store_true')
parser.add_argument('--include-mixed-yaw', action='store_true')
parser.add_argument('--gait-diagnostics', action='store_true')
parser.add_argument('--auxiliary-probe', action='store_true',
                    help='Save a subsampled held-out dataset of actor histories and aligned truth transitions.')
parser.add_argument('--warmup_s', type=float, default=4.0)
parser.add_argument('--measure_s', type=float, default=12.0)
parser.add_argument('--curriculum-stage', default=None,
                    help='Append a fixed stage-difficulty profile; full benchmarks remain unchanged.')
parser.add_argument('--teleop-stage', default=None,
                    help='Evaluate current/easier teleop stages only, including 60s position retention.')
parser.add_argument('--profile', choices=('wbc', 'legacy'), default='wbc',
                    help='Use legacy for checkpoints trained before the WBC/AMASS migration')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.teleop_stage:
    from tools.teleop_curriculum import STAGES
    if (args.teleop_stage not in STAGES or args.task != 'walk_elf3_tang_bxi_robust'
            or args.curriculum_stage or args.include_mixed_yaw or args.include_guarded_neutral
            or args.warmup_s != 4. or args.measure_s != 56. or args.repeats != 32):
        parser.error('Teleop gate needs robust task, 32 trials, 4s warmup + 56s measurement, no extra profiles')
    args.gait_diagnostics = args.arm_guard = True
launcher = AppLauncher(args)
app = launcher.app

import hashlib
import json
import math
from types import MethodType

import numpy as np
import torch
from isaaclab.utils.io import dump_yaml
from legged_lab.envs.elf3_tang.elf3_env import Elf3Env
from legged_lab.envs.elf3_tang.walk_cfg import Elf3WalkTangFlatEnvCfg, Elf3WalkTangAgentCfg
from legged_lab.envs.elf3_tang.commands import rotate_inverse, wrap_to_pi, yaw_frame_velocity
from legged_lab.envs.elf3_tang import rewards
from rsl_rl.runners import OnPolicyRunner


SCENARIOS = [
    ('stand', [0, 0, 0, .8235, 0]),
    ('squat_065', [0, 0, 0, .65, 0]),
    ('squat_050', [0, 0, 0, .50, 0]),
    ('waist_yaw_pos', [0, 0, 0, .8235, .6]),
    ('waist_yaw_neg', [0, 0, 0, .8235, -.6]),
    ('forward_030', [.3, 0, 0, .8235, 0]),
    ('forward_060', [.6, 0, 0, .8235, 0]),
    ('backward_030', [-.3, 0, 0, .8235, 0]),
    ('lateral_pos', [0, .2, 0, .8235, 0]),
    ('lateral_neg', [0, -.2, 0, .8235, 0]),
    ('turn_pos', [0, 0, .5, .8235, 0]),
    ('turn_neg', [0, 0, -.5, .8235, 0]),
    ('crouch_walk_065', [.3, 0, 0, .65, 0]),
    ('crouch_walk_050', [.3, 0, 0, .50, 0]),
    ('walk_waist_yaw', [.3, 0, 0, .8235, .6]),
    ('crouch_walk_yaw_pos', [.3, 0, 0, .50, .6]),
    ('crouch_walk_yaw_neg', [.3, 0, 0, .50, -.6]),
]

if args.task in ('walk_elf3_tang_bxi_robust', 'walk_elf3_tang_bxi_still'):
    from legged_lab.envs.elf3_tang.walk_robust_curriculum import SCENARIOS
if args.teleop_stage:
    SCENARIOS = list(SCENARIOS) + [('walk_then_stop', [0., 0., 0., .8235, 0.])]

if args.include_mixed_yaw:
    SCENARIOS = list(SCENARIOS) + [
        ('arc_forward_pos', [.3, 0, .1, .8235, 0]),
        ('arc_forward_neg', [.3, 0, -.1, .8235, 0]),
        ('arc_backward_pos', [-.3, 0, .1, .8235, 0]),
        ('arc_backward_neg', [-.3, 0, -.1, .8235, 0]),
    ]


@torch.inference_mode()
def main():
    global SCENARIOS
    assert args.repeats > 0 and args.warmup_s >= 3 and args.measure_s > 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.task in ('walk_elf3_tang_bxi_robust', 'walk_elf3_tang_bxi_still'):
        from legged_lab.envs.elf3_tang.walk_robust_cfg import Elf3WalkRobustBxiEnvCfg, Elf3WalkRobustBxiAgentCfg
        cfg, agent = Elf3WalkRobustBxiEnvCfg(), Elf3WalkRobustBxiAgentCfg()
        if args.task == 'walk_elf3_tang_bxi_still':
            from legged_lab.envs.elf3_tang.standing_refine_cfg import Elf3StandingRefineEnvCfg, Elf3StandingRefineAgentCfg
            cfg, agent = Elf3StandingRefineEnvCfg(), Elf3StandingRefineAgentCfg()
    elif args.task == 'walk_elf3_tang_bxi_scratch':
        from legged_lab.envs.elf3_tang.scratch_cfg import Elf3ScratchBxiEnvCfg, Elf3ScratchBxiAgentCfg
        cfg, agent = Elf3ScratchBxiEnvCfg(), Elf3ScratchBxiAgentCfg()
    elif args.task == 'walk_elf3_tang_bxi':
        from legged_lab.envs.elf3_tang.walk_bxi_cfg import Elf3WalkTangBxiEnvCfg, Elf3WalkTangBxiAgentCfg
        cfg, agent = Elf3WalkTangBxiEnvCfg(), Elf3WalkTangBxiAgentCfg()
    else:
        cfg, agent = Elf3WalkTangFlatEnvCfg(), Elf3WalkTangAgentCfg()
    if args.profile == 'legacy':
        from legged_lab.envs.elf3_tang.walk_cfg import LiteRewardCfg
        cfg.reward = LiteRewardCfg()
        cfg.arm_motion.source = 'random'
    fixed_height = None
    if args.task in ('walk_elf3_tang_bxi_robust', 'walk_elf3_tang_bxi_still'):
        from tools.fixed_pelvis_height import checkpoint_height, configure_fixed_height, height_scenarios
        fixed_height = checkpoint_height(args.checkpoint)
        configure_fixed_height(cfg, fixed_height)
        SCENARIOS = height_scenarios(SCENARIOS, fixed_height)
        from legged_lab.envs.elf3_tang.liftoff_rewards import checkpoint_recipe, configure_liftoff_rewards
        if checkpoint_recipe(args.checkpoint) is not None:
            configure_liftoff_rewards(cfg)
    if args.curriculum_stage and args.task != 'walk_elf3_tang_bxi_scratch':
        raise ValueError('--curriculum-stage requires the scratch task')
    if args.arm_guard:
        cfg.arm_motion.collision_guard = True
        cfg.arm_motion.ramp_s = 1.0
        cfg.arm_motion.amass_max_velocity = None
    # A fixed benchmark: all AMASS trials use full amplitude, independent of training age.
    cfg.arm_motion.amass_neutral_fraction = 0.
    cfg.arm_motion.amass_scale_end_iteration = None
    cfg.arm_motion.scale = 1.
    cfg.scene.num_envs = len(SCENARIOS) * args.repeats
    cfg.scene.seed = args.seed
    cfg.scene.max_episode_length_s = args.warmup_s + args.measure_s + 5
    cfg.device = agent.device = args.device
    if args.auxiliary_probe:
        cfg.auxiliary_supervision = True
    cfg.noise.add_noise = True  # Allocate noise even when the training stage disables it.
    # Build the noise vector and interval event normally, then switch them per profile.
    env = Elf3Env(cfg, args.headless)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=env.device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
    targets = torch.tensor([c for _, c in SCENARIOS], device=env.device).repeat_interleave(args.repeats, 0)
    ids = torch.arange(env.num_envs, device=env.device)

    def resample_fixed(generator, env_ids):
        generator.target[env_ids] = targets[env_ids]
        generator.full_command[env_ids, :3] = targets[env_ids, :3]
        generator.is_standing_env[env_ids] = targets[env_ids, :3].abs().sum(-1) < 1e-6
        generator.time_left[env_ids] = 1e9

    env.command_generator._resample = MethodType(resample_fixed, env.command_generator)
    original_apply = env.event_manager.apply
    profile_state = {'push': False}

    def apply_events(*positional, **keywords):
        mode = keywords.get('mode', positional[0] if positional else None)
        if mode == 'interval' and not profile_state['push']:
            return
        return original_apply(*positional, **keywords)

    env.event_manager.apply = apply_events
    original_reset_check = env.check_reset
    captured = {}
    knee_columns = [i for i, joint_id in enumerate(env.policy_joint_ids)
                    if 'knee' in env.robot.data.joint_names[joint_id]]
    knee_names = [env.robot.data.joint_names[env.policy_joint_ids[i]] for i in knee_columns]
    if args.gait_diagnostics:
        from legged_lab.envs.elf3_tang.commands import quat_yaw
        from legged_lab.envs.elf3_tang.swing_quality import sole_clearance
        if cfg.scene.terrain_generator is not None:
            raise ValueError('Sole-height diagnostics require a flat plane')
        foot_names = ['l_ankle_x_link', 'r_ankle_x_link']
        foot_ids, _ = env.robot.find_bodies(foot_names, preserve_order=True)
        hulls = json.loads((ROOT/'legged_lab/assets/elf3_bxi_31/foot_collision_hulls.json').read_text())['feet']
        vertices = torch.tensor([hulls[name]['vertices'] for name in foot_names], device=env.device)

    def capture_before_reset():
        done, timeout = original_reset_check()
        data = env.robot.data
        quat = data.body_quat_w[:, env.pelvis_body_id]
        xy = yaw_frame_velocity(quat, data.body_lin_vel_w[:, env.pelvis_body_id])
        actual = torch.cat((xy, data.body_ang_vel_w[:, env.pelvis_body_id, 2:3],
                            rewards.pelvis_height(env)[:, None], rewards.waist_yaw(env)[:, None]), -1)
        forces = env.contact_sensor.data.net_forces_w_history[:, :, env.termination_contact_cfg.body_ids]
        gravity = rotate_inverse(quat, data.GRAVITY_VEC_W)
        reasons = torch.stack(((forces.norm(dim=-1).amax(1) > cfg.robot.termination_contact_force).any(-1),
                               actual[:, 3] < cfg.robot.termination_pelvis_height,
                               -gravity[:, 2] < cfg.robot.termination_up_z), -1)
        captured.update(actual=actual, command=env.command_generator.full_command.clone(),
                        terminated=done & ~timeout, reasons=reasons,
                        feet_contact=(env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids] > 0),
                        knee_clipped=(env.raw_actions[:, knee_columns] != env.bounded_actions[:, knee_columns]).float(),
                        action_outside=(rewards.raw_action_target_excess(env) > 0).float().mean(-1))
        if args.gait_diagnostics:
            captured.update(
                pelvis_position=data.body_pos_w[:, env.pelvis_body_id]-env.scene.env_origins,
                pelvis_yaw=quat_yaw(quat),
                foot_clearance=sole_clearance(data.body_pos_w[:, foot_ids], data.body_quat_w[:, foot_ids],
                                               vertices, env.scene.env_origins[:, 2]))
        return done, timeout

    env.check_reset = capture_before_reset
    stand_amplitude = env.arm_stand_amplitudes.clone()
    move_amplitude = env.arm_move_amplitudes.clone()
    warmup_steps = round(args.warmup_s / env.step_dt)
    steps = warmup_steps + round(args.measure_s / env.step_dt)
    profiles = [
        {'name': 'controlled_arms_on', 'arms': True, 'noise': False, 'push': False, 'reset_velocity': False},
        {'name': 'controlled_arms_off', 'arms': False, 'noise': False, 'push': False, 'reset_velocity': False},
        {'name': 'training_disturbances', 'arms': True, 'noise': True, 'push': True, 'reset_velocity': True},
    ]
    if args.task in ('walk_elf3_tang_bxi_robust', 'walk_elf3_tang_bxi_still'):
        profiles = [
            dict(name='bootstrap', arms=True, noise=False, push=False, reset_velocity=False,
                 arm_scale=.05, guard_fixed=True),
            dict(name='controlled_arms_on', arms=True, noise=False, push=False,
                 reset_velocity=False, guard_fixed=True),
            dict(name='training_disturbances', arms=True, noise=True, push=True,
                 reset_velocity=True, guard_fixed=True),
        ]
    if args.include_guarded_neutral:
        if not cfg.arm_motion.collision_guard:
            raise ValueError('Guarded-neutral evaluation requires collision_guard')
        profiles.append({'name': 'controlled_neutral_guarded', 'arms': False, 'noise': False,
                         'push': False, 'reset_velocity': False, 'guard_fixed': True})
    if args.curriculum_stage:
        from legged_lab.envs.elf3_tang.scratch_curriculum import stage_settings
        settings = stage_settings(args.curriculum_stage)
        profiles.append(dict(name='curriculum_stage', stage=args.curriculum_stage, arms=True,
                             noise=settings['noise'], push=settings['push_scale'] > 0,
                             reset_velocity=settings['reset_scale'] > 0,
                             arm_scale=settings['arm_scale'], push_scale=settings['push_scale'],
                             reset_scale=settings['reset_scale'], guard_fixed=True))
    if args.teleop_stage:
        from tools.teleop_curriculum import evaluation_profiles, checkpoint_stage
        if checkpoint_stage(args.checkpoint) != args.teleop_stage:
            raise ValueError('Teleop evaluation must use the checkpoint training stage')
        profiles = evaluation_profiles(args.teleop_stage)
    report = {'checkpoint': str(Path(args.checkpoint).resolve()),
              'task': args.task, 'robot_joint_count': env.robot.num_joints,
              'robot_usd_path': cfg.scene.robot.spawn.usd_path,
              'arm_motion_source': cfg.arm_motion.source,
              'configuration_profile': args.profile,
              'height_command': fixed_height,
              'liftoff_reward_recipe': getattr(cfg, 'liftoff_reward_recipe', None),
              'arm_guard_enabled': cfg.arm_motion.collision_guard,
              'amass_scale': 1., 'neutral_reference_fraction_in_amass_profiles': 0.,
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'iteration': runner.current_learning_iteration, 'seed': args.seed, 'repeats': args.repeats,
              'mixed_yaw_scenarios': args.include_mixed_yaw, 'gait_diagnostics': args.gait_diagnostics,
              'auxiliary_probe': args.auxiliary_probe,
              'warmup_s': args.warmup_s, 'measure_s': args.measure_s, 'dt': env.step_dt,
              'metric_order': ['vx_m_s', 'vy_m_s', 'wz_rad_s', 'height_m', 'waist_yaw_rad'],
              'method': 'Deterministic policy; fixed commands with original pose slew rates; startup mass/friction and joint reset randomization retained. First trial only per environment; no samples after first termination. Tracking statistics use post-warmup samples including the terminal transition. Survival covers warmup plus measurement; no timeout inside evaluation horizon.',
              'profiles': []}
    if args.teleop_stage:
        report['teleop_curriculum_stage'] = args.teleop_stage
        report['amass_scale'] = None  # See each explicitly fixed profile's arm_scale.
        report['walk_then_stop_schedule'] = dict(walk_start_s=2., stop_s=10., measurement_start_s=14.)
    full_noise = env.noise_scale_vec.clone()
    dump_yaml(str(output / 'base_env.yaml'), cfg)
    for profile in profiles:
        print('EVALUATING', profile['name'], flush=True)
        env.seed(args.seed)
        cfg.arm_motion.guard_fixed_targets = profile.get('guard_fixed', False)
        env.add_noise = profile['noise']
        env.noise_scale_vec.copy_(full_noise * profile.get('noise_scale', 1.))
        profile_state['push'] = profile['push']
        # Freeze both clocks: late/early checkpoints must see the same benchmark.
        reset_scale = profile.get('reset_scale', float(profile['reset_velocity']))
        push_scale = profile.get('push_scale', float(profile['push']))
        cfg.reset_velocity_curriculum.initial_scale = cfg.reset_velocity_curriculum.final_scale = reset_scale
        cfg.push_velocity_curriculum.initial_scale = cfg.push_velocity_curriculum.final_scale = push_scale
        cfg.arm_motion.scale = profile.get('arm_scale', 1.)
        env._update_push_velocity_curriculum()
        env.arm_stand_amplitudes.copy_(stand_amplitude * float(profile['arms']))
        env.arm_move_amplitudes.copy_(move_amplitude * float(profile['arms']))
        env.arm_targets = (None if profile['arms'] else
                           env.robot.data.default_joint_pos[:, env.arm_joint_ids].clone())
        env.reset(ids)
        # Reset interval timers as well, so every profile gets a full push schedule.
        env.event_manager.reset(ids)
        clip_hash = (hashlib.sha256(env.amass.cpu().numpy().tobytes()).hexdigest()
                     if env.use_amass and profile['arms'] else None)
        obs, _ = env.get_observations()
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        first_failure = torch.full((env.num_envs,), -1., device=env.device)
        failure_reasons = torch.zeros(env.num_envs, 3, dtype=torch.bool, device=env.device)
        history = {key: [] for key in ('actual', 'command', 'valid', 'feet_contact', 'action_outside', 'knee_clipped')}
        if args.gait_diagnostics:
            history.update({key: [] for key in ('pelvis_position', 'pelvis_yaw', 'foot_clearance')})
        probe = {}
        probe_ids = ids.reshape(len(SCENARIOS),args.repeats)[:,::4].reshape(-1)
        with torch.inference_mode():
            for step in range(steps):
                if args.teleop_stage and step in (0, round(2./env.step_dt), round(10./env.step_dt)):
                    group = slice((len(SCENARIOS)-1)*args.repeats, len(SCENARIOS)*args.repeats)
                    vx = .3 if step == round(2./env.step_dt) else 0.
                    targets[group, 0] = vx
                    env.command_generator.target[group, 0] = vx
                    env.command_generator.full_command[group, 0] = vx
                    env.command_generator.is_standing_env[group] = vx == 0.
                probe_step = args.auxiliary_probe and step >= warmup_steps and step % 4 == 0
                if probe_step:
                    probe_obs = obs[probe_ids].clone()
                    probe_alive = alive[probe_ids].clone()
                obs, _, done, info = env.step(policy(obs))
                if probe_step:
                    payload = dict(info['auxiliary'])
                    payload['valid'] = payload['valid'][probe_ids] & probe_alive
                    payload = {k:(v if k=='valid' else v[probe_ids]) for k,v in payload.items()}
                    payload.update(observations=probe_obs, alive=probe_alive,
                                   scenario=probe_ids//args.repeats)
                    for key,value in payload.items():
                        probe.setdefault(key,[]).append(value.cpu().numpy())
                assert torch.isfinite(obs).all() and torch.isfinite(captured['actual']).all()
                assert not env.time_out_buf.any(), 'Timeout should be outside evaluation horizon'
                for key in history:
                    history[key].append((alive.clone() if key == 'valid' else captured[key].clone()))
                failed = alive & captured['terminated']
                first_failure[failed] = (step + 1) * env.step_dt
                failure_reasons[failed] = captured['reasons'][failed]
                alive &= ~captured['terminated']
                if (step + 1) % 200 == 0:
                    print('EVAL_PROGRESS', profile['name'], step + 1, '/', steps, 'surviving', int(alive.sum()), flush=True)
        arrays = {k: torch.stack(v).cpu().numpy() for k, v in history.items()}
        if args.auxiliary_probe:
            np.savez_compressed(output/(profile['name']+'_auxiliary_probe.npz'),
                                **{k:np.concatenate(v) for k,v in probe.items()})
        first_failure = first_failure.cpu().numpy()
        failure_reasons = failure_reasons.cpu().numpy()
        np.savez_compressed(output / (profile['name'] + '.npz'), **arrays,
                            first_failure=first_failure, failure_reasons=failure_reasons,
                            scenario_names=np.array([n for n, _ in SCENARIOS]), repeats=args.repeats)
        error = arrays['actual'] - arrays['command']
        error[..., 4] = np.arctan2(np.sin(error[..., 4]), np.cos(error[..., 4]))
        rows = []
        for index, (name, target) in enumerate(SCENARIOS):
            group = slice(index * args.repeats, (index + 1) * args.repeats)
            begin = round(14./env.step_dt) if args.teleop_stage and name == 'walk_then_stop' else warmup_steps
            mask = arrays['valid'][begin:, group]
            actual = arrays['actual'][begin:, group][mask]
            err = error[begin:, group][mask]
            failures = first_failure[group]
            row = {'scenario': name, 'target': target, 'valid_samples': int(mask.sum()),
                   'survived': int((failures < 0).sum()), 'trials': args.repeats,
                   'failure_fraction': float((failures >= 0).mean()),
                   'warmup_failures': int(((failures >= 0) & (failures <= args.warmup_s)).sum()),
                   'first_failure_times': failures.tolist(),
                   'failure_reasons_counts': failure_reasons[group].sum(0).tolist()}
            if len(err):
                row.update(actual_mean=actual.mean(0).tolist(), mae=np.abs(err).mean(0).tolist(),
                           rmse=np.sqrt((err**2).mean(0)).tolist(), p95_abs_error=np.quantile(np.abs(err), .95, axis=0).tolist(),
                           velocity_xy_error=float(np.linalg.norm(err[:, :2], axis=-1).mean()),
                           mean_actual_yaw_deg=float(actual[:, 4].mean() * 180 / math.pi),
                           yaw_mae_deg=float(np.abs(err[:, 4]).mean() * 180 / math.pi),
                           single_support_fraction=float((arrays['feet_contact'][begin:, group].sum(-1)[mask] == 1).mean()),
                           action_outside_fraction=float(arrays['action_outside'][begin:, group][mask].mean()))
                row['knee_clipped_fraction'] = dict(zip(knee_names,
                    arrays['knee_clipped'][begin:, group][mask].mean(0).tolist()))
            if args.teleop_stage and name in ('stand', 'walk_then_stop'):
                from tools.teleop_curriculum_gates import position_metrics
                row['position'] = position_metrics(arrays['pelvis_position'][:, group],
                    arrays['pelvis_yaw'][:, group], arrays['valid'][:, group], failures, begin, env.step_dt)
            rows.append(row)
            print('EVAL_RESULT', profile['name'], json.dumps(row), flush=True)
        report['profiles'].append({'settings': profile, 'initial_clip_sha256': clip_hash, 'scenarios': rows})
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.curriculum_stage:
        from scratch_training_gates import tracking_gate, full_tracking_gate
        gates = dict(stage=tracking_gate(report, args.curriculum_stage), full=full_tracking_gate(report))
        (output / 'gates.json').write_text(json.dumps(gates, indent=2, allow_nan=False) + '\n')
    print('EVALUATION_COMPLETE', str(output / 'report.json'), flush=True)


from legged_lab.utils.app_lifecycle import run_simulation
run_simulation(main, app, args.headless)
