"""Measure GL165 standing/walking with held-out continuous arm disturbances.

Samples are captured before automatic reset. A trial ends at its first failure;
reset robots never re-enter its statistics. This does not modify training.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]
SCENARIOS = {
    'stand': (0., 0., 0.),
    'forward_030': (.3, 0., 0.),
    'forward_060': (.6, 0., 0.),
    'backward_030': (-.3, 0., 0.),
    'lateral_left': (0., .2, 0.),
    'lateral_right': (0., -.2, 0.),
    'turn_left': (0., 0., .5),
    'turn_right': (0., 0., -.5),
    'walk_stop': (.3, 0., 0.),
}


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=40.)
    parser.add_argument('--warmup', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=7103)
    parser.add_argument('--profile', choices=('controlled', 'training'), default='controlled')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 <= args.warmup < args.seconds / 2:
        parser.error('Require repeats > 0 and 0 <= warmup < seconds / 2')
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.bank = args.bank.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app

    def evaluate():
        import numpy as np
        import torch
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
        from legged_lab.envs.gl165.checkpoint import restore_checkpoint_config
        from legged_lab.envs.gl165.evaluation import configure_controlled_evaluation, load_policy_runner
        from legged_lab.envs.gl165.gl165_env import Gl165Env
        from legged_lab.envs.gl165.wbc_cfg import Gl165WbcEnvCfg

        cfg = Gl165WbcEnvCfg()
        restore_checkpoint_config(cfg, args.checkpoint)
        cfg.scene.num_envs = len(SCENARIOS) * args.repeats
        cfg.scene.seed = args.seed
        cfg.scene.max_episode_length_s = args.seconds + 1.
        cfg.scene.env_spacing = 50.  # Trials must not walk into one another.
        cfg.device = args.device
        cfg.ik_bank_path, cfg.ik_split = str(args.bank), 'test'
        cfg.commands.resampling_time_range = (1e6, 1e6)
        cfg.commands.rel_standing_envs = 0.
        if args.profile == 'controlled':
            configure_controlled_evaluation(cfg)
        env = Gl165Env(cfg, args.headless)
        assert env.arm_motion.manifest['recipe']['motion_mode'] == 'continuous_targets_v1'
        runner = load_policy_runner(env, args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        env.reset(torch.arange(env.num_envs, device=env.device))
        commands = torch.tensor(list(SCENARIOS.values()), device=env.device).repeat_interleave(args.repeats, 0)
        env.command_generator.vel_command_b.copy_(commands)
        env.command_generator.is_standing_env[:] = False
        obs, _ = env.get_observations()
        steps = round(args.seconds / env.step_dt)
        stop_step = steps // 2
        shape = (steps, env.num_envs)
        load_cfg = env.reward_manager.get_term_cfg('stand_foot_load').params
        load_foot_ids = load_cfg['sensor_cfg'].body_ids
        spacing_cfg = env.reward_manager.get_term_cfg('moving_feet_distance').params
        spacing_foot_ids = spacing_cfg['asset_cfg'].body_ids
        hip_yaw_ids, _ = env.robot.find_joints(['lhip_z_joint', 'rhip_z_joint'], preserve_order=True)
        traces = {
            'velocity': torch.empty(*shape, 3, device=env.device),
            'command': torch.empty(*shape, 3, device=env.device),
            'root_pos': torch.empty(*shape, 3, device=env.device),
            'waist': torch.empty(*shape, 3, device=env.device),
            'tilt': torch.empty(*shape, device=env.device),
            'torso_tilt': torch.empty(*shape, device=env.device),
            'base_height': torch.empty(*shape, device=env.device),
            'knee': torch.empty(*shape, 2, device=env.device),
            'foot_force_z': torch.empty(*shape, 2, device=env.device),
            'foot_distance': torch.empty(*shape, device=env.device),
            'foot_lateral_width': torch.empty(*shape, device=env.device),
            'foot_separation_yaw': torch.empty(*shape, 3, device=env.device),
            'foot_velocity_w': torch.empty(*shape, 2, 3, device=env.device),
            'foot_height': torch.empty(*shape, 2, device=env.device),
            'foot_air_time': torch.empty(*shape, 2, device=env.device),
            'gait_reference': torch.empty(*shape, 4, device=env.device),
            'foot_heading': torch.empty(*shape, 2, device=env.device),
            'foot_forward_xy_norm': torch.empty(*shape, 2, device=env.device),
            'chest_heading': torch.empty(*shape, device=env.device),
            'chest_frame_velocity': torch.empty(*shape, 2, device=env.device),
            'hip_yaw': torch.empty(*shape, 2, device=env.device),
            'feet_contact': torch.empty(*shape, 2, device=env.device, dtype=torch.bool),
            'wrist_error': torch.empty(*shape, device=env.device),
            'endpoint': torch.empty(*shape, device=env.device, dtype=torch.bool),
            'region': torch.empty(*shape, device=env.device, dtype=torch.long),
            'valid': torch.empty(*shape, device=env.device, dtype=torch.bool),
            'failed_now': torch.empty(*shape, device=env.device, dtype=torch.bool),
        }
        initial_pos = env.robot.data.root_pos_w.clone()
        failed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        first_failure = torch.full((env.num_envs,), args.seconds, device=env.device)
        check_reset = env.check_reset
        step = 0

        def capture_before_reset():
            done, timeout = check_reset()
            assert not bool(timeout.any()), 'Evaluation unexpectedly reached an episode timeout'
            data = env.robot.data
            velocity = quat_apply_inverse(yaw_quat(data.root_quat_w), data.root_lin_vel_w)
            traces['velocity'][step] = torch.cat((velocity[:, :2], data.root_ang_vel_w[:, 2:3]), -1)
            traces['command'][step] = env.command_generator.command
            traces['root_pos'][step] = data.root_pos_w - initial_pos
            traces['waist'][step] = data.joint_pos[:, env.policy_joint_ids[:3]]
            traces['tilt'][step] = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1))
            chest_gravity = quat_apply_inverse(data.body_quat_w[:, env.torso_body_id], data.GRAVITY_VEC_W)
            traces['torso_tilt'][step] = torch.acos((-chest_gravity[:, 2]).clamp(-1, 1))
            traces['base_height'][step] = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            traces['knee'][step] = data.joint_pos[:, env.knee_joint_ids]
            traces['foot_force_z'][step] = env.contact_sensor.data.net_forces_w[:, load_foot_ids, 2]
            feet = data.body_pos_w[:, spacing_foot_ids]
            separation = feet[:, 0] - feet[:, 1]
            traces['foot_distance'][step] = separation[:, :2].norm(dim=-1)
            traces['foot_lateral_width'][step] = quat_apply_inverse(yaw_quat(data.root_quat_w), separation)[:, 1].abs()
            traces['foot_separation_yaw'][step] = quat_apply_inverse(yaw_quat(data.root_quat_w), separation)
            traces['foot_velocity_w'][step] = data.body_lin_vel_w[:, spacing_foot_ids]
            traces['foot_height'][step] = feet[..., 2] - env.scene.env_origins[:, None, 2]
            traces['foot_air_time'][step] = env.contact_sensor.data.current_air_time[:, load_foot_ids]
            traces['gait_reference'][step] = torch.stack((env.gait.phase, env.gait.activity,
                env.gait.fore_aft, env.gait.standing_alignment), dim=-1)
            # Measure physical link headings: raw hip yaw alone misses the
            # coupled rotations of the GL165's angled hip mounts.
            root_forward = quat_apply(data.root_quat_w, data.FORWARD_VEC_B)
            root_heading = torch.atan2(root_forward[:, 1], root_forward[:, 0])
            foot_forward = quat_apply(data.body_quat_w[:, spacing_foot_ids],
                                     data.FORWARD_VEC_B[:, None, :].expand(-1, 2, -1))
            relative = torch.atan2(foot_forward[..., 1], foot_forward[..., 0]) - root_heading[:, None]
            traces['foot_heading'][step] = torch.atan2(torch.sin(relative), torch.cos(relative))
            traces['foot_forward_xy_norm'][step] = foot_forward[..., :2].norm(dim=-1)
            chest_quat = data.body_quat_w[:, env.torso_body_id]
            chest_forward = quat_apply(chest_quat, data.FORWARD_VEC_B)
            relative = torch.atan2(chest_forward[:, 1], chest_forward[:, 0]) - root_heading
            traces['chest_heading'][step] = torch.atan2(torch.sin(relative), torch.cos(relative))
            traces['chest_frame_velocity'][step] = quat_apply_inverse(yaw_quat(chest_quat), data.root_lin_vel_w)[:, :2]
            traces['hip_yaw'][step] = data.joint_pos[:, hip_yaw_ids]
            traces['feet_contact'][step] = env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids] > 0
            wrists = quat_apply_inverse(
                data.body_quat_w[:, env.torso_body_id, None, :].expand(-1, 2, -1),
                data.body_pos_w[:, env.wrist_body_ids] - data.body_pos_w[:, env.torso_body_id, None, :])
            traces['wrist_error'][step] = (wrists - env.arm_motion.goal_positions[env.arm_motion.index]).norm(dim=-1).amax(-1)
            traces['endpoint'][step] = env.arm_motion.phase == 2
            traces['region'][step] = env.arm_motion.regions[env.arm_motion.index]
            traces['valid'][step] = ~failed & ~done
            newly_failed = done & ~failed
            traces['failed_now'][step] = newly_failed
            first_failure[newly_failed] = (step + 1) * env.step_dt
            failed.logical_or_(done)
            return done, timeout

        env.check_reset = capture_before_reset
        print('GL165_EVAL_READY', args.checkpoint, 'profile=', args.profile, 'trials=', env.num_envs, flush=True)
        with torch.inference_mode():
            for step in range(steps):
                if step == stop_step:
                    commands[-args.repeats:] = 0.
                    env.command_generator.vel_command_b.copy_(commands)
                    obs, _ = env.get_observations()
                obs, reward, _, _ = env.step(policy(obs))
                # Reset failed trials can resample commands, but their data remain excluded.
                env.command_generator.vel_command_b.copy_(commands)
                if step % 250 == 0:
                    assert bool(torch.isfinite(obs).all() & torch.isfinite(reward).all())
                    print('GL165_EVAL_PROGRESS', step, 'first_failures=', int(failed.sum()), flush=True)

        arrays = {name: value.cpu().numpy() for name, value in traces.items()}
        finite_fields = ('velocity', 'command', 'root_pos', 'waist', 'tilt', 'torso_tilt', 'base_height',
                         'knee', 'foot_force_z', 'foot_distance', 'foot_lateral_width', 'wrist_error',
                         'foot_heading', 'foot_forward_xy_norm', 'chest_heading', 'chest_frame_velocity', 'hip_yaw')
        finite_fields += ('foot_separation_yaw', 'foot_velocity_w', 'foot_height', 'foot_air_time', 'gait_reference')
        assert all(np.isfinite(arrays[name]).all() for name in finite_fields)
        assert np.array_equal(arrays['valid'], np.cumsum(arrays['failed_now'], axis=0) == 0)
        assert (arrays['failed_now'].sum(axis=0) <= 1).all()
        expected = np.repeat(np.array(list(SCENARIOS.values())), args.repeats, axis=0)
        for begin, end in ((0, stop_step), (stop_step, steps)):
            if begin == stop_step:
                expected[-args.repeats:] = 0.
            mask = arrays['valid'][begin:end]
            error = np.abs(arrays['command'][begin:end] - expected)
            assert not mask.any() or error[mask].max() < 1e-6
        np.savez_compressed(args.output / 'traces.npz', **arrays,
                            initial_root_pos=initial_pos.cpu().numpy(), dt=env.step_dt,
                            first_failure_s=first_failure.cpu().numpy(), failed=failed.cpu().numpy())
        results = {}
        for index, name in enumerate(SCENARIOS):
            sl = slice(index * args.repeats, (index + 1) * args.repeats)
            valid = arrays['valid'][:, sl]
            measure = valid.copy()
            measure[:round(args.warmup / env.step_dt)] = False
            case_failed = failed[sl].cpu().numpy()
            result = dict(command=list(SCENARIOS[name]), trials=args.repeats,
                          completed=int((~case_failed).sum()), failures=int(case_failed.sum()),
                          survival_fraction=float((~case_failed).mean()),
                          first_failure_s=first_failure[sl].cpu().tolist())

            def describe(mask, anchor_step=None):
                if not mask.any():
                    return dict(valid_samples=0)
                velocity = arrays['velocity'][:, sl][mask]
                command = arrays['command'][:, sl][mask]
                waist = arrays['waist'][:, sl]
                pos = arrays['root_pos'][:, sl, :2].copy()
                if anchor_step is not None:
                    pos -= pos[anchor_step - 1:anchor_step]
                drift = np.linalg.norm(pos, axis=-1)
                peaks = [float(drift[:, j][mask[:, j]].max()) for j in range(args.repeats) if mask[:, j].any()]
                waist_ranges = [np.ptp(waist[:, j][mask[:, j]], axis=0) for j in range(args.repeats) if mask[:, j].any()]
                torso_tilt = np.rad2deg(arrays['torso_tilt'][:, sl][mask])
                base_height = arrays['base_height'][:, sl][mask]
                knee = np.rad2deg(arrays['knee'][:, sl][mask])
                foot_force = arrays['foot_force_z'][:, sl][mask]
                foot_distance = arrays['foot_distance'][:, sl][mask]
                foot_heading = np.rad2deg(arrays['foot_heading'][:, sl][mask])
                separation = arrays['foot_separation_yaw'][:, sl][mask]
                contacts = arrays['feet_contact'][:, sl][mask]
                foot_speed = np.linalg.norm(arrays['foot_velocity_w'][:, sl][mask, :, :2], axis=-1)
                leader_changes = []
                # A 3 cm dead band excludes sign jitter while the feet pass.
                for trial in range(args.repeats):
                    x = arrays['foot_separation_yaw'][:, sl, 0][:, trial][mask[:, trial]]
                    leader = np.sign(x[np.abs(x) > .03])
                    leader_changes.append(int(np.count_nonzero(leader[1:] != leader[:-1])))
                zero_command = (np.abs(command) <= load_cfg['command_tolerance']).all(-1)
                load = foot_force[zero_command]
                support = dict(zero_command_samples=int(zero_command.sum()))
                if len(load):
                    support.update(per_foot_mean_vertical_N=load.mean(0).tolist(),
                                   per_foot_p05_vertical_N=np.percentile(load, 5, axis=0).tolist(),
                                   both_feet_above_threshold_fraction=float((load >= load_cfg['threshold']).all(-1).mean()))
                return dict(valid_samples=int(mask.sum()), mean_velocity=velocity.mean(0).tolist(),
                            velocity_mae=np.abs(velocity - command).mean(0).tolist(),
                            tilt_p95_deg=float(np.rad2deg(np.percentile(arrays['tilt'][:, sl][mask], 95))),
                            torso_tilt_mean_deg=float(torso_tilt.mean()),
                            torso_tilt_p95_deg=float(np.percentile(torso_tilt, 95)),
                            base_height_mean_m=float(base_height.mean()),
                            base_height_p05_p50_p95_m=np.percentile(base_height, [5, 50, 95]).tolist(),
                            waist_abs_mean_deg=np.rad2deg(np.abs(waist[mask])).mean(0).tolist(),
                            knee_mean_deg=knee.mean(0).tolist(),
                            knee_p95_deg=np.percentile(knee, 95, axis=0).tolist(),
                            zero_command_foot_load=support,
                            foot_distance_mean_m=float(foot_distance.mean()),
                            foot_distance_p95_m=float(np.percentile(foot_distance, 95)),
                            foot_distance_above_limit_fraction=float((foot_distance > spacing_cfg['max_distance']).mean()),
                            foot_lateral_width_mean_m=float(arrays['foot_lateral_width'][:, sl][mask].mean()),
                            foot_signed_separation_mean_m=separation.mean(0).tolist(),
                            feet_crossed_laterally_fraction=float((separation[:, 1] < 0.).mean()),
                            left_leads_fraction=float((separation[:, 0] > .03).mean()),
                            right_leads_fraction=float((separation[:, 0] < -.03).mean()),
                            lead_changes_per_trial=leader_changes,
                            stance_foot_speed_mean_m_s=[float(foot_speed[:, j][contacts[:, j]].mean())
                                                       if contacts[:, j].any() else None for j in range(2)],
                            foot_height_p95_m=np.percentile(arrays['foot_height'][:, sl][mask], 95, axis=0).tolist(),
                            foot_air_time_p95_s=np.percentile(arrays['foot_air_time'][:, sl][mask], 95, axis=0).tolist(),
                            foot_heading_abs_mean_deg=np.abs(foot_heading).mean(0).tolist(),
                            foot_heading_abs_p95_deg=np.percentile(np.abs(foot_heading), 95, axis=0).tolist(),
                            foot_heading_circular_mean_deg=np.rad2deg(np.arctan2(
                                np.sin(np.deg2rad(foot_heading)).mean(0),
                                np.cos(np.deg2rad(foot_heading)).mean(0))).tolist(),
                            foot_heading_above_30deg_fraction=(np.abs(foot_heading) > 30.).mean(0).tolist(),
                            foot_forward_xy_norm_min=float(arrays['foot_forward_xy_norm'][:, sl][mask].min()),
                            chest_heading_abs_mean_deg=float(np.rad2deg(np.abs(arrays['chest_heading'][:, sl][mask])).mean()),
                            chest_frame_velocity_mean=arrays['chest_frame_velocity'][:, sl][mask].mean(0).tolist(),
                            hip_yaw_mean_deg=np.rad2deg(arrays['hip_yaw'][:, sl][mask]).mean(0).tolist(),
                            median_trial_max_displacement_m=float(np.median(peaks)),
                            max_trial_displacement_m=max(peaks),
                            median_waist_range_deg=np.rad2deg(np.median(waist_ranges, axis=0)).tolist(),
                            both_feet_contact_fraction=float(arrays['feet_contact'][:, sl][mask].all(-1).mean()))

            if name == 'walk_stop':
                walking = measure.copy()
                walking[stop_step:] = False
                stopped = measure.copy()
                stopped[:stop_step + round(args.warmup / env.step_dt)] = False
                result['walking'] = describe(walking)
                result['stopped'] = describe(stopped, anchor_step=stop_step)
            else:
                result.update(describe(measure))
            result['arm_endpoints'] = {}
            for region_id, region in enumerate(env.arm_motion.region_names):
                selected = valid & arrays['endpoint'][:, sl] & (arrays['region'][:, sl] == region_id)
                error = arrays['wrist_error'][:, sl][selected]
                result['arm_endpoints'][region] = dict(count=len(error),
                    mean_error_m=float(error.mean()) if len(error) else None,
                    p95_error_m=float(np.percentile(error, 95)) if len(error) else None,
                    within_5cm_fraction=float((error < .05).mean()) if len(error) else None)
            results[name] = result
        report = dict(completed_at=datetime.now().isoformat(), checkpoint=str(args.checkpoint),
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      checkpoint_iteration=runner.current_learning_iteration,
                      ik_bank=str(args.bank), ik_split='test',
                      ik_manifest_sha256=hashlib.sha256((args.bank / 'manifest.json').read_bytes()).hexdigest(),
                      evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      profile=args.profile, seconds=args.seconds, warmup_s=args.warmup,
                      gait_enabled=cfg.gait.enabled, gait_config=cfg.gait.to_dict(),
                      gait_reference_columns=['phase_cycles', 'activity', 'fore_aft_target_m', 'standing_alignment'],
                      seed=args.seed, repeats=args.repeats, deterministic_policy=True,
                      first_failure_ends_trial=True, pre_reset_capture=True,
                      foot_load_threshold_N=load_cfg['threshold'],
                      foot_distance_limit_m=spacing_cfg['max_distance'],
                      foot_load_body_order=[env.contact_sensor.body_names[i] for i in load_foot_ids],
                      heading_reference='pelvis horizontal +X; physical foot/chest +X projected onto world XY',
                      foot_heading_body_order=[env.robot.body_names[i] for i in spacing_foot_ids],
                      region_probabilities=env.arm_motion.region_probabilities,
                      settings=dict(observation_noise=cfg.noise.add_noise,
                                    random_mass=cfg.domain_rand.events.add_base_mass is not None,
                                    random_friction=cfg.domain_rand.events.physics_material is not None,
                                    random_pushes=cfg.domain_rand.events.push_robot is not None),
                      results=results)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('GL165_EVAL_COMPLETE', json.dumps(report), flush=True)

    from legged_lab.utils.app_lifecycle import run_simulation
    run_simulation(evaluate, app, args.headless)


if __name__ == '__main__':
    main()
