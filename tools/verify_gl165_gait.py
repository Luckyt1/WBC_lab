"""Test anti-crossing/alternating gait costs and transitions on the real GL165 maps."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    def verify():
        from types import SimpleNamespace as NS
        import torch
        from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
        from legged_lab.assets.gl165.model import FEET_BODY_NAMES
        from legged_lab.envs.gl165.gait import AlternatingGait
        from legged_lab.envs.gl165.gl165_env import Gl165Env
        from legged_lab.envs.gl165.wbc_cfg import Gl165WbcEnvCfg

        cfg = Gl165WbcEnvCfg()
        cfg.scene.num_envs, cfg.device = 8, args.device
        cfg.noise.add_noise = False
        cfg.domain_rand.events.add_base_mass = None
        cfg.domain_rand.events.physics_material = None
        cfg.domain_rand.events.push_robot = None
        cfg.domain_rand.events.reset_base.params.update(pose_range={}, velocity_range={})
        cfg.domain_rand.events.reset_robot_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
        cfg.commands.rel_standing_envs = 0.
        cfg.commands.resampling_time_range = (1e6, 1e6)
        env = Gl165Env(cfg, headless=True)
        names = ('feet_lateral_band', 'gait_contact', 'gait_clearance', 'gait_fore_aft', 'stand_foot_alignment')
        terms = {n: env.reward_manager.get_term_cfg(n) for n in names}
        assert env.reward_manager.get_term_cfg('feet_air_time').weight == 0.
        foot_ids = terms['feet_lateral_band'].params['asset_cfg'].body_ids
        contact_ids = terms['gait_contact'].params['sensor_cfg'].body_ids
        assert [env.robot.body_names[i] for i in foot_ids] == FEET_BODY_NAMES
        assert [env.contact_sensor.body_names[i] for i in contact_ids] == FEET_BODY_NAMES
        assert all(t.weight < 0. for t in terms.values())
        checks = ['live_left_right_body_and_sensor_order', 'old_airtime_reward_disabled']

        class Scene(dict):
            pass

        data = NS(body_pos_w=torch.zeros_like(env.robot.data.body_pos_w),
                  root_quat_w=env.robot.data.root_quat_w.clone())
        data.root_quat_w[:] = data.root_quat_w.new_tensor([1., 0., 0., 0.])
        forces = torch.zeros_like(env.contact_sensor.data.net_forces_w)
        command = torch.zeros(8, 3, device=env.device)
        scene = Scene(robot=NS(data=data), contact_sensor=NS(data=NS(net_forces_w=forces)))
        scene.env_origins = torch.zeros(8, 3, device=env.device)
        gait = AlternatingGait(cfg.gait, 8, env.device, env.step_dt)
        fake = NS(scene=scene, gait=gait, cfg=cfg, command_generator=NS(command=command))

        def cost(name):
            term = terms[name]
            return term.func(fake, **term.params)

        widths = command.new_tensor([.16, .21, .26, .10, 0., -.21, .35, -.046])
        data.body_pos_w[:, foot_ids[0], 1] = widths
        band = cost('feet_lateral_band').clone()
        assert (band[:3] < 1e-10).all() and (band[3:] > 0.).all()
        assert band[5] > band[4] > band[3] and band[7] > band[4]
        yaw = command[:, 0] + 1.37
        rotation = quat_from_euler_xyz(yaw * 0., yaw * 0., yaw)
        data.root_quat_w.copy_(rotation)
        data.body_pos_w = quat_apply(rotation[:, None, :].expand(-1, data.body_pos_w.shape[1], -1),
                                    data.body_pos_w) + command.new_tensor([2., -7., 5.])
        torch.testing.assert_close(cost('feet_lateral_band'), band, atol=2e-5, rtol=1e-4)
        checks += ['16_26cm_signed_band_penalizes_crossing', 'signed_band_world_yaw_and_translation_invariance']

        data.body_pos_w.zero_()
        data.root_quat_w[:] = command.new_tensor([1., 0., 0., 0.])
        command[:, 0] = .3
        gait.phase[:] = torch.arange(8, device=env.device) / 8.
        gait.activity[:] = 1.
        gait.refresh(command)
        reference = gait.fore_aft.clone()
        assert reference.min() < -.1 and reference.max() > .1
        torch.testing.assert_close((gait.foot_phase[:, 1] - gait.foot_phase[:, 0]) % 1., command[:, 0] * 0. + .5)
        gait.phase[:] = (gait.phase + .5) % 1.
        gait.refresh(command)
        torch.testing.assert_close(gait.fore_aft, -reference)
        gait.phase[:] = (gait.phase + .5) % 1.
        command[:, 0] = -.3
        gait.refresh(command)
        torch.testing.assert_close(gait.fore_aft, -reference)
        command[:, 0] = .6
        gait.refresh(command)
        torch.testing.assert_close(gait.fore_aft, 2. * reference)
        assert (torch.sqrt(gait.fore_aft.square() + .26**2) <= .4).all()
        command[:, 0] = .3
        gait.refresh(command)
        data.body_pos_w[:, foot_ids[0], 0] = gait.fore_aft
        assert (cost('gait_fore_aft') < 1e-9).all()
        data.body_pos_w[:, foot_ids[0], 0] = -.395
        staggered = cost('gait_fore_aft').clone()
        assert (staggered > 5.).all()
        data.body_pos_w[:, foot_ids[0], 0] = 0.
        assert cost('gait_fore_aft').mean() > 0.
        for velocity in ([0., .2, 0.], [0., 0., .5], [0., 0., 0.], [.01, 0., 0.]):
            command[:] = command.new_tensor(velocity)
            gait.refresh(command)
            assert (cost('gait_fore_aft') == 0.).all()
        checks += ['half_cycle_swap_and_backward_sign', 'stride_scales_with_speed_within_total_distance',
                   'fixed_leading_foot_and_zero_stride_are_costly', 'pure_lateral_turn_and_standing_have_no_forward_target']

        command[:] = command.new_tensor([.3, 0., 0.])
        gait.refresh(command)
        forces[:, contact_ids, 2] = (~gait.swing) * 200.
        assert (cost('gait_contact') == 0.).all()
        forces[:, contact_ids, 2] = gait.swing * 200.
        assert cost('gait_contact').mean() > .5
        forces[:, contact_ids, 2] = 200.
        assert cost('gait_contact').mean() > 0.
        data.body_pos_w[:, foot_ids, 2] = .032 + gait.clearance
        assert (cost('gait_clearance') < 1e-8).all()
        data.body_pos_w[:, foot_ids, 2] = .032
        assert cost('gait_clearance').mean() > .05
        checks += ['alternating_support_beats_wrong_phase_and_double_stance', 'swing_scuffing_is_costly_without_body_height_target']

        command.zero_()
        # Simulate a moving-to-standing transition, not an already settled reset.
        gait.activity[:] = 1.
        gait.standing_time.zero_()
        data.body_pos_w[:, foot_ids[0], 0] = .4
        for _ in range(10):
            gait.update(command)
            assert (cost('stand_foot_alignment') == 0.).all()
        for _ in range(41):
            gait.update(command)
        assert (gait.activity == 0.).all() and (gait.standing_alignment == 1.).all()
        assert (cost('stand_foot_alignment') > 10.).all()
        data.body_pos_w[:, foot_ids[0], 0] = .06
        assert (cost('stand_foot_alignment') < 1e-9).all()
        assert (cost('gait_contact') == 0.).all() and (cost('gait_clearance') == 0.).all()
        phase = gait.phase.clone()
        gait.update(command)
        torch.testing.assert_close(gait.phase, phase)
        torch.testing.assert_close(gait.observations(), command.new_tensor([0., 0., 0., 1.]).expand(8, -1))
        checks += ['stop_fades_gait_then_ramps_alignment_over_one_second', 'standing_6cm_deadband_and_frozen_clock']
        command[:, 0] = .3
        for _ in range(16):
            gait.update(command)
        assert (gait.activity == 1.).all() and (gait.standing_alignment == 0.).all()
        keep = gait.observations()[1:].clone()
        gait.reset(torch.tensor([0], device=env.device), command)
        torch.testing.assert_close(gait.observations()[1:], keep)
        checks += ['start_ramp_and_partial_reset_isolation']

        env.reset(torch.arange(8, device=env.device))
        commands = command.new_tensor([[0., 0., 0.], [.3, 0., 0.], [-.3, 0., 0.], [0., .2, 0.],
                                       [0., -.2, 0.], [0., 0., .5], [0., 0., -.5], [.6, 0., 0.]])
        mins, maxs = {n: 0. for n in names}, {n: 0. for n in names}
        resets = 0
        for step in range(300):
            env.command_generator.vel_command_b.copy_(commands if step < 150 else commands * 0.)
            env.command_generator.is_standing_env[:] = False
            obs, reward, done, extra = env.step(torch.zeros(8, 15, device=env.device))
            resets += int(done.sum())
            assert obs.shape == (8, 860) and extra['observations']['critic'].shape == (8, 1050)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert (env.reward_manager._step_reward[:, env.reward_manager.active_terms.index('feet_air_time')] == 0.).all()
            assert env.command_generator.command.shape == (8, 3)
            actor, critic = env.compute_current_observations()
            torch.testing.assert_close(actor[:, -4:], env.gait.observations())
            torch.testing.assert_close(critic[:, 82:86], env.gait.observations())
            assert (env.noise_scale_vec[-4:] == 0.).all()
            assert all(torch.isfinite(v) for k, v in extra['log'].items()
                       if k.startswith(('Gait/', 'FootSpacing/', 'Posture/', 'FootLoad/')))
            for n in names:
                value = env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(n)]
                assert torch.isfinite(value).all() and (value <= 1e-7).all()
                mins[n], maxs[n] = min(mins[n], float(value.min())), max(maxs[n], float(value.max()))
        env.gait.activity[:] = 1.
        env.gait.refresh(env.command_generator.command)
        before = env.gait.observations()[1:].clone()
        env.reset(torch.tensor([0], device=env.device))
        torch.testing.assert_close(env.gait.observations()[1:], before)
        assert env.gait.activity[0] == 0.
        checks += ['300_real_physics_steps_with_arms_commands_and_stop_are_finite',
                   '860_1050_observations_include_noise_free_gait_state', 'live_partial_reset_preserves_other_gait_states']
        report = dict(passed=True, completed_at=datetime.now().isoformat(), checks=checks,
                      num_envs=8, physics_steps=300, actor_obs=860, critic_obs=1050, commands=3,
                      gait_config=cfg.gait.to_dict(), synthetic_width_cost=band.cpu().tolist(),
                      synthetic_fixed_lead_cost=staggered.cpu().tolist(),
                      weighted_reward_rate_min=mins, weighted_reward_rate_max=maxs,
                      zero_policy_resets=resets, proves_trained_gait=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('GL165_GAIT_VERIFIED', json.dumps(report), flush=True)

    from legged_lab.utils.app_lifecycle import run_simulation
    run_simulation(verify, app, args.headless)


if __name__ == '__main__':
    main()
