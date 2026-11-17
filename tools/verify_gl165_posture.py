"""Verify GL165 posture penalties against actual link/joint maps and physics."""

import argparse
from datetime import datetime
import json
import math
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
        from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
        from legged_lab.assets.gl165.model import POLICY_JOINT_NAMES, FEET_BODY_NAMES
        from legged_lab.envs.gl165.gl165_env import Gl165Env
        from legged_lab.envs.gl165.wbc_cfg import Gl165WbcEnvCfg

        cfg = Gl165WbcEnvCfg()
        cfg.scene.num_envs = 8
        cfg.device = args.device
        cfg.noise.add_noise = False
        cfg.domain_rand.events.add_base_mass = None
        cfg.domain_rand.events.push_robot = None
        cfg.domain_rand.events.reset_base.params.update(pose_range={}, velocity_range={})
        cfg.domain_rand.events.reset_robot_joints.params.update(position_range=(1., 1.), velocity_range=(0., 0.))
        cfg.commands.rel_standing_envs = 1.
        env = Gl165Env(cfg, headless=True)
        names = ('torso_upright', 'waist_range', 'waist_velocity', 'stand_leg_posture', 'support_knee_bend',
                 'stand_foot_load', 'moving_feet_distance', 'feet_heading', 'hip_yaw_range')
        terms = {n: env.reward_manager.get_term_cfg(n) for n in names}
        assert all(t.weight < 0. for t in terms.values())
        chest = terms['torso_upright'].params['asset_cfg'].body_ids
        assert [env.robot.body_names[i] for i in chest] == ['body_y_link']
        waist = terms['waist_range'].params['asset_cfg'].joint_ids
        legs = terms['stand_leg_posture'].params['asset_cfg'].joint_ids
        knees = terms['support_knee_bend'].params['asset_cfg'].joint_ids
        feet = terms['support_knee_bend'].params['sensor_cfg'].body_ids
        assert [env.robot.joint_names[i] for i in waist] == POLICY_JOINT_NAMES[:3]
        assert [env.robot.joint_names[i] for i in legs] == POLICY_JOINT_NAMES[3:]
        assert [env.robot.joint_names[i] for i in knees] == ['lknee_y_joint', 'rknee_y_joint']
        assert [env.contact_sensor.body_names[i] for i in feet] == FEET_BODY_NAMES
        load_feet = terms['stand_foot_load'].params['sensor_cfg'].body_ids
        assert [env.contact_sensor.body_names[i] for i in load_feet] == FEET_BODY_NAMES
        spacing_feet = terms['moving_feet_distance'].params['asset_cfg'].body_ids
        assert [env.robot.body_names[i] for i in spacing_feet] == FEET_BODY_NAMES
        heading_feet = terms['feet_heading'].params['asset_cfg'].body_ids
        hip_yaw = terms['hip_yaw_range'].params['asset_cfg'].joint_ids
        assert [env.robot.body_names[i] for i in heading_feet] == FEET_BODY_NAMES
        assert [env.robot.joint_names[i] for i in hip_yaw] == ['lhip_z_joint', 'rhip_z_joint']
        checks = ['live_chest_waist_leg_knee_and_foot_maps']

        def cost(name, target=env):
            t = terms[name]
            return t.func(target, **t.params)

        # Actual USD articulation: level pelvis with a bent waist must cost more.
        root = env.robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        q = env.robot.data.default_joint_pos.clone()
        q[1, waist[2]] = math.radians(40.)
        q[2, waist[2]] = math.radians(-40.)
        q[3, waist[1]] = math.radians(40.)
        q[4, waist[2]] = math.radians(40.)
        q[6, waist[2]] = math.radians(-20.)
        q[7, waist[2]] = math.radians(5.)
        roll = torch.zeros(env.num_envs, device=env.device)
        pitch = roll.clone()
        yaw = roll.clone()
        yaw[4:6] = 1.2
        pitch[6] = math.radians(20.)
        root[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        env.robot.write_root_state_to_sim(root)
        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.physics_dt)
        chest_cost = cost('torso_upright')
        expected = q.new_tensor([2. * (1. - math.cos(math.radians(a)))
                                 for a in (0., 40., 40., 40., 40., 0., 0., 5.)])
        torch.testing.assert_close(chest_cost, expected, atol=2e-5, rtol=1e-4)
        assert cost('waist_range')[7] == 0.  # Small balancing motion stays within the band.
        assert cost('waist_range')[6] > 0.  # Counter-bending cannot hide a large waist angle.
        checks += ['live_level_pelvis_bent_chest_is_penalized', 'live_roll_pitch_and_heading_invariance',
                   'live_compensated_chest_and_small_waist_motion']

        # Real USD kinematics: neutral feet face forward despite nonzero hip
        # yaw defaults; rotating both legs sideways must incur the new costs.
        neutral_heading_cost = cost('feet_heading').clone()
        assert (neutral_heading_cost < 1e-8).all()
        assert (cost('hip_yaw_range') < 1e-8).all()
        q[:, hip_yaw] += math.radians(60.)
        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.physics_dt)
        live_crab_cost = cost('feet_heading').clone()
        assert (live_crab_cost > .2).all() and (cost('hip_yaw_range') > .8).all()
        checks += ['live_neutral_angled_hip_mounts_have_forward_feet', 'live_same_side_leg_twist_is_penalized']

        # Controlled sensor states test standing vs. left/right support without
        # depending on a floating foot happening to collide during setup.
        data = NS(joint_pos=env.robot.data.default_joint_pos.clone(),
                  default_joint_pos=env.robot.data.default_joint_pos.clone(),
                  joint_vel=torch.zeros_like(q),
                  body_pos_w=env.robot.data.body_pos_w.clone(),
                  body_quat_w=env.robot.data.body_quat_w.clone(),
                  root_quat_w=env.robot.data.root_quat_w.clone(),
                  FORWARD_VEC_B=env.robot.data.FORWARD_VEC_B.clone(),
                  GRAVITY_VEC_W=env.robot.data.GRAVITY_VEC_W.clone())
        data.body_quat_w[:] = q.new_tensor([1., 0., 0., 0.])
        contacts = torch.zeros_like(env.contact_sensor.data.current_contact_time)
        forces = torch.zeros_like(env.contact_sensor.data.net_forces_w)
        fake = NS(scene={'robot': NS(data=data),
                         'contact_sensor': NS(data=NS(current_contact_time=contacts, net_forces_w=forces))},
                  command_generator=NS(command=torch.zeros(env.num_envs, 3, device=env.device)))
        data.body_quat_w[0, chest[0]] = q.new_tensor([0., 1., 0., 0.])
        assert cost('torso_upright', fake)[0] > 3.99  # An inverted chest cannot earn upright reward.
        data.body_quat_w[:] = q.new_tensor([1., 0., 0., 0.])
        data.joint_pos[0, waist] += q.new_tensor([.1, .05, -.05])
        data.joint_pos[1, waist[2]] = math.radians(40.)
        data.joint_pos[2, waist[2]] = math.radians(-40.)
        data.joint_pos[3, env.arm_joint_ids] += .5
        w = cost('waist_range', fake)
        assert w[0] == 0. and w[1] > 0. and w[3] == 0.
        torch.testing.assert_close(w[1], w[2])
        data.joint_vel[0, waist[1]] = 1.
        data.joint_vel[1, env.arm_joint_ids] = 2.
        assert cost('waist_velocity', fake)[0] > 0. and cost('waist_velocity', fake)[1] == 0.
        checks += ['inverted_chest_is_costly', 'waist_band_symmetry_and_external_arm_isolation']

        data.joint_pos.copy_(data.default_joint_pos)
        data.joint_pos[:, knees] = math.radians(90.)
        fake.command_generator.command[2:, 0] = .3
        contacts[2:, feet[0]] = .1
        data.joint_pos[2, knees[0]] = math.radians(60.)  # Deep swing, normal support.
        data.joint_pos[3, knees[1]] = math.radians(60.)  # Deep support, normal swing.
        contacts[4] = 0.  # No support in this moving sample.
        fake.command_generator.command[5] = q.new_tensor([0., 0., .5])  # Turning is moving too.
        data.joint_pos[6, knees] = math.radians(75.)
        data.joint_pos[7, knees] = math.radians(37.3)
        k = cost('support_knee_bend', fake)
        assert k[0] > 0. and k[1] > 0.  # Standing cost includes airborne knees.
        assert k[2] == 0. and k[3] > 0. and k[4] == 0. and k[5] > 0.
        assert k[6] < 1e-10 and k[7] == 0.
        torch.testing.assert_close(k[0], k[1])
        posture = cost('stand_leg_posture', fake)
        assert posture[0] > 0. and torch.all(posture[2:] == 0.)
        checks += ['standing_deep_knees_even_without_contact', 'ordered_support_vs_swing_knee_cost',
                   'moving_and_turning_keep_swing_flexion', 'standing_posture_is_not_a_walking_pose_target']

        fake.command_generator.command.zero_()
        forces[:, load_feet, 2] = q.new_tensor([[150., 150.], [149., 151.], [0., 400.], [400., 0.],
                                               [100., 100.], [200., 200.], [-10., 150.], [0., 150.]])
        forces[7, load_feet, :2] = 1000.  # A large horizontal force is not vertical support.
        load_cost = cost('stand_foot_load', fake)
        torch.testing.assert_close(load_cost, q.new_tensor([0., 1./300., .5, .5, 1./3., 0., .5, .5]))
        forces.zero_()
        fake.command_generator.command[:] = q.new_tensor([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.],
            [0., 0., .01], [-.01, 0., 0.], [.2, -.1, .5], [1e-7, 0., 0.], [2e-6, 0., 0.]])
        data.root_lin_vel_b = torch.zeros(env.num_envs, 3, device=env.device)
        data.root_lin_vel_b[0, 0] = .3  # A push must not switch off a zero-command standing cost.
        gated_load = cost('stand_foot_load', fake)
        torch.testing.assert_close(gated_load, q.new_tensor([1., 0., 0., 0., 0., 0., 1., 0.]))
        checks += ['150N_per_foot_boundary_and_no_cross_foot_compensation', 'only_vertical_load_counts',
                   'small_motion_and_turn_commands_disable_load_penalty', 'actual_drift_does_not_disable_zero_command_load']

        # Foot separation is horizontal and command-gated, regardless of contact.
        data.body_pos_w.zero_()
        fake.command_generator.command[:] = q.new_tensor([.3, 0., 0.])
        data.body_pos_w[:, spacing_feet[0], 1] = q.new_tensor([.21, .4, .41, .8, 0., .21, .8, .8])
        data.body_pos_w[4, spacing_feet[0], 0] = .8  # Same separation after a 90-degree rotation.
        data.body_pos_w[5, spacing_feet[0], 2] = 1.0  # Swing height is not foot separation.
        fake.command_generator.command[6] = 0.
        data.root_lin_vel_b[6, 0] = .3  # A push in zero-command standing stays outside this cost.
        fake.command_generator.command[7] = q.new_tensor([0., 0., .5])
        spacing_cost = cost('moving_feet_distance', fake)
        assert spacing_cost[0] == 0. and spacing_cost[1] < 1e-10
        assert 0. < spacing_cost[2] < spacing_cost[3]
        torch.testing.assert_close(spacing_cost[3], q.new_tensor(1.))
        torch.testing.assert_close(spacing_cost[3], spacing_cost[4])
        assert spacing_cost[5] == 0. and spacing_cost[6] == 0.
        torch.testing.assert_close(spacing_cost[3], spacing_cost[7])
        data.body_pos_w += q.new_tensor([10., -4., 2.])
        torch.testing.assert_close(cost('moving_feet_distance', fake), spacing_cost, atol=2e-6, rtol=1e-4)
        fake.command_generator.command[:] = q.new_tensor([0., .01, 0.])
        assert cost('moving_feet_distance', fake)[6] > 0.  # Even a small lateral command activates it.
        checks += ['live_foot_body_map_and_40cm_distance_boundary', 'distance_cost_rotational_and_translation_invariance',
                   'swing_height_exempt_but_airborne_separation_counted', 'walking_lateral_turning_vs_zero_command_spacing_gate']

        # Facing is per foot: opposing yaw must not cancel, and facing
        # backwards is worse than facing sideways (no sin(yaw)^2 loophole).
        data.root_quat_w[:] = q.new_tensor([1., 0., 0., 0.])
        angles = torch.deg2rad(q.new_tensor([[0., 0.], [15., -15.], [20., -20.], [60., 60.],
                                           [60., -60.], [180., -180.], [60., 0.], [-60., 0.]]))
        zero = torch.zeros_like(angles)
        data.body_quat_w[:, heading_feet] = quat_from_euler_xyz(zero, zero, angles)
        heading_cost = cost('feet_heading', fake).clone()
        expected = (angles.abs() - math.radians(15.)).clamp_min(0.).square().mean(-1)
        torch.testing.assert_close(heading_cost, expected, atol=2e-6, rtol=1e-5)
        assert heading_cost[0] == 0. and heading_cost[1] < 1e-10
        assert heading_cost[2] < heading_cost[3] < heading_cost[5]
        torch.testing.assert_close(heading_cost[3], heading_cost[4])
        torch.testing.assert_close(heading_cost[6], heading_cost[7])
        torch.testing.assert_close(heading_cost[6] * 2., heading_cost[3])
        # Rotate through the +/-pi discontinuity while keeping relative yaw.
        global_yaw = quat_from_euler_xyz(zero[:, 0], zero[:, 0], zero[:, 0] + 3.13)
        data.root_quat_w.copy_(global_yaw)
        data.body_quat_w[:, heading_feet] = quat_mul(global_yaw[:, None, :].expand(-1, 2, -1),
                                                    data.body_quat_w[:, heading_feet])
        torch.testing.assert_close(cost('feet_heading', fake), heading_cost, atol=2e-6, rtol=1e-5)
        checks += ['heading_15deg_band_and_monotonic_sideways_to_backwards_cost',
                   'per_foot_yaw_does_not_cancel', 'heading_global_rotation_and_pi_wrap_invariance']

        # Lateral, backward and turn commands keep the same physical facing
        # reference; an airborne foot cannot evade the orientation cost.
        fake.command_generator.command[:] = q.new_tensor([[0., 0., 0.], [.3, 0., 0.], [0., .2, 0.],
            [0., -.2, 0.], [-.3, 0., 0.], [0., 0., .5], [0., 0., -.5], [.3, .2, .5]])
        contacts.zero_()
        torch.testing.assert_close(cost('feet_heading', fake), heading_cost, atol=2e-6, rtol=1e-5)
        data.body_quat_w[:, heading_feet] = global_yaw[:, None, :]
        assert (cost('feet_heading', fake) < 1e-10).all()
        data.root_quat_w[:] = q.new_tensor([1., 0., 0., 0.])
        pitch = zero + math.radians(50.)
        pitch[1:3] = math.pi / 2.
        data.body_quat_w[:, heading_feet] = quat_from_euler_xyz(zero + .3, pitch, zero)
        pitch_cost = cost('feet_heading', fake)
        assert pitch_cost[0] == 0. and (pitch_cost[1:3] > 1.6).all()
        checks += ['lateral_backward_turn_commands_allow_forward_feet', 'airborne_feet_keep_facing_cost',
                   'normal_swing_pitch_free_but_vertical_projection_cannot_hide_yaw']

        data.joint_pos.copy_(data.default_joint_pos)
        data.joint_pos[0, hip_yaw] += math.radians(20.)
        data.joint_pos[1, hip_yaw] += math.radians(60.)
        data.joint_pos[2, hip_yaw] -= math.radians(60.)
        data.joint_pos[3, env.arm_joint_ids] += .5
        hip_cost = cost('hip_yaw_range', fake)
        assert hip_cost[0] < 1e-10 and hip_cost[1] > .8 and hip_cost[3] == 0.
        torch.testing.assert_close(hip_cost[1], hip_cost[2])
        checks += ['hip_yaw_band_uses_compensated_neutral_and_excludes_arms']

        env.reset(torch.arange(env.num_envs, device=env.device))
        min_weighted = {name: 0. for name in names}
        max_weighted = {name: -float('inf') for name in names}
        for _ in range(200):
            obs, reward, _, extra = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert obs.shape == (8, 860) and extra['observations']['critic'].shape == (8, 1050)
            assert env.command_generator.command.shape == (8, 3)
            assert all(torch.isfinite(v) for k, v in extra['log'].items()
                       if k.startswith(('Posture/', 'FootLoad/', 'FootSpacing/', 'FootHeading/')))
            for name in names:
                idx = env.reward_manager.active_terms.index(name)
                weighted = env.reward_manager._step_reward[:, idx]
                assert torch.isfinite(weighted).all() and (weighted <= 1e-7).all()
                min_weighted[name] = min(min_weighted[name], float(weighted.min()))
                max_weighted[name] = max(max_weighted[name], float(weighted.max()))
        checks += ['200_physics_steps_with_ik_finite_nonpositive_costs', 'three_commands_and_observed_gait_contract']
        report = dict(completed_at=datetime.now().isoformat(), checks=checks, passed=True,
                      live_chest_cost=chest_cost.cpu().tolist(), synthetic_knee_cost=k.cpu().tolist(),
                      synthetic_foot_load_cost=load_cost.cpu().tolist(), command_gated_foot_load_cost=gated_load.cpu().tolist(),
                      synthetic_foot_spacing_cost=spacing_cost.cpu().tolist(),
                      neutral_heading_cost=neutral_heading_cost.cpu().tolist(),
                      live_crab_heading_cost=live_crab_cost.cpu().tolist(),
                      synthetic_foot_heading_cost=heading_cost.cpu().tolist(),
                      synthetic_hip_yaw_cost=hip_cost.cpu().tolist(),
                      weights={n: t.weight for n, t in terms.items()},
                      min_weighted_reward_rate=min_weighted, max_weighted_reward_rate=max_weighted,
                      steps=200, num_envs=8, actor_obs=860, critic_obs=1050, commands=3,
                      ik_bank=cfg.ik_bank_path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('GL165_POSTURE_VERIFIED', json.dumps(report), flush=True)

    from legged_lab.utils.app_lifecycle import run_simulation
    run_simulation(verify, app, args.headless)


if __name__ == '__main__':
    main()
