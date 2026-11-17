"""Exercise actual GL-165 physics, IK phases, resets and PPO observation contract."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "rsl_rl")]


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/gl165_validation/physics.json')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    def verify():
        import torch
        from legged_lab.envs.gl165.gl165_env import Gl165Env
        from legged_lab.envs.gl165.wbc_cfg import Gl165WbcEnvCfg
        from legged_lab.assets.gl165.model import ARM_JOINT_NAMES, CONTROL_PARAMETERS, CONTROL_SOURCE, POLICY_JOINT_NAMES
        from isaaclab.utils.math import quat_apply_inverse
        from tools.gl165_kinematics import ArmChain

        cfg = Gl165WbcEnvCfg()
        cfg.scene.num_envs, cfg.scene.seed = args.num_envs, 42
        cfg.device = args.device
        cfg.noise.add_noise = False
        cfg.domain_rand.events.push_robot = None
        cfg.domain_rand.events.add_base_mass = None
        cfg.domain_rand.events.reset_base.params.update(pose_range={}, velocity_range={})
        cfg.domain_rand.events.reset_robot_joints.params['position_range'] = (1., 1.)
        cfg.commands.rel_standing_envs = 1.0
        env = Gl165Env(cfg, args.headless)
        assert [env.robot.joint_names[i] for i in env.policy_joint_ids] == POLICY_JOINT_NAMES
        assert [env.robot.joint_names[i] for i in env.arm_joint_ids] == ARM_JOINT_NAMES
        assert env.num_actions == 15 and env.robot.num_joints == 31
        # Check resolved PhysX values by name, including the different waist and
        # wrist order in the ELF3 source, and exercise the action-to-target path.
        for field, values in (('kp', env.robot.root_physx_view.get_dof_stiffnesses()),
                              ('kd', env.robot.root_physx_view.get_dof_dampings())):
            expected = values.new_tensor([CONTROL_PARAMETERS[n][field] for n in env.robot.joint_names])
            torch.testing.assert_close(values, expected.expand_as(values))
        expected_scale = env.action_scale.new_tensor([CONTROL_PARAMETERS[n]['action_scale'] for n in POLICY_JOINT_NAMES])
        torch.testing.assert_close(env.action_scale, expected_scale)
        probe = torch.linspace(-.15, .15, env.num_actions, device=env.device).expand(env.num_envs, -1)
        limits = env.robot.data.soft_joint_pos_limits[:, env.policy_joint_ids]
        expected_target = (env.robot.data.default_joint_pos[:, env.policy_joint_ids]
                           + expected_scale * probe).clamp(limits[..., 0], limits[..., 1])
        env.step(probe)
        torch.testing.assert_close(env.robot.data.joint_pos_target[:, env.policy_joint_ids], expected_target)
        env.reset(torch.arange(env.num_envs, device=env.device))
        # Independently compare imported USD kinematics with the source URDF.
        torso_p = env.robot.data.body_pos_w[0, env.torso_body_id]
        torso_q = env.robot.data.body_quat_w[0, env.torso_body_id]
        wrist_relative = quat_apply_inverse(torso_q.expand(2, -1),
            env.robot.data.body_pos_w[0, env.wrist_body_ids] - torso_p)
        for i, side in enumerate(('l', 'r')):
            chain = ArmChain(Path(cfg.asset_path) / 'source/urdf/gl165.urdf', side)
            q = env.robot.data.joint_pos[0, env.arm_joint_ids].reshape(2, 7)[i].cpu().numpy()
            expected, _, _ = chain.fk(q)
            assert torch.allclose(wrist_relative[i], torch.as_tensor(expected, device=env.device, dtype=torch.float32), atol=1e-4)
        actor, extra = env.get_observations()
        assert actor.shape == (args.num_envs, 860)
        assert extra['observations']['critic'].shape == (args.num_envs, 1050)
        continuous = env.arm_motion.manifest['recipe'].get('motion_mode') == 'continuous_targets_v1'
        # Legacy banks have neutral cycles; continuous banks move immediately.
        if not continuous:
            env.arm_motion.neutral[:] = True
        actions = torch.zeros(args.num_envs, env.num_actions, device=env.device)
        phases = set()
        resets = 0
        max_delta = 0.
        feet_loaded = False
        height_after_settle = None
        snapshots = []
        original_check_reset = env.check_reset

        def trace_reset():
            result = original_check_reset()
            if len(snapshots) < 10:
                forces = env.contact_sensor.data.net_forces_w[0].norm(dim=-1)
                snapshots.append(dict(height=float(env.robot.data.root_pos_w[0, 2]),
                                      reset=bool(result[0][0]),
                                      contacts={name: float(force) for name, force in
                                                zip(env.contact_sensor.body_names, forces) if force > 1.0}))
            return result

        env.check_reset = trace_reset
        for step in range(args.steps):
            if step == 100 and not continuous:
                env.arm_motion.neutral[:] = False
            previous = env.arm_target.clone()
            actor, reward, done, extra = env.step(actions)
            assert torch.isfinite(actor).all() and torch.isfinite(reward).all()
            assert torch.isfinite(extra['observations']['critic']).all()
            assert torch.isfinite(env.robot.data.root_state_w).all()
            limits = env.robot.data.joint_pos_limits[:, env.arm_joint_ids]
            assert ((env.arm_target >= limits[..., 0]) & (env.arm_target <= limits[..., 1])).all()
            # Resets legitimately replace the target with actual reset joint positions.
            if (~done).any():
                delta = (env.arm_target - previous)[~done].abs().max().item()
                max_delta = max(max_delta, delta)
                assert delta <= .04001
            resets += int(done.sum())
            phases.update(env.arm_motion.phase.tolist())
            forces = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2]
            feet_loaded |= bool((forces.sum(-1) > 100).any())
            if step == 99:
                height_after_settle = float(env.robot.data.root_pos_w[:, 2].mean())
        # Force one timeout and verify bootstrapping flags and buffer reset.
        env.episode_length_buf[:] = env.max_episode_length - 1
        _, _, done, extra = env.step(actions)
        assert done.all() and extra['time_outs'].all()
        assert (env.episode_length_buf == 0).all()
        # An episode begins from the actual arm pose, without replaying stale targets.
        assert torch.allclose(env.arm_motion.start, env.robot.data.joint_pos[:, env.arm_joint_ids])
        report = dict(robot='gl165', num_envs=args.num_envs, steps=args.steps, joints=31, actions=15,
                      control_parameters_source=CONTROL_SOURCE,
                      physx_pd_gains_match=True, policy_action_scaling_verified=True,
                      policy_joint_names=POLICY_JOINT_NAMES, action_scale=env.action_scale.cpu().tolist(),
                      actor_obs=860, critic_obs=1050, commands=3, floating_base=not env.robot.is_fixed_base,
                      feet_loaded=feet_loaded, height_after_2s_m=height_after_settle,
                      motion_mode=env.arm_motion.manifest['recipe'].get('motion_mode', 'reach_return'),
                      ik_bank=str(cfg.ik_bank_path),
                      zero_policy_resets=resets, observed_ik_phases=sorted(phases),
                      max_reference_step_rad=max_delta, finite=True, timeout_reset_passed=True,
                      initial_contact_snapshots=snapshots)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        assert feet_loaded, f'Foot sensor never detected ground support; diagnostics: {args.output}'
        assert not any(item['reset'] for item in snapshots), f'Immediate reset at default pose: {args.output}'
        print('GL165_PHYSICS_VERIFIED', json.dumps(report), flush=True)

    from legged_lab.utils.app_lifecycle import run_simulation
    run_simulation(verify, app, args.headless)


if __name__ == '__main__':
    main()
