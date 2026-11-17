"""Show GL-165 balancing against arm IK motion with a trained waist/leg policy."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]

DEMO_SEGMENTS = (
    (8., 'Stand / arm disturbance', (0., 0., 0.)),
    (12., 'Walk forward', (.3, 0., 0.)),
    (8., 'Move left', (0., .2, 0.)),
    (8., 'Move right', (0., -.2, 0.)),
    (8., 'Turn left', (0., 0., .5)),
    (8., 'Turn right', (0., 0., -.5)),
    (8., 'Stop and balance', (0., 0., 0.)),
)


def demo_command(elapsed):
    """Return the phase and command in a repeating, 60-second demonstration."""
    remaining = elapsed % sum(duration for duration, _, _ in DEMO_SEGMENTS)
    for duration, label, command in DEMO_SEGMENTS:
        if remaining < duration:
            return label, command
        remaining -= duration
    return DEMO_SEGMENTS[-1][1:]


class FootSpacingRuler:
    """Visual-only, meter-scale ground ruler beside the projected foot distance."""

    def __init__(self, env):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        params = env.reward_manager.get_term_cfg('moving_feet_distance').params
        self.foot_ids = params['asset_cfg'].body_ids
        self.limit = params['max_distance']
        self.ground_z = float(env.scene.env_origins[0, 2]) + .035
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path='/World/ViewerFootSpacing',
            markers={
                name: sim_utils.CuboidCfg(size=(1., .012, .012),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
                for name, color in [('within', (.05, .85, .18)), ('over', (1., .08, .03)),
                                    ('reference', (.02, .65, 1.))]
            }))

    def update(self, data):
        import math
        import numpy as np

        feet = data.body_pos_w[0, self.foot_ids].detach().cpu().numpy().copy()
        feet[:, 2] = self.ground_z
        delta = feet[0] - feet[1]
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-8)
        if distance < 1e-8:
            direction = np.array([0., 1., 0.])
        perpendicular = np.array([direction[1], -direction[0], 0.])
        # Place the reference beside the measured line, parallel at true scale.
        # Both are outside the robot hierarchy and have no physics/collision APIs.
        midpoint = (feet[0] + feet[1]) / 2.
        reference = midpoint + .35 * perpendicular
        positions, orientations, scales, indices = [], [], [], []

        def line(start, end, prototype):
            vector = end - start
            angle = math.atan2(vector[1], vector[0])
            positions.append((start + end) / 2.)
            orientations.append([math.cos(angle / 2.), 0., 0., math.sin(angle / 2.)])
            scales.append([max(float(np.linalg.norm(vector)), 1e-6), 1., 1.])
            indices.append(prototype)

        color = int(distance > self.limit)
        line(feet[0], feet[1], color)
        for foot in feet:
            line(foot - .045 * perpendicular, foot + .045 * perpendicular, color)
        line(reference - self.limit / 2. * direction, reference + self.limit / 2. * direction, 2)
        for fraction in (0., .25, .5, .75, 1.):
            tick = reference + (fraction - .5) * self.limit * direction
            half_length = .045 if fraction in (0., 1.) else .025
            line(tick - half_length * perpendicular, tick + half_length * perpendicular, 2)
        self.markers.visualize(translations=np.asarray(positions, dtype=np.float32),
                               orientations=np.asarray(orientations, dtype=np.float32),
                               scales=np.asarray(scales, dtype=np.float32), marker_indices=indices)
        return distance


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--command', type=float, nargs=3, default=(0., 0., 0.), metavar=('VX', 'VY', 'WZ'))
    parser.add_argument('--arm-group', choices=('both', 'left', 'right', 'all'), default='both')
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--demo', action='store_true', help='Cycle through standing, walking, strafing and turning.')
    parser.add_argument('--follow-camera', action='store_true',
                        help='Continuously follow the robot; by default the camera is freely movable.')
    parser.add_argument('--show-foot-spacing', action='store_true',
                        help='Show actual horizontal foot separation and a true-scale reference ruler for robot 0.')
    parser.add_argument('--diagnostics', type=Path, help='Write physical joint-connection and rendering diagnostics.')
    parser.add_argument('--max_steps', type=int)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve(strict=True)
    if args.bank is not None:
        args.bank = args.bank.resolve(strict=True)
    app = AppLauncher(args).app

    def play():
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
        from legged_lab.envs.gl165.checkpoint import restore_checkpoint_config
        from legged_lab.envs.gl165.evaluation import configure_controlled_evaluation, load_policy_runner
        from legged_lab.envs.gl165.gl165_env import Gl165Env
        from legged_lab.envs.gl165.wbc_cfg import Gl165WbcEnvCfg

        cfg = Gl165WbcEnvCfg()
        restore_checkpoint_config(cfg, args.checkpoint)
        cfg.scene.num_envs, cfg.scene.seed = args.num_envs, 42
        cfg.scene.max_episode_length_s = 3600. if args.demo else cfg.scene.max_episode_length_s
        cfg.device = args.device
        cfg.ik_split = args.split
        if args.bank is not None:
            cfg.ik_bank_path = str(args.bank)
        configure_controlled_evaluation(cfg)
        cfg.commands.rel_standing_envs = 0.
        cfg.commands.resampling_time_range = (1e6, 1e6)
        for key, value in zip(('lin_vel_x', 'lin_vel_y', 'ang_vel_z'), args.command):
            setattr(cfg.commands.ranges, key, (value, value))
        env = Gl165Env(cfg, args.headless)
        # Legacy previews skip neutral cycles; continuous banks already have none.
        env.arm_motion.manifest['recipe']['neutral_fraction'] = 0.
        if args.arm_group != 'all':
            group = env.arm_motion.group_names.index(args.arm_group)
            weights = env.arm_motion.weights * (env.arm_motion.groups == group)
            env.arm_motion.weights = weights / weights.sum()
        env.reset(torch.arange(env.num_envs, device=env.device))
        runner = load_policy_runner(env, args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        env.command_generator.vel_command_b[:] = torch.tensor(args.command, device=env.device)
        env.command_generator.is_standing_env[:] = False
        labels = None
        status_window = None
        spacing_ruler = None
        spacing_label = None
        if not args.headless:
            import omni.appwindow
            import omni.ui as ui
            window = omni.appwindow.get_default_app_window()
            if hasattr(window, 'set_title'):
                mode = 'Foot spacing ruler' if args.show_foot_spacing else 'Live policy demo'
                window.set_title(f'GL165 | Model {runner.current_learning_iteration} | {mode}')
            # Keep the imported robot's visual hierarchy/material bindings intact
            # after PhysX/Fabric initialization.
            # Visual-only meter grid; these lines have no collision properties.
            grid_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.32, .35, .39))
            for coordinate in range(-20, 21):
                for axis in (0, 1):
                    line = sim_utils.CuboidCfg(
                        size=(.012, 40., .002) if axis == 0 else (40., .012, .002),
                        visual_material=grid_material)
                    pos = (float(coordinate), 0., .002) if axis == 0 else (0., float(coordinate), .002)
                    line.func(f'/World/ViewerGrid/Line_{axis}_{coordinate + 20}', line, translation=pos)
            if args.show_foot_spacing:
                spacing_ruler = FootSpacingRuler(env)
            status_window = ui.Window('GL165 status', width=550, height=430 if spacing_ruler else 290)
            with status_window.frame:
                with ui.VStack(spacing=6, style={'Label': {'font_size': 17, 'color': 0xffffffff}}):
                    if spacing_ruler:
                        ui.Label(f'BLUE ruler = {spacing_ruler.limit * 100:.0f} cm', height=40,
                                 style={'font_size': 27, 'color': 0xffffa605})
                        ui.Label(f'Each division = {spacing_ruler.limit * 25:.0f} cm', height=28)
                        spacing_label = ui.Label('', height=42, style={'font_size': 25})
                        ui.Label('Green: within limit | Red: over limit\nDistance between foot-link origins, projected onto the floor.',
                                 height=50, word_wrap=True)
                    labels = [ui.Label('', height=38, word_wrap=True) for _ in range(6)]
                    ui.Spacer()
            status_window.deferred_dock_in('Stage', ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
            panel = ui.Workspace.get_window('Simulation Settings')
            if panel:
                panel.visible = False

        connections = []
        diagnostic_samples = []
        if args.diagnostics is not None:
            from pxr import Usd, UsdPhysics
            body_ids = {name: i for i, name in enumerate(env.robot.body_names)}
            for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')):
                if not prim.IsA(UsdPhysics.Joint):
                    continue
                joint = UsdPhysics.Joint(prim)
                parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if not parents or not children:
                    continue
                parent, child = parents[0].name, children[0].name
                if parent in body_ids and child in body_ids:
                    connections.append((prim.GetName(), body_ids[parent], body_ids[child],
                                        list(joint.GetLocalPos0Attr().Get()), list(joint.GetLocalPos1Attr().Get())))
            assert len(connections) == 31, f'Expected all 31 physical joints, got {len(connections)}'
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)

        def check_connections(step):
            if not connections:
                return
            data = env.robot.data
            gaps = {}
            for name, parent, child, local0, local1 in connections:
                p0 = data.body_pos_w[0, parent] + quat_apply(data.body_quat_w[0, parent],
                                                          torch.tensor(local0, device=env.device))
                p1 = data.body_pos_w[0, child] + quat_apply(data.body_quat_w[0, child],
                                                         torch.tensor(local1, device=env.device))
                gaps[name] = float((p0 - p1).norm())
            assert bool(torch.isfinite(data.body_state_w).all() & torch.isfinite(data.joint_pos).all())
            sample = dict(step=step, time_s=step * env.step_dt, max_joint_anchor_gap_m=max(gaps.values()),
                          joint_anchor_gaps_m=gaps, root_position=data.root_pos_w[0].cpu().tolist(),
                          body_positions={n: p for n, p in zip(env.robot.body_names, data.body_pos_w[0].cpu().tolist())},
                          failures=failures)
            if spacing_ruler:
                feet = data.body_pos_w[0, spacing_ruler.foot_ids]
                sample['foot_distance_m'] = float((feet[0, :2] - feet[1, :2]).norm())
                sample['reference_ruler_length_m'] = spacing_ruler.limit
            diagnostic_samples.append(sample)
            args.diagnostics.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), samples=diagnostic_samples), indent=2) + '\n')
            print('GL165_VIEWER_CONNECTIONS', step, 'max_anchor_gap_m=', max(gaps.values()), flush=True)

        camera_initialized = False

        def update_view(label, command, failures):
            nonlocal camera_initialized
            if args.headless:
                return
            data = env.robot.data
            root = data.root_pos_w[0].detach().cpu().tolist()
            if not camera_initialized or args.follow_camera:
                top = float(data.body_pos_w[0, :, 2].max()) + .15
                center_z = max(.25, (top + .03) / 2. - .15)
                if spacing_ruler:
                    forward = quat_apply(yaw_quat(data.root_quat_w[:1]), data.root_pos_w.new_tensor([[1., 0., 0.]]))[0]
                    env.sim.set_camera_view(eye=(root[0] + 2.8 * float(forward[0]),
                                                root[1] + 2.8 * float(forward[1]), center_z + .95),
                                            target=(root[0], root[1], center_z))
                else:
                    env.sim.set_camera_view(eye=(root[0] + 2.8, root[1] - 2.8, center_z + 1.),
                                            target=(root[0], root[1], center_z))
                camera_initialized = True
            if spacing_ruler:
                distance = spacing_ruler.update(data)
                spacing_label.text = f'Actual foot distance: {distance * 100:.1f} cm'
                spacing_label.style = {'font_size': 25, 'color': 0xff4040ff if distance > spacing_ruler.limit else 0xff55ee55}
            if labels is not None:
                velocity = quat_apply_inverse(yaw_quat(data.root_quat_w), data.root_lin_vel_w)[0]
                waist = torch.rad2deg(data.joint_pos[0, env.policy_joint_ids[:3]])
                gravity = quat_apply_inverse(data.body_quat_w[:, env.torso_body_id], data.GRAVITY_VEC_W)
                chest = float(torch.rad2deg(torch.acos((-gravity[0, 2]).clamp(-1., 1.))))
                load_cfg = env.reward_manager.get_term_cfg('stand_foot_load').params
                forces = env.contact_sensor.data.net_forces_w[0, load_cfg['sensor_cfg'].body_ids, 2]
                text = (
                    f'GL165 | Model {runner.current_learning_iteration} | {label}',
                    f'Command: vx {command[0]:+.2f}  vy {command[1]:+.2f} m/s  turn {command[2]:+.2f} rad/s',
                    f'Actual:    vx {float(velocity[0]):+.2f}  vy {float(velocity[1]):+.2f} m/s  turn {float(data.root_ang_vel_w[0, 2]):+.2f} rad/s',
                    f'Pelvis height {root[2]:.3f} m | Chest tilt {chest:.1f} deg | Waist pitch {float(waist[2]):+.1f} deg',
                    f'Foot load L {float(forces[0]):.0f} N / R {float(forces[1]):.0f} N | Zero-command threshold 150 N per foot',
                    f'Continuous arm disturbance ({args.split} bank) | No extra pushes | Failure resets: {failures}',
                )
                for widget, value in zip(labels, text):
                    widget.text = value

        obs, _ = env.get_observations()
        print('GL165_VIEWER_READY', str(args.checkpoint), 'command=', args.command,
              'arm_group=', args.arm_group, 'demo=', args.demo, 'split=', args.split,
              'foot_spacing_ruler=', args.show_foot_spacing, 'follow_camera=', args.follow_camera, flush=True)
        step = 0
        failures = 0
        previous_label = None
        while app.is_running() and (args.max_steps is None or step < args.max_steps):
            started = time.monotonic()
            label, command = demo_command(step * env.step_dt) if args.demo else ('Fixed command', args.command)
            env.command_generator.vel_command_b[:] = torch.tensor(command, device=env.device)
            if label != previous_label:
                obs, _ = env.get_observations()
                previous_label = label
                print('GL165_VIEWER_SEGMENT', step * env.step_dt, label, command, flush=True)
            with torch.inference_mode():
                obs, _, done, _ = env.step(policy(obs))
            if bool(done.any()):
                failures += int((done & ~env.time_out_buf).sum())
                env.command_generator.vel_command_b[:] = torch.tensor(command, device=env.device)
                obs, _ = env.get_observations()
                print('GL165_VIEWER_RESET', step * env.step_dt, 'failures=', failures, flush=True)
            if step % 5 == 0:
                update_view(label, command, failures)
            if step % 250 == 0:
                check_connections(step)
                print('GL165_VIEWER_STATE', dict(step=step,
                    label=label, command=command, failures=failures,
                    pelvis_height_m=float(env.robot.data.root_pos_w[0, 2]),
                    phase=env.arm_motion.phase.tolist(),
                    arm_tracking_error_rad=float(env.extras['log']['IK/joint_tracking_error_rad'])), flush=True)
            if not args.headless:
                remaining = env.step_dt - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
            step += 1

    from legged_lab.utils.app_lifecycle import run_simulation
    run_simulation(play, app, args.headless)


if __name__ == '__main__':
    main()
