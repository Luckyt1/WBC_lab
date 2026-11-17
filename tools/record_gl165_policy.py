"""Record an actual GL165 policy rollout in Isaac Sim, with command/state captions."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]
SEGMENTS = [
    (8., '站立：观察腰部姿态和手臂扰动', (0., 0., 0.)),
    (12., '前进：目标 0.30 m/s，观察是否走直', (.3, 0., 0.)),
    (8., '向左侧移：目标 0.20 m/s', (0., .2, 0.)),
    (8., '向右侧移：目标 0.20 m/s', (0., -.2, 0.)),
    (8., '向左转：目标 0.50 rad/s', (0., 0., .5)),
    (8., '向右转：目标 0.50 rad/s', (0., 0., -.5)),
    (8., '停止：速度指令归零，观察能否停稳', (0., 0., 0.)),
]


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=sum(s[0] for s in SEGMENTS))
    parser.add_argument('--seed', type=int, default=7103)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 0 < args.seconds <= sum(s[0] for s in SEGMENTS):
        parser.error('seconds must be positive and within the demo duration')
    args.enable_cameras = True
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.bank = args.bank.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app

    def record():
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import subprocess
        import torch
        import omni.replicator.core as rep
        import isaaclab.sim as sim_utils
        from isaaclab.utils.math import quat_apply_inverse, yaw_quat
        from legged_lab.envs.gl165.checkpoint import restore_checkpoint_config
        from legged_lab.envs.gl165.evaluation import configure_controlled_evaluation, load_policy_runner
        from legged_lab.envs.gl165.gl165_env import Gl165Env
        from legged_lab.envs.gl165.wbc_cfg import Gl165WbcEnvCfg

        cfg = Gl165WbcEnvCfg()
        restore_checkpoint_config(cfg, args.checkpoint)
        cfg.scene.num_envs, cfg.scene.seed = 1, args.seed
        cfg.scene.max_episode_length_s = args.seconds + 1.
        cfg.device = args.device
        cfg.ik_bank_path, cfg.ik_split = str(args.bank), 'test'
        configure_controlled_evaluation(cfg)
        cfg.commands.rel_standing_envs = 0.
        cfg.commands.resampling_time_range = (1e6, 1e6)
        env = Gl165Env(cfg, headless=True)
        runner = load_policy_runner(env, args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)

        # A darker display material makes the CAD joints visible against the floor.
        material_path = '/World/Looks/VideoRobot'
        robot_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.045, .12, .23), roughness=.65)
        robot_material.func(material_path, robot_material)
        sim_utils.bind_visual_material('/World/envs/env_0/Robot', material_path, stronger_than_descendants=True)

        # These floor lines have no collision and serve only as distance references.
        line_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.32, .35, .39))
        for coordinate in range(-20, 21):
            for axis in (0, 1):
                line = sim_utils.CuboidCfg(size=((.012, 40., .002) if axis == 0 else (40., .012, .002)),
                                          visual_material=line_material)
                position = (float(coordinate), 0., .002) if axis == 0 else (0., float(coordinate), .002)
                line.func(f'/World/VideoGrid/Line_{axis}_{coordinate + 20}', line, translation=position)
        camera_path = '/World/RecordingCamera'
        camera = sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=36., clipping_range=(.1, 200.))
        camera.func(camera_path, camera)
        render_product = rep.create.render_product(camera_path, (1280, 720))
        annotator = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        annotator.attach([render_product])
        env.reset(torch.arange(env.num_envs, device=env.device))
        env.command_generator.vel_command_b.zero_()
        env.command_generator.is_standing_env[:] = False
        obs, _ = env.get_observations()

        def aim_camera():
            root = env.robot.data.root_pos_w[0].detach().cpu().numpy()
            top = float(env.robot.data.body_pos_w[0, :, 2].max()) + .15
            center_z = max(.25, (top + .03) / 2. - .15)
            eye = (float(root[0] + 2.8), float(root[1] - 2.8), center_z + 1.0)
            target = (float(root[0]), float(root[1]), center_z)
            env.sim.set_camera_view(eye=eye, target=target, camera_prim_path=camera_path)

        aim_camera()
        for _ in range(12):
            env.sim.render()
        rgb = annotator.get_data()
        assert rgb.shape[:2] == (720, 1280) and rgb[..., :3].std() > 3, 'Camera did not produce a visible scene'
        Image.fromarray(rgb[..., :3]).save(args.output / 'camera_preview.png')
        print('GL165_VIDEO_CAMERA_READY', str(args.output / 'camera_preview.png'), flush=True)
        font_path = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True)
        fonts = {size: ImageFont.truetype(font_path, size) for size in (20, 24, 30)}
        video_path = args.output / f'gl165_{args.checkpoint.stem}_demo.mp4'
        writer = imageio.get_writer(str(video_path), fps=25, codec='libx264', quality=8,
                                    pixelformat='yuv420p', macro_block_size=1,
                                    ffmpeg_params=['-movflags', '+faststart', '-preset', 'fast'])
        frames, last_segment, failures, samples = 0, -1, [], []
        boundaries = np.cumsum([s[0] for s in SEGMENTS])
        for step in range(round(args.seconds / env.step_dt)):
            elapsed = step * env.step_dt
            segment = min(int(np.searchsorted(boundaries, elapsed, side='right')), len(SEGMENTS) - 1)
            _, label, command = SEGMENTS[segment]
            env.command_generator.vel_command_b[:] = torch.tensor(command, device=env.device)
            if segment != last_segment:
                obs, _ = env.get_observations()
                last_segment = segment
                print('GL165_VIDEO_SEGMENT', elapsed, label, flush=True)
            with torch.inference_mode():
                obs, _, done, _ = env.step(policy(obs))
            if bool(done[0]):
                failures.append(dict(time_s=(step + 1) * env.step_dt, segment=segment, automatic_reset=True))
                env.command_generator.vel_command_b[:] = torch.tensor(command, device=env.device)
                obs, _ = env.get_observations()
            if step % 2:
                aim_camera()
                env.sim.render()
                rgb = annotator.get_data()[..., :3]
                data = env.robot.data
                velocity = quat_apply_inverse(yaw_quat(data.root_quat_w), data.root_lin_vel_w)[0, :2]
                actual = [*velocity.detach().cpu().tolist(), float(data.root_ang_vel_w[0, 2])]
                waist = torch.rad2deg(data.joint_pos[0, env.policy_joint_ids[:3]]).detach().cpu().tolist()
                time_s = (step + 1) * env.step_dt
                sample = dict(time_s=time_s, segment=segment, command=command, velocity=actual,
                              waist_deg=waist, root_pos=data.root_pos_w[0].detach().cpu().tolist(),
                              arm_target_id=env.arm_motion.ids[int(env.arm_motion.index[0])], failures=len(failures))
                samples.append(sample)
                frame = Image.fromarray(rgb.copy())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 104), fill=(15, 20, 29))
                draw.text((20, 8), f'GL165 策略 · 第 {runner.current_learning_iteration} 轮 · Isaac Sim · {time_s:05.1f}s', font=fonts[24], fill='white')
                draw.text((20, 49), label, font=fonts[30], fill=(255, 208, 93))
                draw.rectangle((0, 598, 1280, 720), fill=(15, 20, 29))
                draw.text((20, 604), f'指令  vx {command[0]:+.2f}   vy {command[1]:+.2f} m/s   转向 {command[2]:+.2f} rad/s', font=fonts[24], fill=(180, 210, 255))
                draw.text((20, 640), f'实际  vx {actual[0]:+.2f}   vy {actual[1]:+.2f} m/s   转向 {actual[2]:+.2f} rad/s', font=fonts[24], fill='white')
                draw.text((20, 677), f'腰偏航 {waist[0]:+.1f}°   横滚 {waist[1]:+.1f}°   俯仰 {waist[2]:+.1f}°', font=fonts[20], fill=(255, 174, 145))
                draw.text((840, 650), '连续手臂扰动 · 无额外推扰', font=fonts[20], fill=(195, 202, 213))
                draw.text((840, 681), f'原速 1×  |  失败／自动重置 {len(failures)} 次', font=fonts[20], fill=(195, 202, 213))
                if failures and time_s - failures[-1]['time_s'] < 2.:
                    draw.rectangle((300, 305, 980, 370), fill=(154, 35, 35))
                    draw.text((330, 318), '触发环境失败，已自动重置', font=fonts[30], fill='white')
                writer.append_data(np.asarray(frame))
                if frames == 49:
                    frame.save(args.output / 'caption_preview.png')
                frames += 1
            if step % 250 == 0:
                print('GL165_VIDEO_PROGRESS', elapsed, 'frames=', frames, 'failures=', len(failures), flush=True)
        writer.close()
        (args.output / 'telemetry.json').write_text(json.dumps(samples, ensure_ascii=False) + '\n')
        report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      checkpoint_iteration=runner.current_learning_iteration, bank=str(args.bank), split='test', seed=args.seed,
                      created_at=datetime.now().isoformat(), seconds=frames / 25, fps=25, frames=frames,
                      simulator='Isaac Sim 5.0', playback_speed=1., deterministic_policy=True,
                      random_pushes=False, continuous_arm_motion=True, failures=failures,
                      camera='Follows root xy and body vertical center; world camera orientation is fixed; floor grid spacing is 1 m',
                      visual_material_override='Dark blue for visibility; physics and policy are unchanged',
                      segments=[dict(duration_s=s[0], description=s[1], command=s[2]) for s in SEGMENTS],
                      video=str(video_path.resolve()), recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print('GL165_VIDEO_COMPLETE', json.dumps(report, ensure_ascii=False), flush=True)

    from legged_lab.utils.app_lifecycle import run_simulation
    run_simulation(record, app, args.headless)


if __name__ == '__main__':
    main()
