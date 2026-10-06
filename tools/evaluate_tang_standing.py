"""Zero-command foot contact, drift and saturation evaluation before first failure."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'rsl_rl')]
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--task', choices=('walk_elf3_tang_bxi', 'walk_elf3_tang_bxi_scratch'), default='walk_elf3_tang_bxi')
parser.add_argument('--output', required=True)
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--arm-guard', action='store_true')
parser.add_argument('--include-guarded-neutral', action='store_true')
parser.add_argument('--seconds', type=float, default=24.)
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--noise_cap', type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import hashlib
import json
import torch
from legged_lab.envs.elf3_tang.elf3_env import Elf3Env
from legged_lab.envs.elf3_tang.walk_bxi_cfg import Elf3WalkTangBxiEnvCfg, Elf3WalkTangBxiAgentCfg
from legged_lab.envs.elf3_tang import rewards
from rsl_rl.runners import OnPolicyRunner


@torch.inference_mode()
def main():
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg, agent = Elf3WalkTangBxiEnvCfg(), Elf3WalkTangBxiAgentCfg()
    if args.task == 'walk_elf3_tang_bxi_scratch':
        from legged_lab.envs.elf3_tang.scratch_cfg import Elf3ScratchBxiEnvCfg, Elf3ScratchBxiAgentCfg
        cfg, agent = Elf3ScratchBxiEnvCfg(), Elf3ScratchBxiAgentCfg()
    if args.arm_guard:
        cfg.arm_motion.collision_guard = True
        cfg.arm_motion.ramp_s = 1.0
        cfg.arm_motion.amass_max_velocity = None
    cfg.arm_motion.amass_neutral_fraction = 0.
    cfg.arm_motion.amass_scale_end_iteration = None
    cfg.arm_motion.scale = 1.
    cfg.scene.num_envs = args.num_envs
    cfg.scene.seed = args.seed
    cfg.scene.max_episode_length_s = args.seconds + 5
    cfg.noise.add_noise = False
    cfg.domain_rand.events.push_robot = None
    cfg.reset_velocity_curriculum.initial_scale = cfg.reset_velocity_curriculum.final_scale = 0.
    env = Elf3Env(cfg, args.headless)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=env.device)
    runner.load(args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy()
    env.command_generator.set_manual_command([0., 0., 0., cfg.commands.nominal_height, 0.])
    ids = torch.arange(env.num_envs, device=env.device)
    capture = {}
    original_reset_check = env.check_reset
    def check_reset():
        done, timeout = original_reset_check()
        data = env.robot.data
        capture.update(xy=data.body_pos_w[:, env.pelvis_body_id, :2].clone(),
                       height=rewards.pelvis_height(env).clone(),
                       speed=data.body_lin_vel_w[:, env.pelvis_body_id, :2].norm(dim=-1),
                       yaw_speed=data.body_ang_vel_w[:, env.pelvis_body_id, 2].abs(),
                       foot_speed=data.body_lin_vel_w[:, env.feet_body_ids].norm(dim=-1).mean(-1),
                       contacts=env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids].clone(),
                       raw=env.raw_actions.clone(), bounded=env.bounded_actions.clone(),
                       terminated=done & ~timeout)
        return done, timeout
    env.check_reset = check_reset
    result = {'checkpoint': str(Path(args.checkpoint).resolve()), 'iteration': runner.current_learning_iteration,
              'seed': args.seed, 'num_envs': env.num_envs, 'seconds': args.seconds, 'warmup_s': 4.,
              'stochastic': args.stochastic, 'noise_cap': args.noise_cap,
              'arm_guard_enabled': cfg.arm_motion.collision_guard, 'task': args.task,
              'amass_scale': 1., 'neutral_reference_fraction_in_amass_profile': 0.,
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'profiles': [], 'policy_joint_names': [env.robot.data.joint_names[i] for i in env.policy_joint_ids]}
    profiles = [('amass', False, False), ('arms_fixed', True, False)]
    if args.include_guarded_neutral:
        if not cfg.arm_motion.collision_guard:
            raise ValueError('Guarded-neutral evaluation requires collision_guard')
        profiles.append(('neutral_guarded', True, True))
    for name, fixed, guard_fixed in profiles:
        cfg.arm_motion.guard_fixed_targets = guard_fixed
        env.arm_targets = env.robot.data.default_joint_pos[:, env.arm_joint_ids].clone() if fixed else None
        env.seed(args.seed)
        env.reset(ids)
        obs, _ = env.get_observations()
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        anchor = torch.zeros(env.num_envs, 2, device=env.device)
        max_drift = torch.zeros(env.num_envs, device=env.device)
        end_drift = torch.zeros_like(max_drift)
        totals = torch.zeros(8, device=env.device)
        clipped = torch.zeros(env.num_actions, device=env.device)
        raw_abs = torch.zeros(env.num_actions, device=env.device)
        for step in range(round(args.seconds / env.step_dt)):
            action = policy(obs)
            if args.stochastic:
                std = runner.alg.policy.std
                if args.noise_cap is not None:
                    std = std.clamp(max=args.noise_cap)
                action = action + torch.randn_like(action) * std
            obs, _, _, _ = env.step(action)
            if step == 199:
                anchor.copy_(capture['xy'])
            if step >= 200:
                drift = (capture['xy'] - anchor).norm(dim=-1)
                end_drift[alive] = drift[alive]
                max_drift = torch.maximum(max_drift, torch.where(alive, drift, 0.))
                contact = capture['contacts'] > 0
                values = [(contact.sum(-1) == 2).float(), (contact.sum(-1) == 1).float(),
                          (capture['contacts'].amin(-1) > .5).float(), capture['speed'],
                          capture['yaw_speed'], capture['foot_speed'],
                          (capture['height'] - cfg.commands.nominal_height).abs()]
                for index, value in enumerate(values):
                    totals[index] += value[alive].sum()
                totals[-1] += alive.sum()
                clipped += (capture['raw'][alive] != capture['bounded'][alive]).float().sum(0)
                raw_abs += capture['raw'][alive].abs().sum(0)
            alive &= ~capture['terminated']
        denom = totals[-1].clamp(min=1)
        row = dict(zip(['double_support_fraction', 'single_support_fraction', 'sustained_double_support_fraction',
                        'speed_m_s', 'yaw_speed_rad_s', 'foot_speed_m_s', 'height_mae_m'], (totals[:-1]/denom).tolist()))
        row.update(profile=name, survived=int(alive.sum()),
                   drift_survivors_mean_m=float(end_drift[alive].mean()) if alive.any() else None,
                   max_drift_survivors_mean_m=float(max_drift[alive].mean()) if alive.any() else None,
                   drift_survivors_median_m=float(end_drift[alive].median()) if alive.any() else None,
                   drift_survivors_p95_m=float(torch.quantile(end_drift[alive], .95)) if alive.any() else None,
                   clipped_fraction_by_joint=(clipped/denom).tolist(), raw_abs_mean_by_joint=(raw_abs/denom).tolist())
        result['profiles'].append(row)
        print('STANDING_RESULT', json.dumps(row), flush=True)
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


from legged_lab.utils.app_lifecycle import run_simulation
run_simulation(main, launcher.app, args.headless)
