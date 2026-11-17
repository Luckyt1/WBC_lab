# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "rsl_rl")]

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument('--from_scratch', action='store_true', help='Require fresh policy/optimizer and save initial-state evidence.')
parser.add_argument('--scratch_stage', default=None,
                    help='Explicit scratch stage; no promotion based on iteration count.')
parser.add_argument('--stage_report', default=None,
                    help='Frozen walk/report.json for the resumed checkpoint, required when promoting a scratch stage.')
parser.add_argument('--resume_checkpoint', type=Path, default=None,
                    help='Exact checkpoint path; requires --resume and bypasses run-name lookup.')
parser.add_argument('--gait_ablation', choices=('control', 'slide', 'air_time', 'swing_quality', 'mixed_yaw'), default=None,
                    help='One-factor continuation of the standing-refinement task only.')
parser.add_argument('--save_interval', type=int, default=None, help='Override checkpoint interval in updates.')
parser.add_argument('--auxiliary_supervision', choices=('control', 'state', 'dynamics'), default=None,
                    help='Actor-state supervision: retained still-task checkpoint or robust-task training.')
parser.add_argument('--resume_auxiliary_training', action='store_true',
                    help='Restore all learned state for dynamics or plain-PPO control at the next iteration.')
parser.add_argument('--standing_position_penalty', action='store_true',
                    help='Add distance-from-stop penalty to the robust or still task (new reward recipe).')
parser.add_argument('--teleop_standing', action='store_true',
                    help='Robust teleop command mixture (default 80% standing); enable position penalty.')
parser.add_argument('--teleop_standing_fraction', type=float, default=.8,
                    help='Standing probability; remaining commands split forward/back:lateral:turning = 2:1:1.')
parser.add_argument('--pelvis_height', type=float, default=None,
                    help='Explicit fixed-height recipe; permit only this height change during audited dynamics resume.')
parser.add_argument('--teleop_curriculum_stage', default=None,
                    help='Fixed performance-gated teleop difficulty; fresh runs start at quiet.')
parser.add_argument('--teleop_curriculum_decision', type=Path, default=None,
                    help='Audited frozen evaluation decision for an adjacent stage change.')
parser.add_argument('--elf3_symmetry', action='store_true',
                    help='BXI mirror augmentation and actor consistency, including dynamics labels.')
parser.add_argument('--liftoff_rewards', action='store_true',
                    help='B+C+D: standing-only CoM midpoint, lateral knee width and moving load transfer.')
parser.add_argument('--ik_arm_bank', type=Path, help='Procedural deployment-IK reach/hold/return training bank.')
parser.add_argument('--disable_symmetry_on_resume', action='store_true',
                    help='Explicitly remove symmetry while auditing all other continued training settings.')

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.ik_arm_bank and (args_cli.task != 'walk_elf3_tang_bxi_robust' or not (args_cli.from_scratch or args_cli.resume_auxiliary_training)
                            or args_cli.teleop_curriculum_stage is not None):
    parser.error('--ik_arm_bank requires fresh or audited resumed robust training without gated curriculum')
if not 0. <= args_cli.teleop_standing_fraction <= 1.:
    parser.error('--teleop_standing_fraction must be between 0 and 1')
if args_cli.teleop_standing_fraction != .8 and not args_cli.teleop_standing:
    parser.error('--teleop_standing_fraction requires --teleop_standing')
if args_cli.disable_symmetry_on_resume and (args_cli.elf3_symmetry or not args_cli.resume_auxiliary_training):
    parser.error('--disable_symmetry_on_resume requires auxiliary continuation without --elf3_symmetry')
if args_cli.elf3_symmetry and args_cli.task != 'walk_elf3_tang_bxi_robust':
    parser.error('--elf3_symmetry requires walk_elf3_tang_bxi_robust')
if args_cli.liftoff_rewards and args_cli.task != 'walk_elf3_tang_bxi_robust':
    parser.error('--liftoff_rewards requires walk_elf3_tang_bxi_robust')
if args_cli.teleop_curriculum_stage is not None:
    from tools.teleop_curriculum import STAGES
    if (args_cli.teleop_curriculum_stage not in STAGES or not args_cli.teleop_standing
            or args_cli.auxiliary_supervision not in ('dynamics', 'control') or args_cli.pelvis_height is None):
        parser.error('Gated curriculum requires a valid stage, teleop_standing, dynamics/control and fixed pelvis height')
    if args_cli.from_scratch and args_cli.teleop_curriculum_stage != STAGES[0]:
        parser.error('Fresh gated curriculum must start at quiet')
    if not (args_cli.from_scratch or args_cli.resume_auxiliary_training):
        parser.error('Gated curriculum requires an explicit fresh start or audited resume')
elif args_cli.teleop_curriculum_decision is not None:
    parser.error('--teleop_curriculum_decision requires --teleop_curriculum_stage')
if args_cli.pelvis_height is not None:
    from tools.fixed_pelvis_height import validate_height
    validate_height(args_cli.pelvis_height)
    if args_cli.task != 'walk_elf3_tang_bxi_robust':
        parser.error('--pelvis_height requires walk_elf3_tang_bxi_robust')
if args_cli.teleop_standing and args_cli.task != 'walk_elf3_tang_bxi_robust':
    parser.error('--teleop_standing requires walk_elf3_tang_bxi_robust')
if args_cli.standing_position_penalty and args_cli.task not in ('walk_elf3_tang_bxi_robust', 'walk_elf3_tang_bxi_still'):
    parser.error('--standing_position_penalty requires the robust or still task')
if args_cli.auxiliary_supervision is not None:
    auxiliary_resume = args_cli.task == 'walk_elf3_tang_bxi_still' and args_cli.resume_checkpoint is not None
    auxiliary_fresh = args_cli.task == 'walk_elf3_tang_bxi_robust' and args_cli.from_scratch
    auxiliary_continuation = (args_cli.task == 'walk_elf3_tang_bxi_robust'
                              and args_cli.resume_auxiliary_training)
    if not (auxiliary_resume or auxiliary_fresh or auxiliary_continuation) or args_cli.gait_ablation is not None:
        parser.error('--auxiliary_supervision requires a still checkpoint, fresh robust task, or explicit dynamics continuation')
if args_cli.resume_auxiliary_training and (args_cli.task != 'walk_elf3_tang_bxi_robust'
        or args_cli.auxiliary_supervision not in ('dynamics', 'control') or args_cli.resume_checkpoint is None
        or not args_cli.resume or args_cli.from_scratch):
    parser.error('--resume_auxiliary_training requires dynamics/control, robust task and an exact resumed checkpoint')
if args_cli.from_scratch and (args_cli.resume or args_cli.load_run is not None or args_cli.checkpoint is not None
                             or args_cli.resume_checkpoint is not None):
    parser.error('--from_scratch cannot be combined with resume, load_run or checkpoint')
if args_cli.resume_checkpoint is not None and (not args_cli.resume or args_cli.load_run is not None
                                             or args_cli.checkpoint is not None):
    parser.error('--resume_checkpoint requires --resume and cannot be combined with run/checkpoint lookup')
if args_cli.save_interval is not None and args_cli.save_interval <= 0:
    parser.error('--save_interval must be positive')
if args_cli.gait_ablation is not None and (args_cli.task != 'walk_elf3_tang_bxi_still'
                                         or args_cli.resume_checkpoint is None):
    parser.error('--gait_ablation requires walk_elf3_tang_bxi_still and an exact resume checkpoint')
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner | AmpOnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.from_scratch:
        agent_cfg.resume = False
    env_cfg.scene.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join('logs', agent_cfg.experiment_name))
    if args_cli.resume_checkpoint is not None:
        resume_path = str(args_cli.resume_checkpoint.resolve(strict=True))
    else:
        resume_path = (get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
                       if agent_cfg.resume else None)
    ablation = None
    if args_cli.gait_ablation is not None:
        from legged_lab.envs.elf3_tang.gait_ablation import configure_gait_ablation
        ablation = configure_gait_ablation(env_cfg, args_cli.gait_ablation)
    position_penalty = None
    teleop_recipe = None
    liftoff_recipe = None
    if args_cli.liftoff_rewards:
        from legged_lab.envs.elf3_tang.liftoff_rewards import configure_liftoff_rewards
        liftoff_recipe = configure_liftoff_rewards(env_cfg)
    if args_cli.teleop_standing:
        from legged_lab.envs.elf3_tang.teleop_standing import configure_teleop_standing
        teleop_recipe = configure_teleop_standing(env_cfg, args_cli.teleop_standing_fraction)
    if args_cli.standing_position_penalty or args_cli.teleop_standing:
        from legged_lab.envs.elf3_tang.standing_position import configure_standing_position_penalty
        position_penalty = configure_standing_position_penalty(env_cfg)
    if args_cli.pelvis_height is not None:
        from tools.fixed_pelvis_height import configure_fixed_height
        configure_fixed_height(env_cfg, args_cli.pelvis_height)
        if teleop_recipe is not None:
            teleop_recipe['pelvis_height'] = args_cli.pelvis_height
    if args_cli.auxiliary_supervision is not None:
        env_cfg.auxiliary_supervision = args_cli.auxiliary_supervision != 'control'
        if args_cli.auxiliary_supervision != 'control':
            agent_cfg.algorithm.auxiliary_cfg = dict(mode=args_cli.auxiliary_supervision,
                feature_dim=128, joint_dim=15, action_dim=29, warmup_updates=50, ramp_updates=100,
                seed=7341, velocity_scale=[.5,.5,.3], joint_velocity_scale=3., angular_velocity_scale=1.)
        else:
            agent_cfg.algorithm.auxiliary_cfg = None
    if args_cli.elf3_symmetry:
        from legged_lab.envs.elf3_tang.symmetry import symmetry_config
        agent_cfg.algorithm.symmetry_cfg = symmetry_config()
    if args_cli.disable_symmetry_on_resume:
        agent_cfg.algorithm.symmetry_cfg = None
    if args_cli.teleop_curriculum_stage is not None:
        from tools.teleop_curriculum import configure, checkpoint_stage, settings
        if resume_path and checkpoint_stage(resume_path) != args_cli.teleop_curriculum_stage:
            from tools.teleop_curriculum_gates import validate_transition
            if args_cli.teleop_curriculum_decision is None:
                raise ValueError('Difficulty changes require a frozen evaluation decision')
            validate_transition(args_cli.teleop_curriculum_decision, resume_path, args_cli.teleop_curriculum_stage)
        configure(env_cfg, args_cli.teleop_curriculum_stage)
        teleop_recipe['curriculum_stage'] = args_cli.teleop_curriculum_stage
        teleop_recipe['difficulty'] = settings(args_cli.teleop_curriculum_stage)
    if env_class_name == 'walk_elf3_tang_bxi_scratch':
        import hashlib
        import json
        import yaml
        from legged_lab.envs.elf3_tang.scratch_curriculum import configure_scratch_stage
        from tools.scratch_training_gates import validate_promotion
        previous_stage = 'locomotion'
        if resume_path:
            saved_config = Path(resume_path).parent / 'params/env.yaml'
            if saved_config.is_file():
                previous_stage = yaml.load(saved_config.read_text(), Loader=yaml.BaseLoader).get('scratch_stage', 'locomotion')
        stage = args_cli.scratch_stage or previous_stage
        if stage != previous_stage:
            if not resume_path or not args_cli.stage_report:
                raise ValueError('Stage promotion requires a resumed checkpoint and --stage_report')
            validate_promotion(json.loads(Path(args_cli.stage_report).read_text()), previous_stage, stage,
                               hashlib.sha256(Path(resume_path).read_bytes()).hexdigest())
        configure_scratch_stage(env_cfg, stage)
    elif args_cli.scratch_stage is not None or args_cli.stage_report is not None:
        raise ValueError('Scratch stage options require walk_elf3_tang_bxi_scratch')

    ik_recipe = None
    if args_cli.ik_arm_bank:
        from legged_lab.envs.elf3_tang.ik_training import configure_ik_training
        ik_recipe = configure_ik_training(env_cfg, args_cli.ik_arm_bank)

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    if env_class_name == 'walk_gl165_wbc' and resume_path is not None:
        from legged_lab.envs.gl165.checkpoint import restore_checkpoint_config

        restore_checkpoint_config(env_cfg, resume_path)
    env = env_class(env_cfg, args_cli.headless)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    if ik_recipe is not None:
        import json
        os.makedirs(log_dir, exist_ok=True)
        (Path(log_dir)/'ik_training.json').write_text(json.dumps(ik_recipe, indent=2)+'\n')
        print('[IK_RECIPE]', json.dumps(ik_recipe), flush=True)
    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    if args_cli.elf3_symmetry:
        from legged_lab.envs.elf3_tang.symmetry import validate_environment
        validate_environment(env, agent_cfg.empirical_normalization)
        print('[ELF3_SYMMETRY] validated live joint/observation maps and mirrored dynamics labels', flush=True)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        # get path to previous checkpoint
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if liftoff_recipe is not None:
        import json
        with open(os.path.join(log_dir, 'liftoff_rewards.json'), 'w') as stream:
            json.dump(liftoff_recipe, stream, indent=2)
        print('[LIFTOFF_REWARDS]', json.dumps(liftoff_recipe), flush=True)
    if position_penalty is not None:
        import json
        with open(os.path.join(log_dir, 'standing_position_penalty.json'), 'w') as stream:
            json.dump(position_penalty, stream, indent=2)
        print('[STANDING_POSITION_PENALTY]', json.dumps(position_penalty), flush=True)
    if teleop_recipe is not None:
        import json
        with open(os.path.join(log_dir, 'teleop_standing.json'), 'w') as stream:
            json.dump(teleop_recipe, stream, indent=2)
        print('[TELEOP_STANDING]', json.dumps(teleop_recipe), flush=True)
    if args_cli.resume_auxiliary_training:
        import json
        from tools.audit_auxiliary_resume import record_resume
        evidence = record_resume(runner, resume_path, log_dir, target_height=args_cli.pelvis_height,
                                 curriculum_stage=args_cli.teleop_curriculum_stage,
                                 disable_symmetry=args_cli.disable_symmetry_on_resume,
                                 mode=args_cli.auxiliary_supervision,ik_bank=args_cli.ik_arm_bank)
        print('[AUXILIARY_RESUME]', json.dumps(evidence), flush=True)
    if args_cli.auxiliary_supervision is not None:
        import hashlib
        import json
        import yaml
        metadata = dict(variant=args_cli.auxiliary_supervision, checkpoint=resume_path,
            checkpoint_sha256=hashlib.sha256(Path(resume_path).read_bytes()).hexdigest() if resume_path else None,
            from_scratch=args_cli.from_scratch, task=env_class_name,
            iteration_before_learning=runner.current_learning_iteration,
            optimizer_state_entries=len(runner.alg.optimizer.state), num_envs=env.num_envs, seed=agent_cfg.seed,
            actor_observation_dim=env.get_observations()[0].shape[-1], planned_updates=agent_cfg.max_iterations,
            auxiliary_cfg=getattr(agent_cfg.algorithm,'auxiliary_cfg',None),
            reward_sha256=hashlib.sha256(json.dumps(yaml.load((Path(log_dir)/'params/env.yaml').read_text(),
                Loader=yaml.BaseLoader)['reward'],sort_keys=True).encode()).hexdigest(),
            command_sha256=hashlib.sha256(json.dumps(yaml.load((Path(log_dir)/'params/env.yaml').read_text(),
                Loader=yaml.BaseLoader)['commands'],sort_keys=True).encode()).hexdigest(),
            labels=(None if args_cli.auxiliary_supervision == 'control' else
                    'Current body-frame velocity/contact; action-conditioned next body omega and policy joint velocity; pre-reset, done masked'))
        if args_cli.from_scratch:
            env_recipe = yaml.load((Path(log_dir)/'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
            algorithm_recipe = dict(agent_cfg.algorithm.to_dict())
            algorithm_recipe.pop('auxiliary_cfg', None)
            metadata['environment_sha256'] = hashlib.sha256(json.dumps(env_recipe, sort_keys=True).encode()).hexdigest()
            metadata['algorithm_sha256'] = hashlib.sha256(json.dumps(algorithm_recipe, sort_keys=True).encode()).hexdigest()
        with open(os.path.join(log_dir,'auxiliary_supervision.json'),'w') as stream:
            json.dump(metadata,stream,indent=2)
        print('[AUXILIARY_SUPERVISION]',json.dumps(metadata),flush=True)
    if ablation is not None:
        import hashlib
        import json
        metadata = dict(**ablation, checkpoint=resume_path,
                        checkpoint_sha256=hashlib.sha256(Path(resume_path).read_bytes()).hexdigest(),
                        iteration_before_learning=runner.current_learning_iteration,
                        optimizer_state_entries=len(runner.alg.optimizer.state),
                        num_envs=env.num_envs, seed=agent_cfg.seed,
                        actor_observation_dim=env.get_observations()[0].shape[-1],
                        planned_updates=agent_cfg.max_iterations, save_interval=agent_cfg.save_interval)
        with open(os.path.join(log_dir, 'gait_ablation.json'), 'w') as stream:
            json.dump(metadata, stream, indent=2)
        print('[GAIT_ABLATION]', json.dumps(metadata), flush=True)
    if args_cli.from_scratch:
        import json
        assert runner.current_learning_iteration == 0
        assert len(runner.alg.optimizer.state) == 0
        initial = dict(model_state_dict=runner.alg.policy.state_dict(),
                       optimizer_state_dict=runner.alg.optimizer.state_dict(), iter=0,
                       infos={'from_scratch': True, 'before_first_update': True})
        if getattr(runner.alg, 'auxiliary', None) is not None:
            initial.update(auxiliary_state_dict=runner.alg.auxiliary.state_dict(),
                           auxiliary_optimizer_state_dict=runner.alg.auxiliary_optimizer.state_dict(),
                           auxiliary_cfg=runner.alg.auxiliary_cfg)
        torch.save(initial, os.path.join(log_dir, 'initial_policy.pt'))
        metadata = dict(from_scratch=True, checkpoint_loaded=None, teacher_checkpoint=None,
                        iteration_before_learning=0, optimizer_state_entries=0,
                        num_envs=env.num_envs, seed=agent_cfg.seed,
                        actor_observation_dim=env.get_observations()[0].shape[-1],
                        scratch_stage=getattr(env_cfg, 'scratch_stage', None),
                        initial_checkpoint=os.path.join(log_dir, 'initial_policy.pt'))
        with open(os.path.join(log_dir, 'fresh_start.json'), 'w') as stream:
            json.dump(metadata, stream, indent=2)
        print('[FRESH_START]', json.dumps(metadata), flush=True)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                     init_at_random_ep_len=not (ik_recipe and ik_recipe['recipe'] in ('reach_hold_return_v2','reach_hold_return_v3')))
    finally:
        if runner.writer is not None:
            runner.writer.close()


if __name__ == "__main__":
    from legged_lab.utils.app_lifecycle import run_simulation

    run_simulation(train, simulation_app, args_cli.headless)
