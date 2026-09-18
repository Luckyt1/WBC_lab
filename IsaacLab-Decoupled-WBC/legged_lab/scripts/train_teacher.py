# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import argparse

from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--save_interval", type=int, default=1000, help="Save a checkpoint every N iterations.")
parser.add_argument("--training_diagnostics", action="store_true", help="Record rollout anomalies and PPO update diagnostics.")
parser.add_argument("--diagnostics_max_events", type=int, default=50, help="Maximum detailed anomaly records per process.")
parser.add_argument("--diagnostics_cooldown_iterations", type=int, default=100, help="Minimum iterations between anomaly records.")
parser.add_argument("--diagnostics_rolling", action="store_true", help="Retain recent and severe incidents by category, without a lifetime capture cap.")
parser.add_argument("--initialize_from_checkpoint", help="Initialize a NEW auto-resume run from this exact checkpoint, including optimizer and iteration. Later invocations resume the new run's own checkpoints.")
parser.add_argument(
    "--auto_resume",
    action="store_true",
    help=(
        "Use a deterministic per-config log folder named after --run_name and "
        "automatically resume from its latest checkpoint if one exists. Intended "
        "for preemptible cluster jobs: a requeued job continues its own run. With "
        "this flag --max_iterations is treated as an absolute target."
    ),
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import glob
import os
import re
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import apply_cfg_overrides, update_rsl_rl_cfg, validate_trained_robot_layout

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _latest_checkpoint(log_dir: str) -> str | None:
    """Return the highest-iteration ``model_<N>.pt`` in ``log_dir``, or None."""
    if not os.path.isdir(log_dir):
        return None
    ckpts = glob.glob(os.path.join(log_dir, "model_*.pt"))

    def _iter(path: str) -> int:
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(ckpts, key=_iter) if ckpts else None


def train():
    runner: OnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.save_interval = args_cli.save_interval
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    # apply generic --set KEY=VALUE overrides (sweeps over curriculum, gains, etc.)
    apply_cfg_overrides(env_cfg, agent_cfg, args_cli.overrides)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    resume_path = None
    initialize_branch = False
    if args_cli.initialize_from_checkpoint and not args_cli.auto_resume:
        raise ValueError('--initialize_from_checkpoint requires --auto_resume and a distinct --run_name')
    if args_cli.auto_resume:
        # Preemption-friendly mode: a deterministic per-config folder (named after
        # run_name) so a requeued job lands in the same place and continues from
        # its own latest checkpoint. First launch starts fresh; requeues resume.
        if not agent_cfg.run_name:
            raise ValueError("--auto_resume requires --run_name to identify the config folder.")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name)
        resume_path = _latest_checkpoint(log_dir)
        if resume_path is None and args_cli.initialize_from_checkpoint:
            import json
            from pathlib import Path
            source = Path(args_cli.initialize_from_checkpoint).resolve(strict=True)
            destination = Path(log_dir).resolve()
            if destination == source.parent:
                raise ValueError('A recovery branch must use a distinct log directory')
            origin = destination / 'checkpoint_origin.json'
            expected = {'source_checkpoint': str(source), 'task': args_cli.task}
            if origin.exists() and json.loads(origin.read_text()) != expected:
                raise ValueError('Existing branch origin does not match requested initialization')
            if not origin.exists():
                existing = list(destination.iterdir()) if destination.exists() else []
                # The launcher owns only a lock and console-session directory.
                if any(p.name != '.train.lock' and not p.name.startswith('session_') for p in existing):
                    raise ValueError('Refusing to initialize an existing unmarked experiment')
                destination.mkdir(parents=True, exist_ok=True)
                origin.write_text(json.dumps(expected, indent=2) + '\n')
            resume_path, initialize_branch = str(source), True
        agent_cfg.resume = resume_path is not None
        if resume_path:
            print(f"[INFO]: Auto-resume: continuing from {resume_path}")
        else:
            print(f"[INFO]: Auto-resume: no checkpoint in {log_dir}; starting fresh.")
    elif agent_cfg.resume:
        # Resume into the SAME run folder so checkpoints, tensorboard events and
        # the wandb run all continue the original run instead of starting fresh.
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
        print(f"[INFO]: Resuming run in directory: {log_dir}")
    else:
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    if resume_path:
        validate_trained_robot_layout(env_cfg, os.path.dirname(resume_path))
    env = env_class(env_cfg, args_cli.headless)

    # Make wandb append to the same run on resume: use a deterministic run id
    # derived from the run-folder name. WANDB_RESUME=allow creates a new run for
    # fresh training and resumes the existing run when that id already exists.
    if agent_cfg.logger == "wandb":
        os.environ["WANDB_RUN_ID"] = os.path.basename(log_dir)
        os.environ["WANDB_RESUME"] = "allow"
        # On resume rsl-rl re-uploads the config via wandb.config.update() WITHOUT
        # allow_val_change. If the re-derived config differs from what the run
        # already stored (e.g. a different Isaac Lab version serializes the config
        # slightly differently), wandb raises "Attempted to change value of key
        # ...". Default allow_val_change=True so the resumed run overwrites instead.
        try:
            from wandb.sdk.wandb_config import Config as _WandbConfig

            _orig_cfg_update = _WandbConfig.update

            def _cfg_update(self, d=None, allow_val_change=True, _orig=_orig_cfg_update):
                return _orig(self, d, allow_val_change=allow_val_change)

            _WandbConfig.update = _cfg_update
        except Exception:
            pass

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model (restores weights, optimizer and iteration)
        runner.load(resume_path)
        # Keep the adaptive learning rate continuous: load() restores the optimizer
        # state (incl. the adapted lr in its param groups) but not alg.learning_rate,
        # which otherwise resets to the config default and re-adapts -- causing a
        # transient bump right after resume. Seed it from the restored optimizer.
        try:
            runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
            print(f"[INFO]: Restored learning rate: {runner.alg.learning_rate}")
        except (AttributeError, IndexError, KeyError):
            pass
        # continue the command curriculum from the resumed iteration
        env.set_curriculum_start_iteration(runner.current_learning_iteration)

    if getattr(agent_cfg, 'ppo_stability', None):
        from legged_lab.utils.ppo_stability import install_ppo_stability
        install_ppo_stability(runner.alg, agent_cfg.ppo_stability, reset_learning_rate=initialize_branch)
        print(f'[INFO]: PPO stability active: {agent_cfg.ppo_stability}; effective lr={runner.alg.learning_rate}')
        # The environment was constructed before load(). Start its very first
        # rollout with resumed posture ranges as well as the resumed iteration.
        env.command_generator.update_curriculum(runner.current_learning_iteration)
        env.command_generator._resample_command(torch.arange(env.num_envs, device=env.device))

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    if args_cli.auto_resume:
        # Preemption mode: max_iterations is an ABSOLUTE target. rsl-rl runs
        # [start_iter, start_iter + num_learning_iterations), so pass the remaining
        # count -> a requeued job converges to max_iterations total instead of
        # adding a full budget on every restart.
        num_iterations = max(agent_cfg.max_iterations - runner.current_learning_iteration, 0)
        print(
            f"[INFO]: Auto-resume target {agent_cfg.max_iterations} "
            f"(starting at {runner.current_learning_iteration}, {num_iterations} remaining)."
        )
    else:
        # Fresh run or manual --resume: original semantics -- max_iterations is the
        # number of iterations to run in THIS invocation (additive when resuming).
        num_iterations = agent_cfg.max_iterations
    diagnostics = None
    try:
        if args_cli.training_diagnostics:
            from legged_lab.utils.training_diagnostics import DiagnosticsConfig, TrainingDiagnostics
            diagnostics = TrainingDiagnostics(runner, DiagnosticsConfig(
                max_events=args_cli.diagnostics_max_events,
                cooldown_iterations=args_cli.diagnostics_cooldown_iterations,
                rolling_retention=args_cli.diagnostics_rolling,
                kl_threshold=.02 if args_cli.diagnostics_rolling else .1,
            ))
        runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=True)
    finally:
        if diagnostics is not None:
            diagnostics.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        train()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        exit_code = 130
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"Error: {exc}")
        exit_code = 1
    finally:
        # simulation_app.close() can hang on some Isaac Sim setups; force exit so
        # the process terminates cleanly. os._exit() skips stdio flushing, so flush
        # explicitly first or trailing console output (logs) would be lost. Preserve a
        # NONZERO code on failure so Slurm reports FAILED instead of masking a crash as
        # COMPLETED (the SIGUSR1 self-requeue path exits 0 separately, as intended).
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
