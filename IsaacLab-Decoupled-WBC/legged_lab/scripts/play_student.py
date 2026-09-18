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
import os
from typing import Sequence

# for logs
from datetime import datetime

import torch
import torch.nn as nn
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained student policy in simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# Student policy play arguments
parser.add_argument("--student_checkpoint", type=str, required=True, help="Path to trained student checkpoint")
parser.add_argument("--max_steps", type=int, default=10000, help="Maximum play steps (default: 10000)")
parser.add_argument("--use_jit", action="store_true", help="Load as JIT model instead of PT checkpoint")
parser.add_argument(
    "--action_log_interval",
    type=int,
    default=100,
    help="Print action diagnostics every N policy steps after warmup. Set 0 to disable.",
)
parser.add_argument(
    "--action_log_warmup_steps",
    type=int,
    default=50,
    help="Print action diagnostics every step for the first N policy steps.",
)
parser.add_argument(
    "--action_log_env",
    type=int,
    default=0,
    help="Environment index to print detailed action diagnostics for.",
)
parser.add_argument(
    "--joint_limit_margin",
    type=float,
    default=0.05,
    help="Joint target distance-to-limit threshold (rad) counted as near-limit.",
)
parser.add_argument(
    "--log_obs",
    action="store_true",
    help="Write detailed observation history logs next to the checkpoint (disabled by default).",
)
parser.add_argument(
    "--continuous_torso",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Hold torso keys (H/J/Z/X/C/V/B/N) to ramp commands continuously. Default: disabled (discrete step per key press).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import apply_trained_actuator_gains, update_rsl_rl_cfg
from legged_lab.utils.student_observations import (
    command_ranges,
    command_limit_tensors,
    strip_student_teacher_contacts,
    STUDENT_LAYOUT_EXTRA_FILE,
    student_layout,
    validate_jit_student_layout_extra_file,
    validate_student_layout,
)


def _action_scale_for_logging(env, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(env.action_scale):
        return env.action_scale[: env.num_actions].to(device=device, dtype=dtype)
    return torch.full((env.num_actions,), float(env.action_scale), device=device, dtype=dtype)


def _log_action_diagnostics(
    env,
    step: int,
    actions: torch.Tensor,
    prev_actions: torch.Tensor | None,
) -> None:
    interval = args_cli.action_log_interval
    if interval <= 0:
        return
    if step >= args_cli.action_log_warmup_steps and step % interval != 0:
        return

    env_idx = min(max(args_cli.action_log_env, 0), actions.shape[0] - 1)
    raw = actions.detach()
    delta = torch.zeros_like(raw) if prev_actions is None else raw - prev_actions
    clipped = torch.clamp(raw, -env.clip_actions, env.clip_actions)
    scale = _action_scale_for_logging(env, raw.device, raw.dtype)
    targets = env.custom_default_joint_pos[:, : env.num_actions] + clipped * scale

    joint_ids = torch.as_tensor(env.custom_joint_ids[: env.num_actions], device=raw.device, dtype=torch.long)
    limits = env.robot.data.soft_joint_pos_limits[:, joint_ids, :]
    lower = limits[:, :, 0]
    upper = limits[:, :, 1]
    margin = torch.minimum(targets - lower, upper - targets)
    violation = torch.relu(lower - targets) + torch.relu(targets - upper)

    env_margin = margin[env_idx]
    worst_idx = int(torch.argmin(env_margin).item())
    joint_id = int(joint_ids[worst_idx].item())
    joint_name = env.robot.data.joint_names[joint_id]
    near_count = int((env_margin < args_cli.joint_limit_margin).sum().item())
    outside_count = int((violation[env_idx] > 0.0).sum().item())

    print(
        "[action_diag] "
        f"step={step} env={env_idx} "
        f"raw_abs_max={raw[env_idx].abs().max().item():.4f} "
        f"raw_abs_mean={raw[env_idx].abs().mean().item():.4f} "
        f"delta_abs_max={delta[env_idx].abs().max().item():.4f} "
        f"delta_abs_mean={delta[env_idx].abs().mean().item():.4f} "
        f"min_limit_margin={env_margin.min().item():.4f} "
        f"near_limit_count={near_count} "
        f"outside_limit_count={outside_count} "
        f"worst_joint={joint_name} "
        f"target={targets[env_idx, worst_idx].item():.4f} "
        f"lower={lower[env_idx, worst_idx].item():.4f} "
        f"upper={upper[env_idx, worst_idx].item():.4f} "
        f"global_raw_abs_max={raw.abs().max().item():.4f} "
        f"global_delta_abs_max={delta.abs().max().item():.4f} "
        f"global_min_margin={margin.min().item():.4f}"
    )


def _should_log_action_step(step: int) -> bool:
    interval = args_cli.action_log_interval
    return interval > 0 and (step < args_cli.action_log_warmup_steps or step % interval == 0)


def _log_joint_torque(env, step: int) -> None:
    if not _should_log_action_step(step):
        return

    env_idx = min(max(args_cli.action_log_env, 0), env.num_envs - 1)
    joint_ids = torch.as_tensor(env.custom_joint_ids[: env.num_actions], device=env.device, dtype=torch.long)
    joint_names = getattr(env, "action_joint_names", None)
    if joint_names is None:
        all_names = env.robot.data.joint_names
        joint_names = [all_names[int(joint_id)] for joint_id in joint_ids.detach().cpu().tolist()]
    torque = env.robot.data.computed_torque.detach()
    action_torque = torque[:, joint_ids]
    env_abs = action_torque[env_idx].abs()
    top_count = min(5, env.num_actions)
    _, top_indices = torch.topk(env_abs, k=top_count)
    top_parts = [
        f"{joint_names[int(idx)]}={action_torque[env_idx, int(idx)].item():+.4f}"
        for idx in top_indices.detach().cpu().tolist()
    ]
    print(
        "[joint_torque] "
        f"step={step} env={env_idx} "
        f"policy_joint_abs_max={env_abs.max().item():.4f} "
        f"global_policy_joint_abs_max={action_torque.abs().max().item():.4f} "
        f"top_policy_torques={','.join(top_parts)}"
    )


class StudentMLP(nn.Module):
    """Simple MLP student policy """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        act_cls = nn.ELU if activation.lower() == "elu" else nn.ReLU

        # Build MLP layers
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(act_cls())
            in_dim = dim

        # Output layer
        layers.append(nn.Linear(in_dim, act_dim))

        self.policy = nn.Sequential(*layers)

        print(f"Student MLP Policy: {self.policy}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            obs: [batch_size, obs_dim] where obs_dim = history_len * base_obs_dim
                 OR [batch_size, history_len, base_obs_dim] (will be flattened)

        Returns:
            actions: [batch_size, act_dim]
        """
        # If 3D input [batch, seq, dim], flatten the history dimension
        # This allows MLP to process entire history at once
        if len(obs.shape) == 3:
            batch_size, seq_len, obs_dim = obs.shape
            obs = obs.reshape(batch_size, seq_len * obs_dim)

        # Apply MLP (no tanh - outputs raw actions like teacher)
        actions = self.policy(obs)  # [batch, act_dim]

        return actions

class ObservationHistoryBuffer:
    """Buffer to store observation history for student policy"""
    def __init__(self, history_length, obs_dim, num_envs, device):
        self.history_length = history_length
        self.obs_dim = obs_dim
        self.num_envs = num_envs
        self.device = device

        # Buffer: [num_envs, history_length, obs_dim]
        # Initially filled with zeros (like action buffer at reset)
        self.buffer = torch.zeros(num_envs, history_length, obs_dim, device=device, dtype=torch.float32)

    def update(self, new_obs):
        """
        Update buffer with new observations
        new_obs: [num_envs, obs_dim]
        Shifts buffer left and adds new observation at the end
        """
        # Shift left: move [1:] to [:-1]
        self.buffer[:, :-1, :] = self.buffer[:, 1:, :].clone()
        # Add new observation at the end
        self.buffer[:, -1, :] = new_obs

    def reset(self, env_ids):
        """Reset specific environments to zeros (called after reset)"""
        if len(env_ids) > 0:
            self.buffer[env_ids] = 0.0

    def get_history_flat(self):
        """Return flattened history [num_envs, history_length * obs_dim]"""
        return self.buffer.reshape(self.num_envs, -1)

    def get_history(self):
        """Return history [num_envs, history_length, obs_dim]"""
        return self.buffer


def _state_dict_model_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int | None, int | None]:
    linear_weights = [
        tensor for key, tensor in state_dict.items()
        if key.endswith(".weight") and getattr(tensor, "ndim", 0) == 2
    ]
    if not linear_weights:
        return None, None
    return int(linear_weights[0].shape[1]), int(linear_weights[-1].shape[0])


def write_observation_history_log(log_file, step, obs_history, actions, action_joint_names, env_idx=0):
    """
    Write detailed observation history and actions to log file.

    Args:
        log_file: File handle to write to
        step: Current simulation step
        obs_history: Tensor of shape [num_envs, history_len, obs_dim] containing observation history
        actions: Tensor of shape [num_envs, num_actions] containing actions
        env_idx: Which environment to log (default: 0)

    Observation structure:
        [0:3]   ang_vel (3)
        [3:6]   projected_gravity (3)
        [6:13]  command (7): lin_vel_x, lin_vel_y, ang_vel_z, height, body_roll, body_pitch, body_yaw
        [13:13+num_actions] joint_pos
        [13+num_actions:13+2*num_actions] joint_vel
        [13+2*num_actions:13+3*num_actions] last_actions
    """
    log_file.write(f"\n{'='*100}\n")
    log_file.write(f"OBSERVATION AT STEP {step} (Environment {env_idx})\n")
    log_file.write(f"{'='*100}\n\n")

    # Get data for the specified environment and convert to CPU numpy for readability
    obs_hist = obs_history[env_idx].cpu().numpy()
    acts = actions[env_idx].cpu().numpy()
    history_len = obs_hist.shape[0]
    num_actions = len(action_joint_names)
    joint_pos_start = 13
    joint_vel_start = joint_pos_start + num_actions
    last_actions_start = joint_vel_start + num_actions

    # Write all history steps
    for hist_idx in range(history_len):
        log_file.write(f"--- HISTORY {hist_idx + 1} ---\n")

        # Angular velocity [0:3]
        ang_vel = obs_hist[hist_idx, 0:3]
        log_file.write(f"  ang_vel          = [{ang_vel[0]:8.4f}, {ang_vel[1]:8.4f}, {ang_vel[2]:8.4f}]\n")

        # Projected gravity [3:6]
        proj_grav = obs_hist[hist_idx, 3:6]
        log_file.write(f"  projected_gravity = [{proj_grav[0]:8.4f}, {proj_grav[1]:8.4f}, {proj_grav[2]:8.4f}]\n")

        # Command [6:13]: lin_vel_x, lin_vel_y, ang_vel_z, height, body_roll, body_pitch, body_yaw
        cmd = obs_hist[hist_idx, 6:13]
        log_file.write(f"  command          = [lin_vel_x:{cmd[0]:7.3f}, lin_vel_y:{cmd[1]:7.3f}, ang_vel_z:{cmd[2]:7.3f}, "
                      f"height:{cmd[3]:6.3f}, body_roll:{cmd[4]:7.3f}, body_pitch:{cmd[5]:7.3f}, body_yaw:{cmd[6]:7.3f}]\n")

        joint_pos = obs_hist[hist_idx, joint_pos_start:joint_vel_start]
        log_file.write(f"  joint_pos        = [")
        for i, name in enumerate(action_joint_names):
            log_file.write(f"{name}:{joint_pos[i]:7.3f}")
            if i < num_actions - 1:
                log_file.write(", ")
        log_file.write("]\n")

        joint_vel = obs_hist[hist_idx, joint_vel_start:last_actions_start]
        log_file.write(f"  joint_vel        = [")
        for i, name in enumerate(action_joint_names):
            log_file.write(f"{name}:{joint_vel[i]:7.3f}")
            if i < num_actions - 1:
                log_file.write(", ")
        log_file.write("]\n")

        last_act = obs_hist[hist_idx, last_actions_start:last_actions_start + num_actions]
        log_file.write(f"  last_actions     = [")
        for i, name in enumerate(action_joint_names):
            log_file.write(f"{name}:{last_act[i]:7.3f}")
            if i < num_actions - 1:
                log_file.write(", ")
        log_file.write("]\n\n")

    # Write current actions
    log_file.write(f"{'='*100}\n")
    log_file.write(f"ACTION AT STEP {step}\n")
    log_file.write(f"{'='*100}\n")
    log_file.write(f"  actions = [")
    for i, name in enumerate(action_joint_names):
        log_file.write(f"{name}:{acts[i]:7.3f}")
        if i < num_actions - 1:
            log_file.write(", ")
    log_file.write("]\n\n")

    log_file.flush()  # Ensure data is written immediately


def play():
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    keyboard_command_ranges = command_ranges(env_cfg)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.action_delay.enable = False  # eval with zero actuation latency by default
    env_cfg.commands.curriculum.enable = False
    env_cfg.commands.rel_high_stance_turn_envs = 0.0
    env_cfg.commands.rel_high_stance_move_turn_envs = 0.0
    env_cfg.scene.max_episode_length_s = 20.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    # Match the actuator gains (kp/kd) the student was distilled under. train_student.py
    # distills in the teacher's env (apply_trained_actuator_gains), but the student
    # checkpoint itself doesn't store the gains -- so resolve the teacher run via
    # --load_run/--checkpoint (same as training) and apply its saved gains here. Without
    # this, eval runs the default gains and the PD closed-loop won't match distillation.
    # Gate on the EXPLICIT CLI arg, not agent_cfg.load_run: the latter defaults to ".*"
    # (never None), which would otherwise silently resolve to the most recent run and
    # apply the wrong gains.
    if args_cli.load_run is not None:
        log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
        teacher_dir = os.path.dirname(
            get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        )
        print(f"[INFO] Applying teacher gains from: {teacher_dir}")
        apply_trained_actuator_gains(env_cfg, teacher_dir)
    else:
        print(
            "[WARN] --load_run not set; playing with DEFAULT actuator gains, which may "
            "not match the gains the student was distilled under. Pass --load_run=<teacher_run> "
            "(and --checkpoint) to load the correct kp/kd."
        )

    # Apply after restoring the teacher configuration: keyboard play supplies
    # its own commands, including for teachers with opt-in low-stance sampling.
    env_cfg.commands.rel_low_stance_move_envs = 0.0
    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard
        keyboard = Keyboard(
            env,
            continuous_torso=args_cli.continuous_torso,
            command_ranges_override=keyboard_command_ranges,
        )

    obs, _ = env.get_observations()

    # Load trained student checkpoint or JIT model
    checkpoint_path = args_cli.student_checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Student checkpoint not found at: {checkpoint_path}")

    history_length = 2  # Number of past observations to use
    layout = student_layout(env, history_length)
    base_obs_dim = layout["base_obs_dim"]
    student_obs_dim = layout["input_dim"]
    student_action_dim = layout["action_dim"]
    action_joint_names = layout["joint_names"]
    student_hidden_dims = [512, 256, 128]
    student_activation = "elu"

    obs_history_buffer = ObservationHistoryBuffer(
        history_length=history_length,
        obs_dim=base_obs_dim,
        num_envs=env_cfg.scene.num_envs,
        device=env.device
    )

    print(f"\n{'='*50}")

    if args_cli.use_jit:
        # Load JIT model directly (same as deploy_student.py)
        print(f"Loading JIT model from: {checkpoint_path}")
        extra_files = {STUDENT_LAYOUT_EXTRA_FILE: ""}
        student_net = torch.jit.load(checkpoint_path, map_location=env.device, _extra_files=extra_files)
        validate_jit_student_layout_extra_file(extra_files[STUDENT_LAYOUT_EXTRA_FILE], layout, checkpoint_path)
        student_net.eval()
        with torch.inference_mode():
            probe = torch.zeros(1, history_length, base_obs_dim, device=env.device)
            probe_actions = student_net(probe)
        if probe_actions.shape[-1] != student_action_dim:
            raise RuntimeError(
                f"JIT student output dimension mismatch in {checkpoint_path}: "
                f"{probe_actions.shape[-1]} != {student_action_dim}"
            )
        print(f"✅ JIT model loaded successfully")
    else:
        # Load PT checkpoint (original behavior)
        print(f"Loading PT checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=env.device)
        validate_student_layout(checkpoint, layout, checkpoint_path)

        config = checkpoint.get("config", {})
        student_hidden_dims = list(config.get("hidden_dims", student_hidden_dims))
        student_activation = config.get("activation", student_activation)

        # Initialize student policy (MLP with history input)
        student_net = StudentMLP(
            obs_dim=student_obs_dim,
            act_dim=student_action_dim,
            hidden_dims=student_hidden_dims,
            activation=student_activation,
        ).to(env.device)

        state_input_dim, state_action_dim = _state_dict_model_dims(checkpoint["model_state_dict"])
        if state_input_dim is not None and state_input_dim != student_obs_dim:
            raise RuntimeError(
                f"Student model input dimension mismatch in {checkpoint_path}: "
                f"{state_input_dim} != {student_obs_dim}"
            )
        if state_action_dim is not None and state_action_dim != student_action_dim:
            raise RuntimeError(
                f"Student model output dimension mismatch in {checkpoint_path}: "
                f"{state_action_dim} != {student_action_dim}"
            )
        student_net.load_state_dict(checkpoint['model_state_dict'])
        student_net.eval()

        print(f"Loaded model from checkpoint step: {checkpoint.get('step', 'N/A')}")
        print(f"Training loss at checkpoint: {checkpoint.get('loss', 'N/A')}")

    print(f"{'='*50}\n")

    # Play configuration
    max_steps = args_cli.max_steps
    current_step = 0
    prev_actions_student = None

    print(f"Student Policy Play Configuration:")
    print(f"  - Input dim: {student_obs_dim} ({history_length} steps x {base_obs_dim} obs)")
    print(f"  - Output dim: {student_action_dim}")
    print(f"  - Action joints: {', '.join(action_joint_names)}")
    print(f"  - Hidden dims: {student_hidden_dims}")
    print(f"  - Activation: {student_activation}")
    print(f"  - Max play steps: {max_steps}")
    print(f"  - Num environments: {env_cfg.scene.num_envs}")
    print(f"  - Mode: Play (no training)\n")

    cmd_min, cmd_max = command_limit_tensors(keyboard_command_ranges, env.device)

    # Optionally write detailed observation history logs (--log_obs to enable)
    log_file = None
    if args_cli.log_obs:
        log_dir = os.path.join(os.path.dirname(checkpoint_path), "observation_logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"obs_history_{timestamp}.log")
        log_file = open(log_path, 'w')
        print(f"Logging observation history to: {log_path}\n")

    while simulation_app.is_running() and current_step < max_steps:

        # Override keyboard commands (only in non-headless mode)
        if not args_cli.headless:
            keyboard.advance()
            with torch.inference_mode():
                cmd = torch.tensor(
                    [
                        keyboard.cmd_x,
                        keyboard.cmd_y,
                        keyboard.cmd_z,
                        keyboard.hight,
                        keyboard.body_roll,
                        keyboard.body_pitch,
                        keyboard.body_yaw,
                    ],
                    device=obs.device,
                    dtype=obs.dtype,
                )
                obs[0][6:13] = torch.clamp(cmd, cmd_min, cmd_max)
            
        # Update observation history buffer with current observation, then read it.
        # History includes the current obs as the most recent entry (matches deployment).
        obs_without_contact = strip_student_teacher_contacts(obs, env.num_actions)
        obs_history_buffer.update(obs_without_contact)
        obs_history = obs_history_buffer.get_history()  # Shape: [num_envs, history_len, base_obs_dim]

        # Get student actions (no gradients - play mode)
        # Student uses PAST observations (history) to predict current action
        with torch.inference_mode():
            actions_student = student_net(obs_history)  # Shape: [num_envs, num_actions]
            # _log_action_diagnostics(env, current_step, actions_student, prev_actions_student)
            prev_actions_student = actions_student.detach().clone()

        if log_file is not None and (current_step == 0 or current_step % 1000 == 1 or current_step == 26):
            write_observation_history_log(log_file, current_step, obs_history, actions_student, action_joint_names, env_idx=0)
            print(f"[Step {current_step}] Logged observation history to file")

        # Increment step counter
        policy_step = current_step
        current_step += 1

        # Print progress every 1000 steps
        if current_step % 1000 == 0:
            print(f"[Step {current_step}/{max_steps}] Play in progress...")
            
        # Step environment with student actions
        with torch.inference_mode():
            # print("actions_student:", actions_student)
            obs, _, reset_buf, _ = env.step(actions_student)
            _log_joint_torque(env, policy_step)

            # Reset history buffer for environments that were reset
            reset_env_ids = reset_buf.nonzero(as_tuple=False).flatten()
            # print("reset_env_ids:", reset_env_ids)
            if len(reset_env_ids) > 0:
                obs_history_buffer.reset(reset_env_ids)


    if log_file is not None:
        log_file.close()

    print(f"\n{'='*50}")
    print(f"Play completed! Total steps: {current_step}")
    if log_file is not None:
        print(f"Observation history log saved to: {log_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import sys
    import traceback

    exit_code = 0
    try:
        play()
    except KeyboardInterrupt:
        print("Play interrupted by user.")
        exit_code = 130
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
