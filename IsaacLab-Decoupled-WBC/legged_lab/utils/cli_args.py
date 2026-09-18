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

from __future__ import annotations

import argparse
import ast
import os
import random
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseAgentConfig


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume training from a checkpoint (selected via --load_run / --checkpoint).",
    )
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
    )
    arg_group.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override a config field by dotted path, e.g. "
            "--set commands.curriculum.body_roll.start_iter=20000. "
            "Targets env_cfg by default; prefix with 'agent.' for the agent cfg. "
            "Repeatable. VALUE is parsed as a Python literal (int/float/tuple/dict/bool), "
            "falling back to a string."
        ),
    )


def _resolve_parent(root, path_parts):
    """Walk a dotted path over objects (attributes) and dicts (keys)."""
    obj = root
    for part in path_parts:
        if isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    return obj


def _set_leaf(parent, key, value):
    if isinstance(parent, dict):
        parent[key] = value
    else:
        setattr(parent, key, value)


def apply_cfg_overrides(env_cfg, agent_cfg, overrides):
    """Apply ``KEY=VALUE`` overrides to env_cfg (default) or agent_cfg.

    Targets env_cfg unless the key is prefixed with ``env.`` or ``agent.``.
    The value is parsed with ``ast.literal_eval`` so numbers, tuples, dicts and
    bools work; anything that fails to parse is kept as a raw string.
    """
    if not overrides:
        return
    roots = {"env": env_cfg, "agent": agent_cfg}
    for assignment in overrides:
        if "=" not in assignment:
            raise ValueError(f"Invalid --set override (missing '='): {assignment!r}")
        key, _, raw = assignment.partition("=")
        key, raw = key.strip(), raw.strip()
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw  # treat unparseable values as plain strings

        root_name, _, sub = key.partition(".")
        if root_name in roots and sub:
            root, path = roots[root_name], sub.split(".")
        else:
            root, path = env_cfg, key.split(".")

        parent = _resolve_parent(root, path[:-1])
        _set_leaf(parent, path[-1], value)
        print(f"[INFO] Override applied: {key} = {value!r}")


def validate_trained_robot_layout(env_cfg, run_dir: str) -> dict:
    """Reject same-sized policies belonging to another robot or observation frame."""
    path = os.path.join(run_dir, "params", "env.yaml")
    with open(path) as file:
        saved = yaml.unsafe_load(file)
    robot = saved.get("robot", {})
    current_revision = getattr(env_cfg.robot, "asset_revision", None)
    if robot.get("asset_revision") != current_revision:
        raise ValueError(
            f"Robot physical asset revision does not match checkpoint config: {path}. "
            "Start a new run for the current USD instead of automatically resuming the old asset."
        )
    saved_names = robot.get("action_joint_names")
    if saved_names is not None:
        if list(saved_names) != list(env_cfg.robot.action_joint_names):
            raise ValueError(f"Robot action joint order does not match checkpoint config: {path}")
        if robot.get("base_body_name") != env_cfg.robot.base_body_name:
            raise ValueError(f"Robot observation frame does not match checkpoint config: {path}")
    else:
        # Bundled G1 checkpoints predate explicit joint/frame metadata.
        saved_asset = saved.get("scene", {}).get("robot", {}).get("spawn", {}).get("usd_path", "")
        if os.path.basename(saved_asset) != "g1.usd" or env_cfg.robot.base_body_name is not None:
            raise ValueError(f"Checkpoint lacks a verifiable robot joint layout: {path}")
        if os.path.basename(env_cfg.scene.robot.spawn.usd_path) != "g1.usd":
            raise ValueError(f"Legacy G1 checkpoint cannot be loaded for this robot: {path}")
    # The height command occupies the same observation slot for torso and pelvis
    # tasks, but has different physical meaning. Equal network shapes are not
    # enough to make these checkpoints interchangeable.
    height_term = getattr(getattr(env_cfg, "reward", None), "track_body_height_exp", None)
    if height_term is not None:
        current_bodies = height_term.params["asset_cfg"].body_names
        saved_bodies = (saved.get("reward", {}).get("track_body_height_exp", {})
                        .get("params", {}).get("asset_cfg", {}).get("body_names"))
        current_bodies = ([current_bodies] if isinstance(current_bodies, str)
                          else list(current_bodies) if current_bodies is not None else None)
        saved_bodies = ([saved_bodies] if isinstance(saved_bodies, str)
                        else list(saved_bodies) if saved_bodies is not None else None)
        if ((saved_bodies is not None and current_bodies != saved_bodies)
                or (current_bodies == ["waist_z_link"] and saved_bodies is None)):
            raise ValueError(
                f"Height-command reference does not match checkpoint config: {path}. "
                f"Checkpoint uses {saved_bodies}, task uses {current_bodies}. "
                "Select the checkpoint's original task or start a new training run."
            )
    return saved


def _restore_high_stance_commands(command_cfg, saved_commands):
    """Old checkpoints predate this opt-in distribution; do not enable it in play."""
    if command_cfg is None:
        return
    for field in ("rel_high_stance_turn_envs", "rel_high_stance_move_turn_envs", "rel_low_stance_move_envs"):
        if hasattr(command_cfg, field):
            setattr(command_cfg, field, saved_commands.get(field, 0.0))
    for field in ("high_stance_height_range", "high_stance_speed_range", "high_stance_yaw_rate_range",
                  "low_stance_height_range", "low_stance_speed_range", "low_stance_yaw_rate_range"):
        if field in saved_commands and hasattr(command_cfg, field):
            setattr(command_cfg, field, tuple(saved_commands[field]))
    if hasattr(command_cfg, "low_stance_turn_fraction"):
        command_cfg.low_stance_turn_fraction = saved_commands.get("low_stance_turn_fraction", .5)


def apply_trained_actuator_gains(env_cfg, run_dir: str, *, strict: bool = False) -> None:
    """Apply the actuator PD gains (kp/kd) a checkpoint was trained with.

    Runs may train with different leg/waist stiffness & damping (e.g. via ``--set``
    overrides), which ``train_teacher.py`` persists to ``<run_dir>/params/env.yaml``.
    The default config from the task registry would otherwise evaluate or distill a
    checkpoint with the baseline gains, changing the closed-loop PD behavior. This
    reloads saved stiffness/damping and action settings after validating the robot
    joint order and observation frame. Missing files warn and fall back to the
    defaults unless ``strict`` is set. Student
    distillation uses strict mode because silently changing the teacher dynamics
    invalidates the imitation target.
    """
    params_path = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(params_path):
        if strict:
            raise FileNotFoundError(f"No saved env config at {params_path}")
        print(f"[WARN] No saved env config at {params_path}; using default actuator gains.")
        return
    saved_env = validate_trained_robot_layout(env_cfg, run_dir)
    _restore_high_stance_commands(getattr(env_cfg, "commands", None), saved_env.get("commands", {}))
    if saved_env.get("robot", {}).get("clip_joint_targets", False):
        print("[INFO] This checkpoint used joint-target clipping. That controller layer has "
              "been removed; saved clip_joint_targets=True is ignored.")
    for field in (
        "action_scale", "nominal_height", "termination_height", "arm_motion_source",
        "arm_motion_scale", "arm_motion_ramp_s", "arm_motion_max_velocity",
        "arm_motion_timing",
    ):
        if field in saved_env.get("robot", {}):
            setattr(env_cfg.robot, field, saved_env["robot"][field])
    if saved_env.get("robot", {}).get("arm_motion_source") == "amass_elf3":
        # Existing ELF3 AMASS runs predate the timing option and used source FPS.
        env_cfg.robot.arm_motion_timing = saved_env["robot"].get("arm_motion_timing", "source_fps")
    # Restore task settings that now differ between ELF3 experiment generations.
    # Play applies its deliberate evaluation overrides (zero velocity sampling,
    # disabled pushes/noise) after this function. Keep scene entity mappings and
    # event functions from the verified robot task, restoring saved numeric data.
    saved_physics = saved_env.get("scene", {}).get("robot", {}).get("spawn", {}).get("articulation_props", {})
    for field in ("enabled_self_collisions", "solver_position_iteration_count", "solver_velocity_iteration_count"):
        if field in saved_physics:
            setattr(env_cfg.scene.robot.spawn.articulation_props, field, saved_physics[field])
    for field, value in saved_env.get("commands", {}).get("ranges", {}).items():
        setattr(env_cfg.commands.ranges, field, value)
    for name, saved_event in saved_env.get("domain_rand", {}).get("events", {}).items():
        current = getattr(env_cfg.domain_rand.events, name, None)
        if saved_event is None:
            setattr(env_cfg.domain_rand.events, name, None)
        elif current is not None:
            if "interval_range_s" in saved_event:
                current.interval_range_s = saved_event["interval_range_s"]
            for key, value in saved_event.get("params", {}).items():
                if key != "asset_cfg":
                    current.params[key] = value
    # The dumped env.yaml embeds Python objects (e.g. slice tags) that yaml.full_load
    # refuses to construct. These are our own trusted files, so read with unsafe_load.
    try:
        saved_actuators = saved_env["scene"]["robot"]["actuators"]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        if strict:
            raise ValueError(f"Could not read actuators from {params_path}: {exc}") from exc
        print(f"[WARN] Could not read actuators from {params_path} ({exc}); using default gains.")
        return

    if strict:
        missing = []
        for name in env_cfg.scene.robot.actuators:
            saved_act = saved_actuators.get(name) if isinstance(saved_actuators, dict) else None
            if not isinstance(saved_act, dict):
                missing.append(name)
                continue
            for field in ("stiffness", "damping"):
                if saved_act.get(field) is None:
                    missing.append(f"{name}.{field}")
        if missing:
            raise ValueError(f"Incomplete actuator gains in {params_path}: {', '.join(missing)}")

    applied = []
    for name, act_cfg in env_cfg.scene.robot.actuators.items():
        saved_act = saved_actuators.get(name) if isinstance(saved_actuators, dict) else None
        if not isinstance(saved_act, dict):
            continue
        for field in ("stiffness", "damping"):
            if saved_act.get(field) is not None:
                setattr(act_cfg, field, saved_act[field])
        applied.append(name)

    if applied:
        print(f"[INFO] Applied trained actuator gains from {params_path}:")
        for name in applied:
            act = env_cfg.scene.robot.actuators[name]
            print(f"        {name}: stiffness={act.stiffness}, damping={act.damping}")
    else:
        if strict:
            raise ValueError(f"No matching actuator groups in {params_path}")
        print(f"[WARN] No matching actuator groups in {params_path}; using default gains.")


def apply_trained_command_config(env_cfg, run_dir: str, *, strict: bool = False) -> None:
    """Apply the command sampling distribution saved with a teacher checkpoint.

    Student distillation should visit the same command modes and ranges as the
    teacher. In particular, heading-controlled, direct-yaw, exact-zero-yaw, and
    standing environments produce different state distributions even when their
    instantaneous 7-D commands sometimes coincide. Load those settings from the
    teacher's ``params/env.yaml`` instead of relying on today's task defaults.

    The teacher's curriculum schedule is intentionally not restored: student
    training disables command curriculum and uses the saved final ``ranges`` from
    step zero. Unknown fields are ignored for forward compatibility.
    """
    params_path = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(params_path):
        if strict:
            raise FileNotFoundError(f"No saved env config at {params_path}")
        print(f"[WARN] No saved env config at {params_path}; using default command sampling.")
        return

    try:
        with open(params_path) as f:
            saved_commands = yaml.unsafe_load(f)["commands"]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        if strict:
            raise ValueError(f"Could not read commands from {params_path}: {exc}") from exc
        print(f"[WARN] Could not read commands from {params_path} ({exc}); using defaults.")
        return

    if not isinstance(saved_commands, dict):
        if strict:
            raise ValueError(f"Invalid commands block in {params_path}")
        print(f"[WARN] Invalid commands block in {params_path}; using default command sampling.")
        return

    required_scalar_fields = (
        "resampling_time_range",
        "rel_standing_envs",
        "rel_heading_envs",
        "heading_command",
        "heading_control_stiffness",
    )
    optional_scalar_fields = ("rel_zero_vel_yaw_envs", "rel_in_place_turn_envs", "rel_nominal_height_envs",
                              "in_place_small_turn_fraction", "rel_zero_pose_envs", "rel_single_axis_pose_envs")
    required_range_fields = (
        "lin_vel_x",
        "lin_vel_y",
        "ang_vel_z",
        "heading",
        "body_height",
        "body_roll",
        "body_pitch",
        "body_yaw",
    )
    saved_ranges = saved_commands.get("ranges")
    if strict:
        missing = [field for field in required_scalar_fields if field not in saved_commands]
        missing.extend(
            f"ranges.{field}"
            for field in required_range_fields
            if not isinstance(saved_ranges, dict) or field not in saved_ranges
        )
        unsupported = [field for field in required_scalar_fields if not hasattr(env_cfg.commands, field)]
        unsupported.extend(
            f"ranges.{field}" for field in required_range_fields if not hasattr(env_cfg.commands.ranges, field)
        )
        if missing:
            raise ValueError(f"Incomplete command config in {params_path}: {', '.join(missing)}")
        if unsupported:
            raise ValueError(f"Current task cannot apply saved commands from {params_path}: {', '.join(unsupported)}")

    applied = []
    for field in required_scalar_fields + optional_scalar_fields:
        if field in saved_commands and hasattr(env_cfg.commands, field):
            setattr(env_cfg.commands, field, saved_commands[field])
            applied.append(field)
    _restore_high_stance_commands(env_cfg.commands, saved_commands)

    # Saved configs from before the direct-zero-yaw feature predate this field.
    # Their historical behavior was a purely uniform direct-yaw subset, i.e. a
    # conditional exact-zero probability of 0.0. Restoring that value is harmless
    # for the common heading-only v7 runs and correct for any older mixed run.
    if "rel_zero_vel_yaw_envs" not in saved_commands and hasattr(env_cfg.commands, "rel_zero_vel_yaw_envs"):
        env_cfg.commands.rel_zero_vel_yaw_envs = 0.0
        applied.append("rel_zero_vel_yaw_envs (legacy default 0.0)")

    if isinstance(saved_ranges, dict):
        for field, value in saved_ranges.items():
            if hasattr(env_cfg.commands.ranges, field):
                setattr(env_cfg.commands.ranges, field, value)
                applied.append(f"ranges.{field}")

    if applied:
        print(f"[INFO] Applied trained command sampling from {params_path}:")
        print(f"        {', '.join(applied)}")
    else:
        if strict:
            raise ValueError(f"No matching command fields in {params_path}")
        print(f"[WARN] No matching command fields in {params_path}; using defaults.")


def update_rsl_rl_cfg(agent_cfg: BaseAgentConfig, args_cli: argparse.Namespace):

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
