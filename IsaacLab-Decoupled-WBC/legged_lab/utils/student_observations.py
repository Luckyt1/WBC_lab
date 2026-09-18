"""Shared helpers for student-policy observation and layout metadata."""

from __future__ import annotations

import json
from typing import Any

import torch


COMMAND_FIELDS = (
    "lin_vel_x",
    "lin_vel_y",
    "ang_vel_z",
    "body_height",
    "body_roll",
    "body_pitch",
    "body_yaw",
)

LEGACY_G1_ACTION_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

STUDENT_LAYOUT_EXTRA_FILE = "student_layout.json"


def student_base_obs_dim(num_actions: int) -> int:
    """Return ang_vel + gravity + command + q + qd + last_action dimensions."""
    return 13 + 3 * int(num_actions)


def student_input_dim(num_actions: int, history_length: int) -> int:
    return int(history_length) * student_base_obs_dim(num_actions)


def strip_student_teacher_contacts(obs: torch.Tensor, num_actions: int) -> torch.Tensor:
    """Remove teacher-only feet contact channels from actor observations."""
    return obs[:, : student_base_obs_dim(num_actions)]


def policy_joint_names(env: Any) -> list[str]:
    if hasattr(env, "action_joint_names"):
        return list(env.action_joint_names)
    joint_ids = list(env.custom_joint_ids[: env.num_actions])
    names = env.robot.data.joint_names
    return [names[int(joint_id)] for joint_id in joint_ids]


def student_layout(env: Any, history_length: int) -> dict[str, Any]:
    joint_names = policy_joint_names(env)
    base_dim = student_base_obs_dim(env.num_actions)
    return {
        "history_length": int(history_length),
        "base_obs_dim": base_dim,
        "input_dim": int(history_length) * base_dim,
        "action_dim": int(env.num_actions),
        "joint_names": joint_names,
    }


def validate_student_layout(
    checkpoint: dict[str, Any],
    expected_layout: dict[str, Any],
    checkpoint_path: str,
) -> None:
    """Reject checkpoints trained for a different robot/action layout.

    Older G1 student checkpoints did not include a layout block. Those are
    allowed only when the current policy joint order is the known legacy G1
    order; ELF3 and other robots must carry explicit layout metadata.
    """
    saved_layout = checkpoint.get("student_layout")
    if saved_layout is None:
        if tuple(expected_layout["joint_names"]) != LEGACY_G1_ACTION_JOINT_NAMES:
            raise RuntimeError(
                f"Legacy student checkpoint has no layout metadata in {checkpoint_path}; "
                "only the legacy G1 action joint order may load metadata-free checkpoints."
            )
        config = checkpoint.get("config", {})
        saved_input_dim = config.get("input_dim", config.get("obs_dim"))
        saved_action_dim = config.get("action_dim", config.get("act_dim"))
        mismatches = []
        if saved_input_dim is not None and int(saved_input_dim) != int(expected_layout["input_dim"]):
            mismatches.append(f"input_dim {saved_input_dim} != {expected_layout['input_dim']}")
        if saved_action_dim is not None and int(saved_action_dim) != int(expected_layout["action_dim"]):
            mismatches.append(f"action_dim {saved_action_dim} != {expected_layout['action_dim']}")
        if mismatches:
            raise RuntimeError(
                f"Legacy student checkpoint shape mismatch in {checkpoint_path}: "
                + ", ".join(mismatches)
            )
        return

    validate_saved_student_layout(saved_layout, expected_layout, checkpoint_path)


def validate_saved_student_layout(
    saved_layout: dict[str, Any],
    expected_layout: dict[str, Any],
    checkpoint_path: str,
) -> None:
    scalar_keys = ("history_length", "base_obs_dim", "input_dim", "action_dim")
    mismatches = [
        f"{key} {saved_layout.get(key)} != {expected_layout[key]}"
        for key in scalar_keys
        if int(saved_layout.get(key, -1)) != int(expected_layout[key])
    ]
    if list(saved_layout.get("joint_names", [])) != list(expected_layout["joint_names"]):
        mismatches.append("joint_names differ")
    if mismatches:
        raise RuntimeError(
            f"Student checkpoint layout mismatch in {checkpoint_path}: "
            + "; ".join(mismatches)
        )


def encode_student_layout_extra_file(layout: dict[str, Any]) -> str:
    return json.dumps(layout, sort_keys=True)


def decode_student_layout_extra_file(extra_file_value: str | bytes, checkpoint_path: str) -> dict[str, Any]:
    if isinstance(extra_file_value, bytes):
        extra_file_value = extra_file_value.decode("utf-8")
    if not extra_file_value:
        raise RuntimeError(
            f"JIT student checkpoint has no {STUDENT_LAYOUT_EXTRA_FILE} metadata: {checkpoint_path}"
        )
    try:
        decoded = json.loads(extra_file_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JIT student layout metadata in {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Invalid JIT student layout metadata in {checkpoint_path}: expected object")
    return decoded


def validate_jit_student_layout_extra_file(
    extra_file_value: str | bytes,
    expected_layout: dict[str, Any],
    checkpoint_path: str,
) -> None:
    """Validate TorchScript student metadata, allowing only legacy G1 when absent."""
    if isinstance(extra_file_value, bytes):
        extra_file_value = extra_file_value.decode("utf-8")
    if not extra_file_value:
        if tuple(expected_layout["joint_names"]) == LEGACY_G1_ACTION_JOINT_NAMES:
            return
        raise RuntimeError(
            f"JIT student checkpoint has no {STUDENT_LAYOUT_EXTRA_FILE} metadata in {checkpoint_path}; "
            "only the legacy G1 action joint order may load metadata-free JIT checkpoints."
        )
    saved_layout = decode_student_layout_extra_file(extra_file_value, checkpoint_path)
    validate_saved_student_layout(saved_layout, expected_layout, checkpoint_path)


def command_ranges(cfg_or_env: Any) -> dict[str, tuple[float, float]]:
    if isinstance(cfg_or_env, dict):
        return {field: tuple(float(v) for v in cfg_or_env[field]) for field in COMMAND_FIELDS}
    cfg = getattr(cfg_or_env, "cfg", cfg_or_env)
    ranges = cfg.commands.ranges
    return {field: tuple(float(v) for v in getattr(ranges, field)) for field in COMMAND_FIELDS}


def nominal_height(cfg_or_env: Any) -> float:
    cfg = getattr(cfg_or_env, "cfg", cfg_or_env)
    robot_cfg = getattr(cfg, "robot", None)
    configured = getattr(robot_cfg, "nominal_height", None)
    if configured is not None:
        return float(configured)
    lo, hi = command_ranges(cfg)["body_height"]
    return 0.5 * (lo + hi)


def command_limit_tensors(
    cfg_or_env: Any,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    ranges = command_ranges(cfg_or_env)
    lows = [ranges[field][0] for field in COMMAND_FIELDS]
    highs = [ranges[field][1] for field in COMMAND_FIELDS]
    return torch.tensor(lows, device=device, dtype=dtype), torch.tensor(highs, device=device, dtype=dtype)
