from types import SimpleNamespace

import pytest
import torch

from legged_lab.utils.student_observations import (
    command_ranges,
    command_limit_tensors,
    decode_student_layout_extra_file,
    encode_student_layout_extra_file,
    LEGACY_G1_ACTION_JOINT_NAMES,
    nominal_height,
    strip_student_teacher_contacts,
    STUDENT_LAYOUT_EXTRA_FILE,
    student_base_obs_dim,
    validate_jit_student_layout_extra_file,
    validate_student_layout,
)


def _cfg(nominal=None):
    ranges = SimpleNamespace(
        lin_vel_x=(-0.2, 0.4),
        lin_vel_y=(-0.3, 0.3),
        ang_vel_z=(-1.0, 1.0),
        body_height=(0.4, 0.8),
        body_roll=(-0.5, 0.5),
        body_pitch=(-0.6, 0.7),
        body_yaw=(-0.8, 0.9),
    )
    robot = SimpleNamespace()
    if nominal is not None:
        robot.nominal_height = nominal
    return SimpleNamespace(commands=SimpleNamespace(ranges=ranges), robot=robot)


def test_student_base_obs_strips_teacher_contacts():
    obs = torch.arange(2 * 60, dtype=torch.float32).reshape(2, 60)
    stripped = strip_student_teacher_contacts(obs, num_actions=15)

    assert student_base_obs_dim(15) == 58
    assert stripped.shape == (2, 58)
    assert torch.equal(stripped, obs[:, :58])


def test_command_limits_and_nominal_height_from_config():
    cfg = _cfg(nominal=0.72)
    lo, hi = command_limit_tensors(cfg, device="cpu")

    assert lo.tolist() == pytest.approx([-0.2, -0.3, -1.0, 0.4, -0.5, -0.6, -0.8])
    assert hi.tolist() == pytest.approx([0.4, 0.3, 1.0, 0.8, 0.5, 0.7, 0.9])
    assert nominal_height(cfg) == pytest.approx(0.72)


def test_nominal_height_falls_back_to_body_height_midpoint():
    assert nominal_height(_cfg()) == pytest.approx(0.6)


def test_layout_validation_rejects_same_shape_different_joint_order():
    checkpoint = {
        "student_layout": {
            "history_length": 2,
            "base_obs_dim": 58,
            "input_dim": 116,
            "action_dim": 15,
            "joint_names": [f"g1_joint_{i}" for i in range(15)],
        }
    }
    expected = {
        "history_length": 2,
        "base_obs_dim": 58,
        "input_dim": 116,
        "action_dim": 15,
        "joint_names": [f"elf3_joint_{i}" for i in range(15)],
    }

    with pytest.raises(RuntimeError, match="joint_names differ"):
        validate_student_layout(checkpoint, expected, "student_step.pt")


def test_legacy_checkpoint_without_layout_only_allowed_for_g1_joint_order():
    legacy_checkpoint = {"config": {"obs_dim": 116, "act_dim": 15}}
    g1_expected = {
        "history_length": 2,
        "base_obs_dim": 58,
        "input_dim": 116,
        "action_dim": 15,
        "joint_names": list(LEGACY_G1_ACTION_JOINT_NAMES),
    }
    elf3_expected = {
        **g1_expected,
        "joint_names": [f"elf3_joint_{i}" for i in range(15)],
    }

    validate_student_layout(legacy_checkpoint, g1_expected, "legacy_g1.pt")
    with pytest.raises(RuntimeError, match="no layout metadata"):
        validate_student_layout(legacy_checkpoint, elf3_expected, "legacy_g1.pt")


def test_jit_layout_extra_file_round_trip_and_missing_rejected():
    layout = {
        "history_length": 1,
        "base_obs_dim": 58,
        "input_dim": 58,
        "action_dim": 15,
        "joint_names": list(LEGACY_G1_ACTION_JOINT_NAMES),
    }
    encoded = encode_student_layout_extra_file(layout)

    assert decode_student_layout_extra_file(encoded, "student_policy_jit.pt") == layout
    with pytest.raises(RuntimeError, match=STUDENT_LAYOUT_EXTRA_FILE):
        decode_student_layout_extra_file("", "student_policy_jit.pt")


def test_jit_layout_validation_allows_missing_metadata_only_for_legacy_g1():
    g1_expected = {
        "history_length": 2,
        "base_obs_dim": 58,
        "input_dim": 116,
        "action_dim": 15,
        "joint_names": list(LEGACY_G1_ACTION_JOINT_NAMES),
    }
    elf3_expected = {
        **g1_expected,
        "joint_names": [f"elf3_joint_{i}" for i in range(15)],
    }

    validate_jit_student_layout_extra_file("", g1_expected, "legacy_g1_jit.pt")
    with pytest.raises(RuntimeError, match="no student_layout.json metadata"):
        validate_jit_student_layout_extra_file("", elf3_expected, "legacy_g1_jit.pt")


def test_jit_layout_validation_rejects_invalid_non_empty_metadata():
    expected = {
        "history_length": 2,
        "base_obs_dim": 58,
        "input_dim": 116,
        "action_dim": 15,
        "joint_names": list(LEGACY_G1_ACTION_JOINT_NAMES),
    }

    with pytest.raises(RuntimeError, match="Invalid JIT student layout metadata"):
        validate_jit_student_layout_extra_file("{not-json", expected, "bad_jit.pt")


def test_command_range_snapshot_survives_play_speed_zeroing():
    cfg = _cfg(nominal=0.72)
    saved_ranges = command_ranges(cfg)
    cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.ranges.ang_vel_z = (0.0, 0.0)

    lo, hi = command_limit_tensors(saved_ranges, device="cpu")

    assert lo[:3].tolist() == pytest.approx([-0.2, -0.3, -1.0])
    assert hi[:3].tolist() == pytest.approx([0.4, 0.3, 1.0])
