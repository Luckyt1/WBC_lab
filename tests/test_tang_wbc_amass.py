"""AMASS transfer geometry, full-amplitude playback, and WBC command layout."""
import importlib
import numpy as np
import pytest
import torch
import math
from test_tang_commands import commands

amass = importlib.import_module('tang_cpu.wbc.amass')
mapping = importlib.import_module('tang_cpu.wbc.elf3_arm_motion')
AMassDatasetLoader = amass.AMassDatasetLoader
ELF3_AMASS_JOINT_NAMES = mapping.ELF3_AMASS_JOINT_NAMES
Elf3ArmMotion = mapping.Elf3ArmMotion
map_g1_arms_to_elf3 = mapping.map_g1_arms_to_elf3

def rotation(axis, angle):
    result = torch.eye(3, dtype=torch.float64)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    result[i, i] = result[j, j] = math.cos(angle)
    result[i, j], result[j, i] = -math.sin(angle), math.sin(angle)
    return result

def test_shoulder_transfer_preserves_orientation_and_named_order():
    poses = torch.tensor([[0.7, 0.5, -0.3, 0.6, 0.1, 0.2, 0.3,
                           -0.4, -0.6, 0.2, 0.7, -0.1, -0.2, -0.3]], dtype=torch.float64)
    mapped = map_g1_arms_to_elf3(poses, ELF3_AMASS_JOINT_NAMES)
    for offset, mount in ((0, math.radians(16)), (7, -math.radians(16))):
        p, r, y = poses[0, offset:offset + 3]
        reference = rotation("x", mount) @ rotation("y", p) @ rotation("x", r - mount) @ rotation("z", y)
        p, r, y = mapped[0, offset:offset + 3]
        actual = rotation("y", p) @ rotation("x", r) @ rotation("z", y)
        assert torch.allclose(actual, reference, atol=1e-10)
        assert torch.equal(poses[:, offset + 3:offset + 7], mapped[:, offset + 3:offset + 7])
    reversed_order = map_g1_arms_to_elf3(poses, ELF3_AMASS_JOINT_NAMES[::-1])
    assert torch.equal(reversed_order, mapped.flip(-1))
    assert torch.allclose(map_g1_arms_to_elf3(torch.zeros_like(poses), ELF3_AMASS_JOINT_NAMES), torch.zeros_like(poses))
    with pytest.raises(ValueError, match="14 named"):
        map_g1_arms_to_elf3(poses, ["wrong"] * 14)

def test_disabled_ramp_and_rate_apply_each_mapped_frame_immediately():
    defaults = torch.zeros(2, 14)
    limits = torch.tensor([-0.5, 0.5]).expand(2, 14, 2).clone()
    motion = Elf3ArmMotion(ELF3_AMASS_JOINT_NAMES, defaults, limits, 0.02, ramp_s=0, max_velocity=None)
    for pose in (torch.ones(2, 14), -torch.ones(2, 14)):
        expected = map_g1_arms_to_elf3(pose, ELF3_AMASS_JOINT_NAMES).clamp(-0.5, 0.5)
        actual = motion.step(pose).clone()
        assert torch.equal(actual, expected)
        assert actual.abs().max() > 0.04
    motion.reset(torch.tensor([0]), torch.full((1, 14), 0.1))
    assert torch.equal(motion.step(pose), expected)


def test_legacy_frames_preserve_source_samples_and_pad_short_clip(tmp_path):
    root = tmp_path / 'g1' / 'CMU'
    root.mkdir(parents=True)
    data = np.zeros((4, 36), dtype=np.float32)
    data[:, 22:36] = np.arange(4)[:, None]
    np.save(root / 'short_120_jpos.npy', data)
    loader = AMassDatasetLoader(tmp_path, 'cpu')
    samples = loader.sample_arm_sequence(2, 7, interpolation_factor=1, control_dt=None)
    expected = torch.tensor([0., 1., 2., 3., 3., 3., 3.])[None, :, None].expand(2, 7, 14)
    torch.testing.assert_close(samples, expected)


def test_amass_seed_replays_clip_choices_and_offsets(tmp_path):
    root = tmp_path / 'g1' / 'CMU'
    root.mkdir(parents=True)
    for i in range(3):
        data = np.zeros((100, 36), dtype=np.float32)
        data[:, 22:36] = i * 1000 + np.arange(100)[:, None]
        np.save(root / f'clip{i}_120_jpos.npy', data)
    a = AMassDatasetLoader(tmp_path, 'cpu', seed=42)
    b = AMassDatasetLoader(tmp_path, 'cpu', seed=42)
    for dt in (None, .02):
        a.seed(42)
        b.seed(42)
        first = a.sample_arm_sequence(16, 20, interpolation_factor=1, control_dt=dt)
        torch.testing.assert_close(first, b.sample_arm_sequence(16, 20, interpolation_factor=1, control_dt=dt))
        second = a.sample_arm_sequence(16, 20, interpolation_factor=1, control_dt=dt)
        assert not torch.equal(first, second)
        a.seed(42)
        torch.testing.assert_close(first, a.sample_arm_sequence(16, 20, interpolation_factor=1, control_dt=dt))


def test_wbc_commands_keep_velocity_height_yaw_and_zero_roll_pitch():
    cmd = commands.TangCommands(commands.TangCommandCfg(), 2, 'cpu')
    cmd.full_command[:] = torch.tensor([[.3, -.2, .4, .6, -.5], [-.1, .2, -.3, .7, .6]])
    before = cmd.full_command.clone()
    torch.testing.assert_close(cmd.wbc_command, torch.tensor([
        [.3, -.2, .4, .6, 0., 0., -.5], [-.1, .2, -.3, .7, 0., 0., .6]]))
    torch.testing.assert_close(cmd.full_command, before)
    cmd.full_command[:, 4] = -.6
    assert torch.equal(cmd.wbc_command[:, 6], torch.full((2,), -.6))
