import math

import numpy as np
import pytest
import torch

from legged_lab.utils.amass import AMassDatasetLoader
from legged_lab.utils.elf3_arm_motion import ELF3_AMASS_JOINT_NAMES, Elf3ArmMotion, map_g1_arms_to_elf3


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


def test_limits_rate_ramp_and_per_environment_reset():
    defaults = torch.zeros(2, 14)
    limits = torch.tensor([-0.5, 0.5]).expand(2, 14, 2).clone()
    motion = Elf3ArmMotion(ELF3_AMASS_JOINT_NAMES, defaults, limits, 0.02, ramp_s=1, max_velocity=2)
    pose = torch.ones(2, 14)
    old = motion.target.clone()
    first = motion.step(pose).clone()
    assert first.abs().max() < 0.01
    for _ in range(100):
        target = motion.step(pose).clone()
        assert (target.abs() <= 0.5).all()
        assert ((target - old).abs() <= 0.040001).all()
        old = target
    untouched = motion.target[1].clone()
    motion.reset(torch.tensor([0]), torch.full((1, 14), -0.2))
    assert torch.equal(motion.target[1], untouched)
    assert motion.elapsed[0] == 0 and motion.elapsed[1] > 1
    motion.step(-pose)
    assert (motion.target[0] + 0.2).abs().max() < 0.01
    assert motion.clipped_fraction > 0


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


@pytest.mark.parametrize("fps", [60, 120])
def test_amass_resampling_uses_seconds_and_handles_short_clips(tmp_path, fps):
    root = tmp_path / "g1" / "CMU"
    root.mkdir(parents=True)
    # Unit slope: every joint moves at 1 rad/s. Source FPS must not change speed.
    data = np.zeros((3 * fps + 1, 36), dtype=np.float32)
    data[:, 22:36] = (np.arange(len(data)) / fps)[:, None]
    np.save(root / f"linear_{fps}_jpos.npy", data)
    loader = AMassDatasetLoader(tmp_path, "cpu")
    samples = loader.sample_arm_sequence(3, 20, control_dt=0.02)
    assert samples.shape == (3, 20, 14)
    assert torch.allclose(samples[:, 1:] - samples[:, :-1], torch.full((3, 19, 14), 0.02), atol=1e-6)
    # Longer requests reflect at the boundaries, rather than jumping or freezing.
    samples = loader.sample_arm_sequence(3, 400, control_dt=0.02)
    assert torch.isfinite(samples).all() and samples.min() >= 0 and samples.max() <= 3
    assert (samples[:, 1:] - samples[:, :-1]).abs().max() <= 0.020001
    assert samples[:, -50:].std(dim=1).min() > 0.1


def test_single_frame_and_missing_dataset(tmp_path):
    root = tmp_path / "g1" / "CMU"
    root.mkdir(parents=True)
    loader = AMassDatasetLoader(tmp_path, "cpu")
    with pytest.raises(ValueError, match="No AMASS"):
        loader.sample_arm_sequence(1, 10, control_dt=0.02)
    np.save(root / "one_60_jpos.npy", np.ones((1, 36), dtype=np.float32))
    loader = AMassDatasetLoader(tmp_path, "cpu")
    assert torch.equal(loader.sample_arm_sequence(2, 10, control_dt=0.02), torch.ones(2, 10, 14))
