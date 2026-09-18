import math
import torch
from legged_lab.utils.recovery_tracking import (
    angular_tracking, high_stance_commands, low_stance_commands, mix_pose_samples, small_turn_samples, smooth_stance,
    standing_foot_lift, yaw_rate_tracking,
)


def test_low_stance_mix_preserves_pose_and_has_balanced_motion():
    torch.manual_seed(31)
    original = torch.randn(60000, 7)
    before = original.clone()
    command = low_stance_commands(original, (.4735, .6), (.2, .4), (.2, .6))
    assert torch.equal(original, before)
    assert torch.equal(command[:, 4:], before[:, 4:])
    assert (command[:, 3] >= .4735).all() and (command[:, 3] <= .6).all()
    assert (command[:, :2].count_nonzero(-1) == 1).all()
    speed = command[:, :2].norm(dim=-1)
    assert (speed >= .2).all() and (speed <= .4).all()
    yaw = command[:, 2]
    turning = yaw != 0
    assert .49 < float(turning.float().mean()) < .51
    assert (yaw[turning].abs() >= .2).all() and (yaw[turning].abs() <= .6).all()
    assert .49 < float((yaw[turning] > 0).float().mean()) < .51
    for direction in (command[:, 0] > 0, command[:, 1] > 0, command[:, 1] < 0):
        assert abs(float(direction.float().mean()) - 1/3) < .01
    for fraction in (0., 1.):
        sample = low_stance_commands(original[:100], (.4735, .6), (.2, .4), (.2, .6), fraction)
        assert bool((sample[:, 2] != 0).all()) if fraction else not sample[:, 2].any()


def test_high_stance_commands_cover_both_turns_and_three_translation_directions():
    torch.manual_seed(42)
    for moving in (False, True):
        command = high_stance_commands(30000, 'cpu', moving, (.78, .8235), (.3, .6), (.2, .6))
        assert (command[:, 3] >= .78).all() and (command[:, 3] <= .8235).all()
        assert (command[:, 2].abs() >= .2).all() and (command[:, 2].abs() <= .6).all()
        assert .48 < float((command[:, 2] > 0).float().mean()) < .52
        assert not command[:, 4:].any()
        if moving:
            assert (command[:, :2].count_nonzero(-1) == 1).all()
            speed = command[:, :2].norm(dim=-1)
            assert (speed >= .3).all() and (speed <= .6).all()
            for direction in (command[:, 0] > 0, command[:, 1] > 0, command[:, 1] < 0):
                assert abs(float(direction.float().mean()) - 1/3) < .015
        else:
            assert not command[:, :2].any()


def test_small_turn_and_pose_distribution():
    torch.manual_seed(42)
    turns = small_turn_samples(100000, 'cpu')
    assert .69 < float((turns.abs() <= .6).float().mean()) < .71
    assert .49 < float((turns > 0).float().mean()) < .51
    assert turns.abs().min() >= .15 and turns.abs().max() <= 1.57
    pose = mix_pose_samples(torch.ones(100000, 3), .2, .3)
    axes = pose.count_nonzero(dim=-1)
    for count, expected in ((0, .2), (1, .3), (3, .5)):
        assert abs(float((axes == count).float().mean()) - expected) < .01
    torch.testing.assert_close(pose[axes == 1].mean(0), torch.full((3,), 1/3), atol=.01, rtol=0)


def test_shortest_yaw_across_global_heading_boundary():
    measured = torch.tensor([math.radians(-358), math.radians(358)])
    target = torch.tensor([math.radians(2), math.radians(-2)])
    torch.testing.assert_close(angular_tracking(measured, target, .3), torch.ones(2))
    torch.testing.assert_close(yaw_rate_tracking(torch.tensor([.2]), torch.tensor([0.])),
                               torch.tensor([math.exp(-(.2/.15)**2)]))


def test_stance_is_dense_and_does_not_penalize_turning():
    contact = torch.tensor([[.02, .02], [.3, 0.], [0., 0.], [.3, .3]])
    command = torch.zeros(4, 7)
    command[2:, 2] = torch.tensor([.15, -.6])
    reward = smooth_stance(contact, command)
    assert reward[0] > 0
    torch.testing.assert_close(reward[1:], torch.zeros(3))
    torch.testing.assert_close(standing_foot_lift(contact, command), torch.tensor([0., .5, 0., 0.]))
