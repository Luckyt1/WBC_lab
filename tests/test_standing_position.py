"""Behavioral checks for stop-position anchoring, independent of Isaac Sim."""
import importlib
import math

import pytest
import torch

from test_tang_commands import commands

position = importlib.import_module('tang_cpu.standing_position')


def step(tracker, xy, standing):
    return tracker.step(torch.tensor(xy, dtype=torch.float32), torch.tensor(standing))


def test_distance_remains_penalized_after_motion_stops_and_recovery_reduces_cost():
    tracker = position.StandingPositionTracker(1, 'cpu', .02, settling_time=0.)
    assert step(tracker, [[4., -3.]], [True]).item() == 0
    for distance, expected in [(0.005, 0), (.01, 0), (.05, .4), (.10, .9), (1., 9.9), (.05, .4), (0., 0)]:
        # Holding the same displaced position for many frames must not erase it.
        for _ in range(20):
            assert step(tracker, [[4. + distance, -3.]], [True]).item() == pytest.approx(expected, abs=3e-6)


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_finite_braking_window_freezes_anchor_even_without_slowing_down(dt):
    tracker = position.StandingPositionTracker(1, 'cpu', dt)
    grace_steps = math.ceil(.5 / dt)
    # Moving 1 m/s cannot postpone the anchor indefinitely.
    for frame in range(1, grace_steps + 1):
        assert step(tracker, [[frame * dt, 0.]], [True]).item() == 0
    anchor = grace_steps * dt
    for frame in range(grace_steps + 1, grace_steps + 101):
        distance = frame * dt - anchor
        assert step(tracker, [[frame * dt, 0.]], [True]).item() == pytest.approx(max(distance - .01, 0.) / .1, abs=1e-5)


def test_walking_turning_and_return_to_standing_use_new_local_reference():
    tracker = position.StandingPositionTracker(3, 'cpu', .02, settling_time=0.)
    cmd = commands.TangCommands(commands.TangCommandCfg(), 3, 'cpu')
    for command in ([.3, 0., 0., .8235, 0.], [0., 0., .4, .8235, 0.]):
        cmd.set_manual_command(command)
        assert not tracker.step(torch.full((3, 2), 100.), cmd.is_standing_env).any()
    cmd.set_manual_command([0., 0., 0., .8235, 0.])
    assert not tracker.step(torch.full((3, 2), 100.), cmd.is_standing_env).any()
    torch.testing.assert_close(tracker.step(torch.tensor([[100.1, 100.]]).expand(3, -1), cmd.is_standing_env),
                               torch.full((3,), .9), atol=2e-5, rtol=0)
    cmd.set_manual_command([.3, 0., 0., .8235, 0.])
    assert not tracker.step(torch.full((3, 2), 200.), cmd.is_standing_env).any()
    cmd.set_manual_command([0., 0., 0., .8235, 0.])
    assert not tracker.step(torch.full((3, 2), 201.), cmd.is_standing_env).any()


def test_repeated_standing_resamples_and_height_yaw_targets_do_not_erase_offset():
    tracker = position.StandingPositionTracker(1, 'cpu', .02)
    cmd = commands.TangCommands(commands.TangCommandCfg(resampling_time_range=(.1, .1)), 1, 'cpu')
    cmd.set_manual_command([0., 0., 0., .8235, 0.])
    for _ in range(25):
        cmd.compute(.02)
        assert tracker.step(torch.zeros(1, 2), cmd.is_standing_env).item() == 0
    for _ in range(100):
        cmd.compute(.02)
        assert tracker.step(torch.tensor([[.11, 0.]]), cmd.is_standing_env).item() == pytest.approx(1.)
    cmd.set_manual_command([0., 0., 0., .65, .3])
    assert tracker.step(torch.tensor([[.11, 0.]]), cmd.is_standing_env).item() == pytest.approx(1.)


def test_partial_reset_does_not_erase_other_environments_or_charge_teleports():
    tracker = position.StandingPositionTracker(2, 'cpu', .02, settling_time=0.)
    step(tracker, [[0., 0.], [3., 4.]], [True, True])
    torch.testing.assert_close(step(tracker, [[.11, 0.], [3., 4.11]], [True, True]), torch.ones(2))
    tracker.reset(torch.tensor([0]))
    torch.testing.assert_close(step(tracker, [[100., 200.], [3., 4.11]], [True, True]), torch.tensor([0., 1.]))
    tracker.reset(torch.tensor([], dtype=torch.long))
    torch.testing.assert_close(step(tracker, [[100., 200.], [3., 4.11]], [True, True]), torch.tensor([0., 1.]))
    tracker.reset()
    assert not step(tracker, [[-100., 200.], [-300., 400.]], [True, True]).any()


def test_horizontal_distance_is_translation_and_rotation_invariant():
    tracker = position.StandingPositionTracker(3, 'cpu', .02, settling_time=0.)
    step(tracker, [[0., 0.], [9., -12.], [-6., 8.]], [True] * 3)
    costs = step(tracker, [[.03, .04], [9.04, -12.03], [-6.05, 8.]], [True] * 3)
    torch.testing.assert_close(costs, torch.full((3,), .4), atol=4e-6, rtol=0)


@pytest.mark.parametrize('kwargs', [dict(dt=0), dict(dt=float('nan')), dict(settling_time=-1),
                                   dict(deadzone=-.01), dict(distance_scale=0), dict(distance_scale=float('inf'))])
def test_invalid_parameters_fail_early(kwargs):
    with pytest.raises(ValueError):
        position.StandingPositionTracker(1, 'cpu', **(dict(dt=.02) | kwargs))
