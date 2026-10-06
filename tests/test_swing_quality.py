"""Physical scoring edge cases and opt-in command coverage, without Isaac Sim."""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
package = ModuleType('swing_cpu')
package.__path__ = [str(ROOT/'legged_lab/envs/elf3_tang')]
sys.modules[package.__name__] = package
swing = importlib.import_module('swing_cpu.swing_quality')
commands = importlib.import_module('swing_cpu.commands')


def step(tracker, contact=(True, True), heights=(0., 0.), active=True):
    return tracker.step(torch.tensor([contact]), torch.tensor([heights]), torch.tensor([active]))[0].item()


def flight(tracker, foot=0, duration=.16, height=.03):
    contact, heights = [True, True], [0., 0.]
    contact[foot], heights[foot] = False, height
    for _ in range(round(duration/tracker.dt)):
        step(tracker, contact, heights)
    value = step(tracker)
    return value*tracker.dt, tracker.last_metrics['qualified'].item()


@pytest.mark.parametrize('dt', [.01, .02])
def test_complete_alternating_swings_and_time_scaling(dt):
    tracker = swing.SwingTracker(1, 'cpu', dt)
    step(tracker)
    assert flight(tracker, 0) == pytest.approx((.35, 1))
    assert flight(tracker, 0) == pytest.approx((-.10, 0))
    assert flight(tracker, 1) == pytest.approx((.35, 1))


def test_contact_flicker_low_clearance_hopping_and_hovering():
    tracker = swing.SwingTracker(1, 'cpu', .02)
    step(tracker)
    event, qualified = flight(tracker, duration=.02)
    assert event < 0 and qualified == 0
    assert flight(tracker, height=.005) == (0, 0)
    for _ in range(8):
        assert step(tracker, (False, False), (.03, .03)) == 0
    assert step(tracker) == 0
    assert tracker.last_metrics['qualified'].item() == 0
    for _ in range(30):
        value = step(tracker, (False, True), (.03, 0.))
    assert value < 0
    assert step(tracker) == 0


def test_reset_and_standing_break_swing_history():
    tracker = swing.SwingTracker(1, 'cpu', .02)
    step(tracker)
    flight(tracker)
    tracker.reset(torch.tensor([0]))
    assert step(tracker) == 0
    assert flight(tracker)[1] == 1
    assert step(tracker, (False, True), (.03, 0.), active=False) == 0
    assert step(tracker) == 0
    assert flight(tracker)[1] == 1
    tracker = swing.SwingTracker(2, 'cpu', .02)
    tracker.air[:] = .2
    tracker.last_foot[:] = 1
    tracker.reset(torch.tensor([0]))
    assert tracker.air.tolist() == [[0., 0.], pytest.approx([.2, .2])]
    assert tracker.last_foot.tolist() == [-1, 1]


def test_collision_hull_height_is_minimum_transformed_vertex():
    feet = json.loads((ROOT/'legged_lab/assets/elf3_bxi_31/foot_collision_hulls.json').read_text())['feet']
    vertices = torch.tensor([feet[name]['vertices'] for name in ('l_ankle_x_link', 'r_ankle_x_link')])
    torch.manual_seed(37)
    q = torch.randn(31, 2, 4)
    q /= q.norm(dim=-1, keepdim=True)
    p, ground = torch.randn(31, 2, 3), torch.randn(31)
    v = vertices[None].expand(31, -1, -1, -1)
    xyz = q[..., None, 1:].expand_as(v)
    cross = 2*torch.cross(xyz, v, dim=-1)
    rotated = v + q[..., None, :1]*cross + torch.cross(xyz, cross, dim=-1)
    expected = rotated[..., 2].amin(-1) + p[..., 2] - ground[:, None]
    torch.testing.assert_close(swing.sole_clearance(p, q, vertices, ground), expected)


def test_mixed_commands_preserve_modes_and_other_targets():
    def sample(fraction):
        cfg = commands.TangCommandCfg(rel_standing_envs=.35, rel_turning_envs=.15,
            rel_lateral_envs=.15, sagittal_speed_range=(.15, .6), turning_speed_range=(.15, .6),
            mixed_yaw_fraction=fraction)
        cmd = commands.TangCommands(cfg, 20000, 'cpu')
        cmd.iteration = 26000
        torch.manual_seed(42)
        cmd.reset(torch.arange(cmd.num_envs))
        return cmd
    baseline, mixed = sample(0), sample(.5)
    torch.testing.assert_close(baseline.sampled_mode, mixed.sampled_mode)
    torch.testing.assert_close(baseline.target[:, [0, 1, 3, 4]], mixed.target[:, [0, 1, 3, 4]])
    sagittal = mixed.sampled_mode == 2
    torch.testing.assert_close(baseline.target[~sagittal], mixed.target[~sagittal])
    arcs = sagittal & (mixed.command[:, 2] != 0)
    assert .48 < arcs.sum()/sagittal.sum() < .52
    assert .16 < arcs.float().mean() < .19
    assert .47 < (mixed.command[arcs, 2] > 0).float().mean() < .53
    assert mixed.command[arcs, 2].abs().min() >= .02
    assert mixed.command[arcs, 2].abs().max() <= .15
    mixed.set_manual_command([.3, 0., -.1, .8235, 0.])
    torch.testing.assert_close(mixed.command, torch.tensor([.3, 0., -.1]).expand(20000, -1))


@pytest.mark.parametrize('fraction,bounds', [(-.1, (.02, .15)), (1.1, (.02, .15)), (.5, (0., .15)), (.5, (.02, 1.))])
def test_invalid_mixed_command_config(fraction, bounds):
    with pytest.raises(ValueError):
        commands.TangCommands(commands.TangCommandCfg(sagittal_speed_range=(.1, .6),
            mixed_yaw_fraction=fraction, mixed_yaw_range=bounds), 1, 'cpu')
