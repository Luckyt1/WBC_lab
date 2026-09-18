from types import SimpleNamespace

import pytest
import torch

from legged_lab.utils.tracking_repair import TrackingRepairSchedule, shallow_precision_reward


def generator(count=1000):
    gen = SimpleNamespace(command=torch.randn(count, 7), time_left=torch.ones(count),
                          _env=SimpleNamespace(step_dt=.02))
    for name in ('is_heading_env', 'is_zero_vel_yaw_env', 'is_in_place_turn_env',
                 'is_nominal_height_env', 'is_high_stance_turn_env',
                 'is_high_stance_move_turn_env', 'is_low_stance_move_env', 'is_standing_env'):
        setattr(gen, name, torch.ones(count, dtype=torch.bool))
    gen._resample_command = lambda ids: None
    gen._update_command = lambda: None
    return gen


@pytest.mark.parametrize('mode', ['shallow', 'yaw'])
def test_specialist_range_transitions_and_untouched_general_environments(mode):
    torch.manual_seed(10)
    gen = generator(10000)
    before = gen.command.clone()
    schedule = TrackingRepairSchedule(gen, mode)
    torch.testing.assert_close(gen.command[2000:], before[2000:])
    assert not gen.command[:2000, :3].any()
    assert not gen.command[:2000, 4:].any()
    assert (gen.time_left[:2000] == 16.).all()
    assert (gen.time_left[2000:] == 1.).all()
    schedule.elapsed[:2000] = 4.
    schedule.update()
    torch.testing.assert_close(gen.command[:2000], schedule.target[:2000])
    torch.testing.assert_close(gen.command[2000:], before[2000:])
    if mode == 'shallow':
        assert (gen.command[:2000, 3] >= .72).all()
        assert (gen.command[:2000, 3] <= .78).all()
        assert not gen.command[:2000, 4:].any()
        assert .45 < (schedule.prepare[:2000, 3] > .8).float().mean() < .55
    else:
        assert (gen.command[:2000, 6].abs() >= .1).all()
        assert (gen.command[:2000, 6].abs() <= .2).all()
        assert .45 < (gen.command[:2000, 6] > 0).float().mean() < .55
        assert not gen.command[:2000, 4:6].any()


def test_partial_reset_does_not_restart_other_specialist_episodes():
    gen = generator()
    schedule = TrackingRepairSchedule(gen, 'shallow')
    schedule.elapsed[:] = 7.
    schedule.apply(schedule.ids)
    before = gen.command.clone()
    gen._resample_command(torch.tensor([0, 450]))
    assert schedule.elapsed[0] == 0
    assert (schedule.elapsed[1:] == 7.).all()
    torch.testing.assert_close(gen.command[1:], before[1:])


def test_precision_reward_is_gated_and_uses_pelvis_height():
    gen = generator(10)
    schedule = TrackingRepairSchedule(gen, 'shallow', fraction=.4)
    schedule.elapsed[:] = 5.
    schedule.apply(schedule.ids)
    pos = torch.zeros(10, 2, 3)
    pos[:, 1, 2] = gen.command[:, 3]
    pos[1, 1, 2] += .1
    schedule.elapsed[2] = 0.
    env = SimpleNamespace(tracking_repair_schedule=schedule, command_generator=gen,
                          robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=pos)), base_body_id=1)
    reward = shallow_precision_reward(env)
    assert reward[0] == reward[3] == 1.
    assert 0 < reward[1] < .02
    assert reward[2] == 0 and not reward[4:].any()


@pytest.mark.parametrize('kwargs', [{'mode': 'bad'}, {'mode': 'yaw', 'yaw_max': .8},
                                  {'mode': 'shallow', 'fraction': 0}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        TrackingRepairSchedule(generator(), **kwargs)
