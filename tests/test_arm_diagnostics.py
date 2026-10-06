"""Arm telemetry and paired replay checks; CPU only, no Isaac startup."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from test_tang_commands import commands  # registers the simulator-free package

diagnostics = importlib.import_module('tang_cpu.arm_diagnostics')
loader_module = importlib.import_module('tang_cpu.wbc.amass')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from arm_speed_metrics import resample_source, summarize


def update(monitor, reference, target=None, velocity=None, standing=None):
    n = len(reference)
    flag = torch.zeros(n, dtype=torch.bool)
    return monitor.update(reference, reference if target is None else target,
                          torch.zeros_like(reference), torch.zeros_like(reference) if velocity is None else velocity,
                          ~flag if standing is None else standing, ~flag, flag, flag)


def test_derivatives_units_and_reset_boundaries_without_mutation_or_rng():
    monitor = diagnostics.ArmMotionDiagnostics(2, 14, .02, 'cpu')
    q = torch.ones(2, 14)
    before = q.clone()
    rng = torch.random.get_rng_state().clone()
    first, _ = update(monitor, q)
    assert not first['velocity_valid'].any()
    assert not first['actual_acceleration_valid'].any()
    second, _ = update(monitor, q + .02, velocity=torch.ones_like(q))
    torch.testing.assert_close(second['target_velocity'], torch.ones_like(q))
    assert not second['acceleration_valid'].any()
    torch.testing.assert_close(second['actual_acceleration'], torch.full_like(q, 50.))
    third, _ = update(monitor, q + .06, velocity=torch.ones_like(q))
    torch.testing.assert_close(third['target_acceleration'], torch.full_like(q, 50.), atol=.001, rtol=1e-5)
    monitor.reset(torch.tensor([0]))
    next_q = q + .10
    next_q[0] = 100.
    fourth, log = update(monitor, next_q, standing=torch.zeros(2, dtype=torch.bool))
    assert fourth['target_velocity'][0].eq(0).all()
    assert fourth['target_acceleration'][0].eq(0).all()
    assert fourth['velocity_valid'].tolist() == [False, True]
    assert all(torch.isfinite(x) for x in log.values())
    assert log['StandingArms/sample_fraction'] == 0
    torch.testing.assert_close(q, before)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_reference_target_and_actual_movement_are_distinct():
    m = diagnostics.ArmMotionDiagnostics(2, 14, .02, 'cpu')
    zero = torch.zeros(2, 14)
    update(m, zero)
    target = zero.clone()
    target[0] = .02
    actual_velocity = zero.clone()
    actual_velocity[1] = .1
    sample, log = update(m, zero+.1, target, actual_velocity, torch.tensor([True, False]))
    assert sample['target_moving'].tolist() == [True, False]
    assert sample['actual_moving'].tolist() == [False, True]
    assert log['StandingArms/target_moving_fraction'] == 1
    assert log['StandingArms/actual_moving_fraction'] == 0
    assert log['ArmMotion/actual_moving_fraction'] == .5
    assert log['ArmMotion/reference_velocity_abs_mean_rad_s'] == 5
    assert log['ArmMotion/target_velocity_abs_mean_rad_s'] == .5


def test_padding_metadata_does_not_change_legacy_samples(tmp_path):
    folder = tmp_path/'g1/CMU'
    folder.mkdir(parents=True)
    raw = np.arange(3*36).reshape(3, 36).astype(np.float32)
    np.save(folder/'short_120_jpos.npy', raw)
    loader = loader_module.AMassDatasetLoader(tmp_path, 'cpu', seed=17)
    result = loader.sample_arm_sequence(2, sequence_length=8, interpolation_factor=1)
    assert loader.last_valid_lengths.tolist() == [3, 3]
    np.testing.assert_array_equal(result[0, :3], raw[:, 22:36])
    np.testing.assert_array_equal(result[0, 3:], np.repeat(raw[-1:, 22:36], 5, 0))
    loader.seed(17)
    torch.testing.assert_close(result, loader.sample_arm_sequence(2, 8, 1))


def test_source_speed_uses_physical_time_and_explicit_tail():
    arm = np.arange(121, dtype=np.float32)[:, None].repeat(14, axis=1) / 120.
    one, tail = resample_source(arm, 0, 120., .02, 1., 80)
    half, half_tail = resample_source(arm, 0, 120., .02, .5, 80)
    np.testing.assert_allclose(one[20], .4, atol=1e-6)
    np.testing.assert_allclose(half[20], .2, atol=1e-6)
    assert not tail[50] and tail[51] and not half_tail.any()
    np.testing.assert_allclose(one[-1], 1.)
    with pytest.raises(ValueError):
        resample_source(arm, 0, 120., .02, 0, 80)


def test_summary_excludes_warmup_failures_and_post_reset_spikes():
    m = diagnostics.ArmMotionDiagnostics(3, 14, .02, 'cpu')
    history = {}
    for step in range(6):
        sample, _ = update(m, torch.full((3, 14), step*.02))
        for key, value in sample.items():
            history.setdefault(key, []).append(value.numpy().copy())
    arrays = {k: np.stack(v) for k, v in history.items()}
    arrays['valid'] = np.ones((6, 3), dtype=bool)
    arrays['valid'][1:, 0] = False  # trial 0 failed during warmup
    arrays['valid'][5:, 1] = False  # trial 1 failed in measurement
    arrays['terminated'] = np.zeros((6, 3), dtype=bool)
    arrays['terminated'][0, 0] = arrays['terminated'][4, 1] = True
    arrays['pelvis_xy'] = np.zeros((6, 3, 2))
    arrays['pelvis_xy'][:, 2, 0] = np.arange(6)*.01
    arrays['pelvis_xy'][5, 1] = 1000.  # reset must never count toward drift
    arrays['pelvis_yaw'] = np.zeros((6, 3))
    arrays['horizontal_speed'] = np.zeros((6, 3))
    arrays['height'] = np.ones((6, 3))
    result = summarize(arrays, 2, .02)
    assert result['survived'] == 1 and result['warmup_failures'] == 1
    assert result['measurement_sample_count'] == 7
    assert result['drift_m']['max'] == pytest.approx(.04)
    assert result['max_drift_survivors_m']['mean'] == pytest.approx(.04)
    assert result['first_failure_s'] == [.02, .1, None]
