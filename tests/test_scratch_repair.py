"""Actual WBC reward regressions, frozen curricula and per-task acceptance."""
import ast
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from test_tang_commands import commands, yaw_quat
from test_tang_rewards import fake_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from scratch_training_gates import STAGES, FULL, tracking_gate, validate_promotion

curriculum = importlib.import_module('tang_cpu.scratch_curriculum')
standing = importlib.import_module('tang_cpu.standing')


def wbc_functions():
    # Execute the actual WBC functions, without importing the simulator application.
    # Math stubs below cover yaw-only test quaternions (including either quaternion sign).
    path = ROOT / 'legged_lab/envs/elf3_tang/wbc/rewards.py'
    names = {'feet_air_time_positive_biped', 'feet_contact_standing_stable', 'track_body_yaw_exp'}
    definitions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             *definitions], type_ignores=[])
    def multiply(a, b):
        q = torch.zeros_like(a)
        q[:, 0] = a[:, 0]*b[:, 0] - a[:, 3]*b[:, 3]
        q[:, 3] = a[:, 0]*b[:, 3] + a[:, 3]*b[:, 0]
        return q
    namespace = dict(torch=torch, SceneEntityCfg=lambda *a, **kw: NS(**kw),
                     math_utils=NS(yaw_quat=lambda q: q, quat_mul=multiply,
                                   quat_conjugate=lambda q: q*q.new_tensor([1, -1, -1, -1])))
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return NS(**{n: namespace[n] for n in names})


def test_actual_wbc_small_motion_and_turn_do_not_receive_stationary_bonus():
    env = fake_env(3)
    env.command_generator.full_command[:, :3] = torch.tensor([[0., 0., 0.], [.02, 0., 0.], [0., 0., .02]])
    env.command_generator.is_standing_env[:] = torch.tensor([True, False, False])
    env.scene = NS(sensors={'contact_sensor': env.contact_sensor})
    cfg = NS(name='contact_sensor', body_ids=env.feet_cfg.body_ids)
    wbc = wbc_functions()
    reward = wbc.feet_contact_standing_stable(env, cfg, use_standing_mask=True)
    assert reward[0] > 0 and reward[1:].sum() == 0
    # Reference WBC mode remains available for historical parity checks.
    assert (wbc.feet_contact_standing_stable(env, cfg) > 0).all()
    env.contact_sensor.data.current_contact_time[:, cfg.body_ids[0]] = 0
    env.contact_sensor.data.current_air_time[:, cfg.body_ids[0]] = .2
    torch.testing.assert_close(wbc.feet_air_time_positive_biped(env, .4, cfg, True), torch.tensor([0., .2, .2]))


def test_actual_wbc_yaw_wrap_and_quaternion_sign_invariance():
    env = fake_env(3)
    env.torso_body_id = 1
    env.scene = {'robot': env.robot}
    env.robot.data.body_quat_w[:, 0] = yaw_quat([3.1, -3.1, 0.])
    env.robot.data.body_quat_w[:, 1] = yaw_quat([-3.0, 3.0, .2])
    env.command_generator.full_command[:, 4] = commands.wrap_to_pi(torch.tensor([-6.1, 6.1, .2]))
    cfg = NS(name='robot', body_ids=[0, 1])
    reward = wbc_functions().track_body_yaw_exp
    torch.testing.assert_close(reward(env, .3, cfg, True), torch.ones(3), atol=1e-6, rtol=0)
    env.robot.data.body_quat_w[:, 1] *= -1
    torch.testing.assert_close(reward(env, .3, cfg, True), torch.ones(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(standing.waist_yaw_tracking_error(env), torch.zeros(3), atol=1e-6, rtol=0)
    env.command_generator.full_command[:, 4] -= .6
    torch.testing.assert_close(standing.waist_yaw_tracking_error(env), torch.full((3,), .6), atol=1e-6, rtol=0)


def test_basic_stage_never_expands_by_iteration_and_keeps_manual_range():
    torch.manual_seed(13)
    cmd = commands.TangCommands(curriculum.stage_commands('locomotion'), 12000, 'cpu')
    for age in (0, 30000, 1000000):
        cmd.iteration = age
        cmd.reset(torch.arange(cmd.num_envs))
        mode = cmd.sampled_mode
        assert abs(float((mode == 0).float().mean()) - .25) < .02
        assert abs(float((mode == 1).float().mean()) - .25) < .02
        move, turn = mode == 2, mode == 1
        assert (cmd.target[move, 0].abs() >= .2).all() and (cmd.target[move, 0].abs() <= .4).all()
        assert .45 < float((cmd.target[move, 0] > 0).float().mean()) < .55
        assert (cmd.target[turn, 2].abs() >= .3).all()
        assert (cmd.target[move, 1:3] == 0).all()
        assert (cmd.target[:, 3] == .8235).all() and (cmd.target[:, 4] == 0).all()
    cmd.set_manual_command([0., 0., 0., .5, .6])
    torch.testing.assert_close(cmd.target[0], torch.tensor([0., 0., 0., .5, .6]))


def test_pose_stage_practices_one_pose_axis_and_retains_locomotion():
    torch.manual_seed(14)
    cmd = commands.TangCommands(curriculum.stage_commands('pose'), 12000, 'cpu')
    cmd.reset(torch.arange(cmd.num_envs))
    low, yaw = cmd.target[:, 3] < .82, cmd.target[:, 4].abs() > .01
    assert low.sum() > 1000 and yaw.sum() > 1000
    assert not (low & yaw).any()
    assert (cmd.is_standing_env[low | yaw]).all()
    assert (cmd.sampled_mode == 3).sum() > 1500


def test_all_stages_have_frozen_separate_disturbances():
    assert STAGES == curriculum.STAGES
    for stage in STAGES:
        cfg = NS(arm_motion=NS(), noise=NS(), push_velocity_curriculum=NS(), reset_velocity_curriculum=NS())
        curriculum.configure_scratch_stage(cfg, stage)
        assert cfg.arm_motion.amass_scale_end_iteration is None
        assert cfg.push_velocity_curriculum.initial_scale == cfg.push_velocity_curriculum.final_scale
    assert [curriculum.stage_settings(s)['push_scale'] for s in STAGES] == [0, 0, 0, .125, .25, .5, 1]
    with pytest.raises(ValueError):
        curriculum.stage_commands('typo')


def good_report():
    rows = [dict(scenario=n, survived=32, trials=32, valid_samples=100,
                 mae=[.01]*5, knee_clipped_fraction={'left': .1, 'right': .1}) for n in FULL]
    return dict(arm_guard_enabled=True, repeats=32, warmup_s=4, measure_s=12,
                profiles=[dict(settings=dict(name='curriculum_stage', stage='locomotion'), scenarios=rows)])


@pytest.mark.parametrize('field,value', [('mae', [.3, 0., 0., 0., 0.]), ('mae', [float('nan')]*5),
                                       ('mae', []), ('survived', 30), ('valid_samples', 0),
                                       ('knee_clipped_fraction', {'left': .96, 'right': .98})])
def test_gate_rejects_stalling_missing_samples_and_saturation(field, value):
    report = good_report()
    assert tracking_gate(report, 'locomotion')['passed']
    row = next(r for r in report['profiles'][0]['scenarios'] if r['scenario'] == 'forward_030')
    row[field] = value
    assert not tracking_gate(report, 'locomotion')['passed']


def test_stage_pass_does_not_claim_full_task_success():
    report = good_report()
    row = next(r for r in report['profiles'][0]['scenarios'] if r['scenario'] == 'squat_050')
    row['mae'][3] = .3
    assert tracking_gate(report, 'locomotion')['passed']
    report['profiles'][0]['settings']['stage'] = 'combined'
    assert not tracking_gate(report, 'combined')['passed']


def test_promotion_requires_matching_checkpoint_and_previous_skill():
    report = good_report()
    report['checkpoint_sha256'] = 'exact-checkpoint'
    assert validate_promotion(report, 'locomotion', 'pose', 'exact-checkpoint')['passed']
    with pytest.raises(ValueError, match='match'):
        validate_promotion(report, 'locomotion', 'pose', 'another-checkpoint')
    with pytest.raises(ValueError, match='one evaluated stage'):
        validate_promotion(report, 'locomotion', 'combined', 'exact-checkpoint')
    report['profiles'][0]['scenarios'][0]['survived'] = 0
    with pytest.raises(ValueError, match='acceptance failed'):
        validate_promotion(report, 'locomotion', 'pose', 'exact-checkpoint')
