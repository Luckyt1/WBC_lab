"""Guard the requested locomotion-only scope and acceptance against regressions."""
import ast
import copy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch
from test_tang_commands import commands

ROOT = Path(__file__).resolve().parents[1]
curriculum = importlib.import_module('tang_cpu.walk_robust_curriculum')
sys.path.insert(0, str(ROOT/'tools'))
from walk_robust_gates import TASKS, PROFILES, evaluate_gates


def test_pose_tasks_cannot_return_at_late_iterations_or_manual_override():
    torch.manual_seed(27)
    cmd = commands.TangCommands(curriculum.walk_commands(), 10000, 'cpu')
    ids = torch.arange(cmd.num_envs)
    for age in (0, 10000, 50000, 1000000):
        cmd.iteration = age
        cmd.reset(ids)
        for _ in range(3):
            cmd.compute(15.)
            assert (cmd.full_command[:, 3] == .8235).all()
            assert (cmd.full_command[:, 4] == 0).all()
            assert not (cmd.sampled_mode == 4).any()
        for mode, fraction in [(0, .15), (1, .15), (2, .55), (3, .15)]:
            assert abs(float((cmd.sampled_mode == mode).float().mean()) - fraction) < .025
        moving = cmd.sampled_mode == 2
        assert .45 < float((cmd.target[moving, 0] > 0).float().mean()) < .55
        assert cmd.target[moving, 0].abs().min() >= .2
    cmd.set_manual_command([.3, 0, .5, .5, .6])
    torch.testing.assert_close(cmd.target[0], torch.tensor([.3, 0, .5, .8235, 0]))
    assert all(c[3:] == [.8235, 0.] for _, c in curriculum.SCENARIOS)


def test_actual_environment_disturbance_schedule_reaches_full_strength():
    cfg = NS(arm_motion=NS(), noise=NS(), push_velocity_curriculum=NS(), reset_velocity_curriculum=NS())
    curriculum.configure_walk_robust(cfg)
    path = ROOT/'legged_lab/envs/elf3_tang/elf3_env.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Elf3Env')
    names = ('_get_push_velocity_curriculum_progress', '_get_reset_velocity_curriculum_progress')
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    env = NS(cfg=cfg, learning_iteration=0, learning_iteration_dt=24*.02, physics_dt=.005)
    for name, finish in zip(names, (20000, 30000)):
        for age, expected in [(0, 0.), (finish//2, .5), (finish, 1.), (50000, 1.)]:
            env.learning_iteration = age
            assert abs(namespace[name](env) - expected) < 1.e-10


def valid_report():
    return dict(repeats=32, warmup_s=4., measure_s=12., arm_guard_enabled=True,
                profiles=[dict(settings=dict(name=p), scenarios=[dict(scenario=n, trials=32, survived=32,
                    valid_samples=19200, mae=[0., 0., 0., .2, .5],
                    knee_clipped_fraction={'left': 0., 'right': 0.}) for n in TASKS]) for p in PROFILES])


def test_walk_gate_requires_motion_and_disturbance_success_not_pose_tracking():
    report = valid_report()
    assert evaluate_gates(report)['passed']
    frozen = copy.deepcopy(report)
    frozen['profiles'][2]['scenarios'][1]['mae'][0] = .3
    gates = evaluate_gates(frozen)
    assert not gates['passed']
    assert 'forward_030' in gates['profiles']['training_disturbances']['failures']
    assert not evaluate_gates(dict(report, profiles=report['profiles'][:2]))['passed']
    assert not evaluate_gates(dict(report, repeats=2))['passed']


def test_mixed_benchmark_requires_tracking_small_yaw_and_keeps_original_gates():
    report = valid_report()
    tasks = ('arc_forward_pos', 'arc_forward_neg', 'arc_backward_pos', 'arc_backward_neg')
    for profile in report['profiles']:
        for task in tasks:
            row = copy.deepcopy(profile['scenarios'][0])
            row.update(scenario=task, mae=[.02, .02, .03, 0., 0.])
            profile['scenarios'].append(row)
    assert evaluate_gates(report)['passed']
    assert evaluate_gates(report, tasks, (.08, .08, .05))['passed']
    report['profiles'][0]['scenarios'][-1]['mae'][2] = .1
    assert not evaluate_gates(report, tasks, (.08, .08, .05))['passed']
    assert evaluate_gates(report)['passed']
