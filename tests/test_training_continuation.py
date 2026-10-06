"""Continuation boundaries must not reset training or start a duplicate run."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'training_continuation', Path(__file__).resolve().parents[1]/'tools/continue_dynamics_training.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def previous():
    return dict(task='walk_elf3_tang_bxi_robust', auxiliary_supervision='dynamics', seed=42,
                status='completed', target_total_iterations=40000, run_dir='/tmp/original_run',
                num_envs=4096, evaluation_interval=5000, save_interval=1000,
                teleop_standing=True, supervisor_pid=123, child_pid=None, eval_pid=None)


@pytest.mark.parametrize('completed,target', [(40000,60000), (60000,80000)])
def test_exact_remaining_updates_and_recipe(completed,target):
    state = previous()
    state['target_total_iterations'] = completed
    checkpoint = f'/tmp/original_run/model_{completed-1}.pt'
    argv = module.continuation_command(state, checkpoint,
                                       completed, target, '/tmp/continuation', '/python', '/deployment_python')
    assert argv[argv.index('--max_iterations')+1] == '20000'
    assert argv[argv.index('--resume_checkpoint')+1] == checkpoint
    assert argv[argv.index('--save_interval')+1] == '1000'
    assert argv[argv.index('--evaluation_interval')+1] == '5000'
    assert '--teleop_standing' in argv and '--from_scratch' not in argv


def test_height_adaptation_survives_later_continuations():
    state = previous()
    state['pelvis_height'] = .78
    argv = module.continuation_command(state, '/tmp/original_run/model_39999.pt',
                                      40000, 60000, '/tmp/out', '/python', '/deployment')
    assert argv[argv.index('--pelvis_height')+1] == '0.78'


@pytest.mark.parametrize('status', ['training', 'interrupted', 'failed'])
def test_command_rejects_unfinished_predecessor(status):
    state = previous(); state['status'] = status
    with pytest.raises(ValueError):
        module.continuation_command(state, '/tmp/original_run/model_39999.pt',
                                    40000, 60000, '/tmp/out', '/python', '/deployment')


@pytest.mark.parametrize('completed,target', [(39999,60000), (40000,40000), (40000,30000)])
def test_checkpoint_count_and_total_are_checked(completed,target):
    with pytest.raises(ValueError):
        module.continuation_command(previous(), '/tmp/original_run/model_39999.pt',
                                    completed, target, '/tmp/out', '/python', '/deployment')


def test_readiness_requires_completion_and_valid_identity():
    state = previous()
    assert module.predecessor_ready(state, 123, 0, 500)
    state['status'] = 'final_evaluation'
    assert not module.predecessor_ready(state, 123, 490, 500)
    with pytest.raises(RuntimeError):
        module.predecessor_ready(state, 123, 0, 500)
    state['status'] = 'interrupted'
    with pytest.raises(RuntimeError):
        module.predecessor_ready(state, 123, 490, 500)
    state['status'] = 'completed'
    with pytest.raises(RuntimeError):
        module.predecessor_ready(state, 456, 490, 500)
    state['eval_pid'] = 789
    with pytest.raises(RuntimeError):
        module.predecessor_ready(state, 123, 490, 500)
