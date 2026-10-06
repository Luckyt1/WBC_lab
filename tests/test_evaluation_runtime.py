"""A venv's symlink entrypoint must survive evaluation launch and recovery."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    'evaluation_runtime', Path(__file__).resolve().parents[1]/'tools/evaluation_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_venv_entrypoint_and_run_local_override(tmp_path):
    venv = tmp_path/'venv/bin/python'
    venv.parent.mkdir(parents=True)
    venv.symlink_to(sys.executable)
    out = tmp_path/'run/evaluations/model_50000'
    out.mkdir(parents=True)
    assert runtime.deployment_python(venv, out) == venv
    assert runtime.deployment_python(venv, out) != venv.resolve()
    (out.parent.parent/'evaluation_runtime.json').write_text(json.dumps({'deployment_python':str(venv)}))
    assert runtime.deployment_python(venv.resolve(), out) == venv


def test_recovery_rejects_wrong_checkpoint_and_missing_artifacts(tmp_path):
    checkpoint = tmp_path/'checkpoint.pt'; checkpoint.write_bytes(b'checkpoint')
    scenarios = ['stand','forward_030','forward_060','backward_030','lateral_pos','lateral_neg','turn_pos','turn_neg']
    names = ['bootstrap','controlled_arms_on','training_disturbances']
    report = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  task='walk_elf3_tang_bxi_robust', arm_guard_enabled=True,
                  repeats=32, measure_s=12, warmup_s=4, gait_diagnostics=True, auxiliary_probe=True,
                  profiles=[dict(settings={'name':name},scenarios=[{'scenario':s} for s in scenarios]) for name in names])
    (tmp_path/'report.json').write_text(json.dumps(report))
    for name in names:
        (tmp_path/f'{name}.npz').touch()
        (tmp_path/f'{name}_auxiliary_probe.npz').touch()
    runtime.validate_saved_isaac(checkpoint,tmp_path,report['task'],gait=True,auxiliary=True)
    (tmp_path/'params').mkdir()
    (tmp_path/'params/env.yaml').write_text('commands:\n  nominal_height: 0.78\n  height_initial_range: [0.78, 0.78]\n  height_range: [0.78, 0.78]\n  sampling_height_range: [0.78, 0.78]\n')
    with pytest.raises(ValueError, match='height command'):
        runtime.validate_saved_isaac(checkpoint,tmp_path,report['task'])
    report['height_command'] = .78
    (tmp_path/'report.json').write_text(json.dumps(report))
    runtime.validate_saved_isaac(checkpoint,tmp_path,report['task'])
    with pytest.raises(ValueError):
        runtime.validate_saved_isaac(checkpoint,tmp_path,report['task'],mixed=True)
    checkpoint.write_bytes(b'different')
    with pytest.raises(ValueError):
        runtime.validate_saved_isaac(checkpoint,tmp_path,report['task'])
    checkpoint.write_bytes(b'checkpoint')
    (tmp_path/'bootstrap_auxiliary_probe.npz').unlink()
    with pytest.raises(FileNotFoundError):
        runtime.validate_saved_isaac(checkpoint,tmp_path,report['task'],auxiliary=True)
