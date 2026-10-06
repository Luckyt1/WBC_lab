"""A controller handoff must observe, never respawn, the existing trainer."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.adopt_training_process import AdoptedTrainingProcess, process_identity


@pytest.fixture
def active_segment(tmp_path):
    # Harmless stand-in with a supervisor-shaped argv, no simulator or CUDA.
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',
                            'tools/run_walk_robust_training.py','--output',str(tmp_path)])
    state=dict(status='training',supervisor_pid=child.pid,child_pid=child.pid,eval_pid=None)
    path=tmp_path/'state.json';path.write_text(json.dumps(state))
    try:
        yield tmp_path,child,state
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_adoption_keeps_process_and_logs_and_waits_for_actual_exit(active_segment):
    out,proc,state=active_segment
    log=out/'supervisor.log';log.write_text('preserved output\n')
    identity=process_identity(proc.pid)
    handle=AdoptedTrainingProcess(out)
    assert handle.pid==proc.pid and handle.poll() is None
    assert process_identity(proc.pid)==identity and log.read_text()=='preserved output\n'
    state.update(status='completed',child_pid=None)
    (out/'state.json').write_text(json.dumps(state))
    assert handle.poll() is None  # Do not overlap evaluation with a live trainer.
    proc.terminate();proc.wait(timeout=5)
    assert handle.wait(timeout=1)==0


def test_supervisor_exit_without_completed_state_is_failure(active_segment):
    out,proc,state=active_segment
    handle=AdoptedTrainingProcess(out)
    proc.terminate();proc.wait(timeout=5)
    assert handle.poll()==1


def test_pid_reuse_is_rejected(active_segment,monkeypatch):
    out,proc,state=active_segment
    handle=AdoptedTrainingProcess(out)
    monkeypatch.setattr('tools.adopt_training_process.process_identity',lambda pid:'other_start_time')
    with pytest.raises(RuntimeError,match='reused'):handle.poll()


def test_cannot_adopt_evaluation_or_wrong_output(active_segment):
    out,proc,state=active_segment
    state['eval_pid']=123
    (out/'state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError):AdoptedTrainingProcess(out)
    other=out/'other';other.mkdir();state['eval_pid']=None
    (other/'state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='command'):AdoptedTrainingProcess(other)
