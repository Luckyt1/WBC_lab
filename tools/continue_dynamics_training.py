"""Continue an authorized dynamics run after its current training/evaluation finishes.

The predecessor is never interrupted. Failed/interrupted/stale predecessors do
not trigger training. Both optimizers and auxiliary heads are restored by the
existing strict continuation audit in train.py.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def continuation_command(previous, checkpoint, completed, target, output, python, deployment_python):
    if (previous['task'] != 'walk_elf3_tang_bxi_robust'
            or previous.get('auxiliary_supervision') != 'dynamics'
            or previous.get('seed') != 42):
        raise ValueError('Unsupported predecessor recipe')
    if previous['status'] != 'completed' or completed != previous['target_total_iterations']:
        raise ValueError('Predecessor must finish its exact planned update count')
    if target <= completed:
        raise ValueError('Target must exceed completed updates')
    if Path(checkpoint).resolve().parent != Path(previous['run_dir']).resolve():
        raise ValueError('Checkpoint must belong to the predecessor run')
    argv = [str(python), '-u', 'tools/run_walk_robust_training.py',
            '--output', str(output), '--task', previous['task'],
            '--auxiliary_supervision', 'dynamics', '--resume_checkpoint', str(checkpoint),
            '--max_iterations', str(target-completed), '--num_envs', str(previous['num_envs']),
            '--evaluation_interval', str(previous['evaluation_interval']),
            '--save_interval', str(previous['save_interval']),
            '--deployment_python', str(deployment_python)]
    if previous.get('teleop_standing'):
        argv.append('--teleop_standing')
    elif previous.get('standing_position_penalty'):
        argv.append('--standing_position_penalty')
    if previous.get('pelvis_height') is not None:
        argv.extend(['--pelvis_height', str(previous['pelvis_height'])])
    if previous.get('elf3_symmetry'):
        argv.append('--elf3_symmetry')
    return argv


def predecessor_ready(state, expected_pid, modified_at, now):
    if state['supervisor_pid'] != expected_pid:
        raise RuntimeError('Predecessor identity changed')
    if state['status'] == 'completed':
        if state.get('child_pid') is not None or state.get('eval_pid') is not None:
            raise RuntimeError('Completed predecessor still reports active children')
        return True
    if state['status'] not in ('starting', 'training', 'final_evaluation'):
        raise RuntimeError(f"Predecessor stopped: {state['status']}")
    if now-modified_at > 300:
        raise RuntimeError('Predecessor heartbeat is stale; refusing automatic launch')
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-total-iterations', type=int, required=True)
    parser.add_argument('--deployment-python', type=Path, required=True)
    args = parser.parse_args()
    source = args.previous_output.resolve()/'state.json'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lease = (out/'continuation.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A launch attempt is never silently repeated, including after worker restart.
    if (out/'launch.lock').exists() or (out/'state.json').exists():
        raise RuntimeError('Continuation already launched; inspect its state instead')
    initial = json.loads(source.read_text())
    if args.target_total_iterations <= initial['target_total_iterations']:
        raise ValueError('Requested total must extend the predecessor target')
    record = dict(status='waiting_for_predecessor', worker_pid=os.getpid(),
                  previous_output=str(source.parent), previous_supervisor_pid=initial['supervisor_pid'],
                  previous_target_total_iterations=initial['target_total_iterations'],
                  target_total_iterations=args.target_total_iterations,
                  output=str(out), requested_at=datetime.now().astimezone().isoformat(),
                  method='Wait for normal completion including evaluation; restore all learned state with recipe audit',
                  post_training_arm_speed_evaluation='Managed separately; this worker only continues training')

    def persist():
        record['updated_at'] = datetime.now().astimezone().isoformat()
        temporary = out/'continuation.json.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(out/'continuation.json')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        persist()
        while True:
            previous = json.loads(source.read_text())
            if previous['target_total_iterations'] != initial['target_total_iterations']:
                raise RuntimeError('Predecessor target changed; continuation needs reconciliation')
            record['predecessor_status'] = previous['status']
            record['predecessor_completed_updates'] = previous.get('total_completed_updates')
            if predecessor_ready(previous, initial['supervisor_pid'], source.stat().st_mtime, time.time()):
                break
            persist()
            time.sleep(15)

        import torch
        from audit_auxiliary_resume import completed_updates
        checkpoint = Path(previous['latest_checkpoint']).resolve(strict=True)
        completed = completed_updates(torch.load(checkpoint, map_location='cpu', weights_only=False))
        argv = continuation_command(previous, checkpoint, completed, args.target_total_iterations,
                                    out, sys.executable, args.deployment_python.absolute())
        record.update(status='launching_continuation', checkpoint=str(checkpoint),
                      completed_updates_before_resume=completed,
                      planned_additional_updates=args.target_total_iterations-completed, argv=argv)
        persist()
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (out/'supervisor.log').open('x') as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(status='continuation_running', continuation_supervisor_pid=child.pid)
        while child.poll() is None:
            if (out/'state.json').exists():
                current = json.loads((out/'state.json').read_text())
                record['continuation_status'] = current['status']
                record['total_completed_updates'] = current.get('total_completed_updates')
                record['resume_audit'] = current.get('resume_audit')
            persist()
            time.sleep(15)
        current = json.loads((out/'state.json').read_text()) if (out/'state.json').exists() else {}
        if child.returncode != 0 or current.get('status') != 'completed':
            raise RuntimeError(f'Continuation exited {child.returncode}; see supervisor.log and state.json')
        record.update(status='completed', total_completed_updates=current['total_completed_updates'])
        persist()
    except BaseException as exc:
        record.update(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=str(exc), traceback=traceback.format_exc())
        persist()
        raise


if __name__ == '__main__':
    main()
