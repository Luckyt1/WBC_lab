"""Finish the existing study, validate cold-start execution, then run three fresh arms."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from run_focused_validation import ROOT, archive_sources, now, read, write
from run_scratch_training import health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for', type=Path, required=True, help='Existing study state.json')
    parser.add_argument('--output', type=Path, required=True, help='New queue directory')
    parser.add_argument('--deployment-python', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    archive_sources(out)
    state = dict(status='waiting_for_previous_study', started_at=now(), supervisor_pid=os.getpid(),
                 child_pid=None, wait_for=str(args.wait_for.resolve()), from_scratch=True,
                 planned_variants=['control','state','dynamics'], updates_per_variant=5000,
                 num_envs=4096, seed=42, checkpoint_loaded=None, teacher_checkpoint=None,
                 preflight_output=str(out/'preflight'), training_output=str(out/'study'), auto_promote=False)
    child = None
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')

    def persist():
        state['updated_at'] = now()
        write(out/'state.json',state)
        write(ROOT/'logs/training_launches/latest_auxiliary_scratch_queue.json',state)

    def interrupted(signum,_frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(signum,interrupted)

    def execute(name, updates, num_envs):
        nonlocal child
        argv=[sys.executable,'-u','tools/run_gait_ablation.py','--from-scratch',
              '--output',str(out/name),'--deployment-python',str(args.deployment_python),
              '--updates',str(updates),'--save-interval','1000','--num-envs',str(num_envs),
              '--seed','42','--variants','control','state','dynamics','--extended-gait','--auxiliary-study']
        state.update(status='gpu_preflight' if name=='preflight' else 'running_study',command=argv)
        with (out/(name+'.log')).open('x') as stream:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
            state['child_pid']=child.pid
            persist()
            while child.poll() is None:
                current=out/name/'state.json'
                if current.exists():
                    progress=read(current)
                    state['progress']={key:progress.get(key) for key in
                                       ('status','active_variant','last_logged_iteration','updated_at')}
                persist()
                time.sleep(5)
            code=child.returncode
        child=None
        state['child_pid']=None
        persist()
        if code:
            raise RuntimeError(f'{name} exited {code}; see {out/(name+".log")}')
        result=read(out/name/'state.json')
        assert result['status']=='completed'
        assert set(result['initializations'])=={'control','state','dynamics'}
        assert len({item['policy_state_sha256'] for item in result['initializations'].values()})==1
        for row in result['variants']:
            health(Path(row['run_dir'])/f'model_{updates-1}.pt',std_bounds=(.1,.6))
        return result

    try:
        persist()
        while True:
            previous=read(args.wait_for)
            state['previous_study_status']=previous['status']
            if previous['status'] in ('completed','failed','interrupted'):
                assert previous.get('child_pid') is None
                break
            os.kill(previous['supervisor_pid'],0)
            persist()
            time.sleep(5)
        preflight=execute('preflight',3,64)
        state['preflight_passed']=True
        state['preflight_initializations']=preflight['initializations']
        persist()
        execute('study',5000,4096)
        state.update(status='completed',completed_at=now())
        persist()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            # The study supervisor propagates SIGTERM to its own simulator child.
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                state['unresponsive_child_pid']=child.pid
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                     stop_reason=str(exc),traceback=traceback.format_exc())
        persist()
        raise


if __name__=='__main__':
    main()
