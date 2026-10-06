"""Visible-terminal fresh training with persistent logs and periodic frozen evaluations."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

from run_focused_validation import ROOT, archive_sources, now, read, sha, write
from scratch_training_gates import tracking_gate, full_tracking_gate


def health(path, initial=False, std_bounds=(.1, .6)):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    def finite(x):
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).all())
        if isinstance(x, dict):
            return all(finite(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return all(finite(v) for v in x)
        return not isinstance(x, float) or math.isfinite(x)
    assert finite(saved), 'Non-finite checkpoint or optimizer'
    groups = saved['optimizer_state_dict']['param_groups']
    assert all(1e-5 <= g['lr'] <= 3e-4 for g in groups)
    std = saved['model_state_dict']['std']
    assert float(std.min()) >= std_bounds[0]-1e-6 and float(std.max()) <= std_bounds[1]+1e-6
    assert saved['model_state_dict']['actor.0.weight'].shape[1] == 840
    if initial:
        assert saved['iter'] == 0 and not saved['optimizer_state_dict']['state']
        assert saved['infos']['from_scratch'] and saved['infos']['before_first_update']
    return dict(iteration=saved['iter'], sha256=sha(path), finite=True,
                optimizer_state_entries=len(saved['optimizer_state_dict']['state']),
                learning_rate=groups[0]['lr'], std_min=float(std.min()), std_max=float(std.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_iterations', type=int, default=5000)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.max_iterations <= 0:
        raise ValueError('max_iterations must be positive')
    # Exclusive launch marker prevents accidental duplicate terminal launches.
    with (out/'launch.lock').open('x') as f:
        f.write(str(os.getpid()))
    archive_sources(out)
    name = out.name + '_4096_fresh'
    state = dict(status='starting', started_at=now(), supervisor_pid=os.getpid(),
                 task='walk_elf3_tang_bxi_scratch', from_scratch=True, resumed_checkpoint=None,
                 teacher_checkpoint=None, num_envs=4096, seed=42, planned_updates=args.max_iterations,
                 curriculum_stage='locomotion', stage_best_checkpoint=None, stage_best_score=None,
                 run_name=name, last_logged_iteration=None, latest_checkpoint=None,
                 best_checkpoint=None, best_score=None, child_pid=None, eval_pid=None,
                 evaluations=[], save_interval=100, evaluation_interval=500,
                 console=str(out/'train.log'), source_archive=str(out/'source.zip'))
    mutex = threading.RLock()
    child = evaluator = None
    eval_stream = None
    def persist():
        with mutex:
            state['updated_at'] = now()
            write(out/'state.json', state)
            write(ROOT/'logs/training_launches/latest_scratch_bxi31.json', state)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    argv = [sys.executable, '-u', 'legged_lab/scripts/train.py', '--task', state['task'],
            '--from_scratch', '--num_envs', '4096', '--max_iterations', str(args.max_iterations),
            '--seed', '42', '--headless', '--device', 'cuda:0', '--logger', 'tensorboard', '--run_name', name]
    state['argv'] = argv
    next_evaluation = 500
    evaluation_checkpoint = None
    evaluation_kind = None
    checked = set()
    seen_initial = False
    log_root = ROOT/'logs/walk_tang_bxi31_scratch'
    completed = set()
    try:
        persist()
        print('开始从零训练：4096 环境，随机策略和空优化器，无旧模型/教师。', flush=True)
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        state.update(status='training', child_pid=child.pid)
        persist()

        def stream_training():
            with (out/'train.log').open('w') as stream:
                for line in child.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end='', flush=True)
                    m = re.search(r'Learning iteration\s+(\d+)', line)
                    if m:
                        with mutex:
                            state['last_logged_iteration'] = int(m[1])
        reader = threading.Thread(target=stream_training, daemon=True)
        reader.start()
        while True:
            code = child.poll()
            if code is not None and code != 0:
                raise RuntimeError(f'Training exited {code}; see train.log')
            runs = list(log_root.glob('*_' + name))
            if len(runs) > 1:
                raise RuntimeError('Ambiguous run directory')
            checkpoints = []
            if runs:
                run = runs[0]
                state['run_dir'] = str(run)
                if not seen_initial and (run/'fresh_start.json').is_file():
                    info = read(run/'fresh_start.json')
                    assert info['from_scratch'] and info['checkpoint_loaded'] is None
                    assert info['num_envs'] == 4096 and info['actor_observation_dim'] == 840
                    state['fresh_start'] = info
                    state['initial_health'] = health(run/'initial_policy.pt', initial=True)
                    seen_initial = True
                checkpoints = sorted((p for p in run.glob('model_*.pt')
                                      if re.fullmatch(r'model_\d+\.pt', p.name)
                                      and time.time()-p.stat().st_mtime > 3),
                                     key=lambda p: int(p.stem.split('_')[-1]))
                for p in checkpoints:
                    if str(p) not in checked:
                        state['latest_health'] = health(p)
                        state['latest_checkpoint'] = str(p)
                        checked.add(str(p))
            if evaluator is not None and evaluator.poll() is not None:
                eval_stream.close()
                eval_stream = None
                state['eval_pid'] = None
                if evaluator.returncode:
                    # An evaluation tool error is surfaced; it does not justify losing a fresh run.
                    state.setdefault('evaluation_errors', []).append(
                        dict(checkpoint=str(evaluation_checkpoint), kind=evaluation_kind,
                             returncode=evaluator.returncode, at=now()))
                    completed.add(str(evaluation_checkpoint))
                    evaluation_checkpoint = None
                elif evaluation_kind == 'walk':
                    evaluation_kind = 'standing'
                else:
                    dest = out/'evaluations'/evaluation_checkpoint.stem
                    walk, stand = read(dest/'walk/report.json'), read(dest/'standing/report.json')
                    counts = {p['settings']['name']: sum(r['survived'] for r in p['scenarios'])
                              for p in walk['profiles']}
                    standing = {r['profile']: {k: r[k] for k in ('survived', 'drift_survivors_mean_m',
                                'drift_survivors_p95_m', 'single_support_fraction')} for r in stand['profiles']}
                    amass = standing['amass']
                    eligible = (amass['survived'] >= 63 and amass['drift_survivors_mean_m'] is not None
                                and amass['drift_survivors_mean_m'] <= .05
                                and amass['drift_survivors_p95_m'] <= .15
                                and amass['single_support_fraction'] <= .03
                                and min(counts.values()) >= 539)
                    stage_gate = tracking_gate(walk, state['curriculum_stage'])
                    full_gate = full_tracking_gate(walk)
                    eligible = eligible and full_gate['passed']
                    # Unit-normalized ranking only AFTER every hard task gate passes.
                    score = full_gate['profiles']['controlled_arms_on']['score']
                    summary = dict(checkpoint=str(evaluation_checkpoint), completed_at=now(),
                                   full_task_survival=counts, standing=standing,
                                   eligible_for_best=eligible, tracking_score=score,
                                   stage_gate=stage_gate, full_tracking_gate=full_gate,
                                   report_dir=str(dest))
                    state['evaluations'].append(summary)
                    write(dest/'summary.json', summary)
                    if eligible and (state['best_score'] is None or score < state['best_score']):
                        state.update(best_checkpoint=str(evaluation_checkpoint), best_score=score)
                    if stage_gate['passed'] and (state['stage_best_score'] is None or
                                                 stage_gate['score'] < state['stage_best_score']):
                        state.update(stage_best_checkpoint=str(evaluation_checkpoint), stage_best_score=stage_gate['score'])
                    print('\n[冻结评估完成]', json.dumps(summary, ensure_ascii=False), flush=True)
                    completed.add(str(evaluation_checkpoint))
                    evaluation_checkpoint = None
                evaluator = None
            if evaluator is None:
                if evaluation_checkpoint is None:
                    candidate = next((p for p in checkpoints if int(p.stem.split('_')[-1]) >= next_evaluation), None)
                    if candidate is None and code == 0 and checkpoints:
                        candidate = checkpoints[-1]
                    if candidate is not None and str(candidate) not in completed:
                        evaluation_checkpoint = candidate
                        evaluation_kind = 'walk'
                        next_evaluation = (int(candidate.stem.split('_')[-1])//500+1)*500
                if evaluation_checkpoint is not None:
                    dest = out/'evaluations'/evaluation_checkpoint.stem
                    dest.mkdir(parents=True, exist_ok=True)
                    script = 'evaluate_walk_tang.py' if evaluation_kind == 'walk' else 'evaluate_tang_standing.py'
                    command = [sys.executable, '-u', 'tools/'+script, '--task', state['task'],
                               '--checkpoint', str(evaluation_checkpoint), '--output', str(dest/evaluation_kind),
                               '--arm-guard', '--include-guarded-neutral', '--headless']
                    if evaluation_kind == 'walk':
                        command += ['--curriculum-stage', state['curriculum_stage']]
                    eval_stream = (dest/(evaluation_kind+'.log')).open('w')
                    evaluator = subprocess.Popen(command, cwd=ROOT, env=env, stdout=eval_stream, stderr=subprocess.STDOUT)
                    state.update(eval_pid=evaluator.pid, evaluating_checkpoint=str(evaluation_checkpoint),
                                 evaluation_kind=evaluation_kind)
            if code == 0:
                state['child_pid'] = None
                state['status'] = 'final_evaluation'
                if checkpoints and str(checkpoints[-1]) in completed and evaluator is None:
                    assert seen_initial
                    reader.join(timeout=5)
                    state.update(status='completed', completed_at=now(), completed_updates=args.max_iterations)
                    persist()
                    return
            persist()
            time.sleep(5)
    except BaseException as exc:
        for process in (child, evaluator):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        if eval_stream is not None:
            eval_stream.close()
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     stop_reason=str(exc), traceback=traceback.format_exc(), child_pid=None, eval_pid=None)
        persist()
        raise


if __name__ == '__main__':
    main()
