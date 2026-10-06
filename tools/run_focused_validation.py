"""Run four 500-update training stages, with frozen evaluations and retention gates.

This continuation never extends beyond 2,000 updates or promotes a regressed
checkpoint. Training rewards are not used as evidence of task completion.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'logs/walk_tang_bxi31_focused'


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reports(walk, stand):
    assert walk['arm_guard_enabled'] and stand['arm_guard_enabled']
    assert len(walk['profiles']) == 3 and len(stand['profiles']) == 2
    assert all(len(p['scenarios']) == 17 for p in walk['profiles'])
    assert walk['repeats'] == 32 and stand['num_envs'] == 64
    assert walk['checkpoint_sha256'] == stand['checkpoint_sha256']
    assert all(r['valid_samples'] > 0 and all(math.isfinite(x) for x in r['mae'])
               for p in walk['profiles'] for r in p['scenarios'])
    assert all(r['drift_survivors_mean_m'] is not None for r in stand['profiles'])


def compare(base_walk, base_stand, walk, stand):
    """Predeclared regression limits, plus stricter targeted skill goals."""
    validate_reports(walk, stand)
    assert walk['seed'] == base_walk['seed'] and stand['seed'] == base_stand['seed']
    failures, observations = [], {}
    for before, after in zip(base_walk['profiles'], walk['profiles']):
        name = before['settings']['name']
        assert after['settings'] == before['settings']
        assert before['initial_clip_sha256'] == after['initial_clip_sha256']
        b = {r['scenario']: r for r in before['scenarios']}
        a = {r['scenario']: r for r in after['scenarios']}
        survival = sum(r['survived'] for r in a.values())
        observations[name + '/survived'] = survival
        if survival < sum(r['survived'] for r in b.values()) - 5:
            failures.append(name + ': lost more than 5 of 544 survivors')
        for scenario, axis, tolerance in (
                ('stand', 3, .015), ('forward_030', 0, .05), ('forward_060', 0, .07),
                ('backward_030', 0, .05), ('turn_pos', 2, .10), ('turn_neg', 2, .10)):
            if a[scenario]['survived'] < b[scenario]['survived'] - 2:
                failures.append(name + '/' + scenario + ': survival regression')
            if a[scenario]['mae'][axis] > b[scenario]['mae'][axis] + tolerance:
                failures.append(name + '/' + scenario + ': tracking regression')
    for before, after in zip(base_stand['profiles'], stand['profiles']):
        name = before['profile']
        assert name == after['profile']
        if after['survived'] < before['survived'] - 1:
            failures.append(name + ': lost more than 1 of 64 standing survivors')
        for key, tolerance, floor in (
                ('drift_survivors_mean_m', .025, .05),
                ('drift_survivors_p95_m', .10, .15),
                ('height_mae_m', .015, .03),
                ('single_support_fraction', .02, .03)):
            if after[key] > max(floor, before[key] + tolerance):
                failures.append(name + '/' + key + ': standing regression')
    rows = {r['scenario']: r for r in walk['profiles'][0]['scenarios']}
    goals, score = {}, 0.
    for name in ('lateral_pos', 'lateral_neg'):
        r = rows[name]
        goals[name] = r['survived'] >= 31 and r['mae'][1] <= .10 and abs(r['actual_mean'][1]) >= .14
        score += r['mae'][1] / .2
    for name in ('crouch_walk_yaw_pos', 'crouch_walk_yaw_neg'):
        r = rows[name]
        goals[name] = (r['survived'] >= 31 and r['actual_mean'][0] >= .2
                       and r['mae'][3] <= .05 and r['mae'][4] <= .15)
        score += r['mae'][0] / .3 + r['mae'][3] / .05 + r['mae'][4] / .15
    return dict(retention_passed=not failures, failures=failures,
                observations=observations, targeted_score=score,
                targeted_goals=goals, targeted_goals_passed=all(goals.values()))


def checkpoint_health(path):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(v) for v in value)
        if isinstance(value, float):
            return math.isfinite(value)
        return True

    assert finite(saved['model_state_dict']), 'Non-finite model'
    assert finite(saved['optimizer_state_dict']), 'Non-finite optimizer'
    std = saved['model_state_dict']['std']
    assert float(std.min()) >= .1 - 1e-6 and float(std.max()) <= .3 + 1e-6
    return dict(iteration=saved['iter'], sha256=sha(path), model_finite=True,
                optimizer_finite=True, std_min=float(std.min()), std_max=float(std.max()))


def archive_sources(out):
    sources = sorted(set(p for folder in ('legged_lab', 'rsl_rl/rsl_rl', 'tools', 'tests')
                         for p in (ROOT / folder).rglob('*')
                         if p.is_file() and p.suffix in ('.py', '.json', '.urdf') and '__pycache__' not in p.parts))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    with zipfile.ZipFile(out / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sources:
            archive.write(p, str(p.relative_to(ROOT)))
    write(out / 'source_sha256.json', hashes)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    base_walk, base_stand = read(out/'baseline/walk/report.json'), read(out/'baseline/standing/report.json')
    validate_reports(base_walk, base_stand)
    assert sha(checkpoint) == base_walk['checkpoint_sha256']
    base_gate = compare(base_walk, base_stand, base_walk, base_stand)
    health = checkpoint_health(checkpoint)
    if args.audit_only:
        print(json.dumps({'checkpoint': health, 'baseline_gate': base_gate}, indent=2))
        return
    state_path = out / 'state.json'
    if state_path.exists():
        raise RuntimeError('An execution already exists here; use a fresh output directory.')
    sources = archive_sources(out)
    state = dict(status='starting', started_at=now(), supervisor_pid=os.getpid(),
                 task='walk_elf3_tang_bxi_focus', num_envs=4096,
                 planned_updates=2000, updates_per_stage=500, completed_updates=0,
                 baseline_checkpoint=str(checkpoint), baseline_health=health,
                 source_archive=str(out/'source.zip'), source_hashes=str(out/'source_sha256.json'),
                 best_checkpoint=str(checkpoint), best_targeted_score=base_gate['targeted_score'],
                 stages=[], baseline_comparison=base_gate)
    child = None

    def persist():
        state['updated_at'] = now()
        write(state_path, state)
        write(ROOT/'logs/training_launches/latest_focused_bxi31.json', state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def run(argv, log, phase, stage, timeout_s):
        nonlocal child
        state.update(status=phase, active_stage=stage, active_command=argv,
                     console=str(log), phase_started_at=now())
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with Path(log).open('w') as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            persist()
            deadline = time.monotonic() + timeout_s
            while child.poll() is None:
                if time.monotonic() > deadline:
                    raise RuntimeError(f'Timeout in {phase}')
                if phase == 'training':
                    with Path(log).open('rb') as current:
                        current.seek(max(0, Path(log).stat().st_size - 32000))
                        recent = current.read().decode(errors='replace')
                    iterations = re.findall(r'Learning iteration\s+(\d+)', recent)
                    if iterations:
                        state['last_logged_iteration'] = int(iterations[-1])
                    dirs = list(EXPERIMENT.glob('*_' + state['active_run_name']))
                    if len(dirs) == 1:
                        state['run_dir'] = str(dirs[0])
                        models = list(dirs[0].glob('model_*.pt'))
                        if models:
                            state['latest_checkpoint'] = str(max(models, key=lambda p: int(p.stem.split('_')[-1])))
                persist()
                time.sleep(5)
            code = child.returncode
            child = None
            state['child_pid'] = None
            if code:
                raise RuntimeError(f'{phase} exited {code}; see {log}')

    try:
        persist()
        warm = EXPERIMENT / ('warm_start_' + str(health['iteration']))
        warm.mkdir(parents=True, exist_ok=True)
        destination = warm / checkpoint.name
        if destination.exists():
            assert sha(destination) == health['sha256']
        else:
            shutil.copy2(checkpoint, destination)
        checkpoint = destination
        for stage in range(1, 5):
            assert all(sha(ROOT/p) == digest for p, digest in sources.items()), 'Source files changed during validation'
            stage_dir = out / f'stage_{stage}'
            stage_dir.mkdir()
            name = out.name + f'_stage{stage}'
            state['active_run_name'] = name
            starting = checkpoint_health(checkpoint)
            state['stage_start_iteration'] = starting['iteration']
            argv = [sys.executable, '-u', 'legged_lab/scripts/train.py', '--task', state['task'],
                    '--num_envs', '4096', '--max_iterations', '500', '--seed', '42',
                    '--headless', '--device', 'cuda:0', '--logger', 'tensorboard',
                    '--run_name', name, '--resume', 'True',
                    '--load_run', re.escape(checkpoint.parent.name) + '$',
                    '--checkpoint', re.escape(checkpoint.name) + '$']
            run(argv, stage_dir/'train.log', 'training', stage, 6*3600)
            dirs = list(EXPERIMENT.glob('*_' + name))
            assert len(dirs) == 1
            checkpoint = max(dirs[0].glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
            health = checkpoint_health(checkpoint)
            assert health['iteration'] == starting['iteration'] + 499, 'Unexpected stage update count'
            state['completed_updates'] += 500
            state['latest_checkpoint'] = str(checkpoint)
            for kind, script, extra in (
                    ('walk', 'evaluate_walk_tang.py', ['--task', 'walk_elf3_tang_bxi']),
                    ('standing', 'evaluate_tang_standing.py', [])):
                argv = [sys.executable, '-u', 'tools/'+script, '--checkpoint', str(checkpoint),
                        '--output', str(stage_dir/kind), '--arm-guard', '--headless', *extra]
                run(argv, stage_dir/(kind+'.log'), 'evaluating_'+kind, stage, 3600)
            walk, stand = read(stage_dir/'walk/report.json'), read(stage_dir/'standing/report.json')
            gate = compare(base_walk, base_stand, walk, stand)
            record = dict(stage=stage, checkpoint=str(checkpoint), health=health, comparison=gate, completed_at=now())
            state['stages'].append(record)
            write(stage_dir/'comparison.json', record)
            if not gate['retention_passed']:
                state.update(status='stopped_regression', stop_reason=gate['failures'])
                persist()
                return
            if gate['targeted_score'] < state['best_targeted_score']:
                state.update(best_checkpoint=str(checkpoint), best_targeted_score=gate['targeted_score'])
            persist()
        state.update(status='completed_validation', completed_at=now(),
                     targeted_goals_passed=gate['targeted_goals_passed'],
                     note='No automatic long continuation. Inspect frozen evaluations before further training.')
        persist()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     stop_reason=str(exc), child_pid=None, traceback=traceback.format_exc())
        persist()
        raise


if __name__ == '__main__':
    main()
