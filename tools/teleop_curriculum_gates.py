"""Frozen 60-second gates and hysteresis for the standing-first curriculum."""
import hashlib
import json
import math
from pathlib import Path

from tools.teleop_curriculum import STAGES, evaluation_profiles

TASKS = ('stand', 'walk_then_stop', 'forward_030', 'forward_060', 'backward_030',
         'lateral_pos', 'lateral_neg', 'turn_pos', 'turn_neg')


def position_metrics(position, yaw, valid, first_failure, start, dt):
    import numpy as np
    keep = (first_failure < 0) & valid[start:].all(axis=0)
    xy = position[start:, keep, :2]
    if xy.shape[0] < 2 or xy.shape[1] == 0:
        return dict(valid=False, survivor_trials=int(keep.sum()))
    headings = np.unwrap(yaw[start:, keep], axis=0)
    if not (np.isfinite(xy).all() and np.isfinite(headings).all()):
        raise ValueError('Nonfinite standing trajectory')
    distance = np.linalg.norm(xy-xy[0:1], axis=-1)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1)/dt
    return dict(valid=True, survivor_trials=int(keep.sum()),
                measurement_seconds=(len(xy)-1)*dt,
                p95_max_drift_m=float(np.quantile(distance.max(axis=0), .95)),
                mean_max_drift_m=float(distance.max(axis=0).mean()),
                mean_horizontal_speed_m_s=float(speed.mean()),
                p95_max_heading_deg=float(np.quantile(
                    np.abs(headings-headings[0:1]).max(axis=0)*180/np.pi, .95)))


def finite_vector(value, size):
    return (isinstance(value, list) and len(value) == size
            and all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in value))


def gate(report, mujoco, stage, checkpoint_sha256, onnx_sha256):
    # A corrupt/mismatched report is a pipeline error, never training evidence.
    if (report.get('checkpoint_sha256') != checkpoint_sha256
            or report.get('teleop_curriculum_stage') != stage
            or report.get('repeats') != 32 or report.get('warmup_s') != 4.
            or report.get('measure_s') != 56. or not report.get('arm_guard_enabled')
            or report.get('neutral_reference_fraction_in_amass_profiles') != 0.
            or mujoco.get('onnx_sha256') != onnx_sha256
            or report.get('height_command') != mujoco.get('height_command')
            or mujoco.get('arms') != 'fixed_training_default'
            or mujoco.get('noise') is not False or mujoco.get('push') is not False):
        raise ValueError('Missing or mismatched curriculum evaluation protocol')
    expected = evaluation_profiles(stage)
    profiles = report.get('profiles', [])
    if [p['settings'] for p in profiles] != expected:
        raise ValueError('Evaluation difficulty does not match the requested stages')
    checks, failures, scores = {}, [], []
    for p in profiles:
        name = p['settings']['stage']
        rows = p.get('scenarios', [])
        if len(rows) != len(TASKS) or {r['scenario'] for r in rows} != set(TASKS):
            raise ValueError('Missing or duplicate curriculum scenarios')
        checks[name] = {}
        for r in rows:
            task = r['scenario']
            stationary = task in ('stand', 'walk_then_stop')
            limits = (.02, .02, .10, .01, .05) if stationary else (.08, .08, .15, .015, .05)
            mae = r.get('mae', [])
            valid = finite_vector(mae, 5) and r.get('valid_samples', 0) > 0
            survived = r.get('trials') == 32 and r.get('survived', 0) >= 31
            knees = r.get('knee_clipped_fraction', {})
            bounded = (len(knees) == 2 and all(isinstance(v, (int, float)) and
                       math.isfinite(v) and 0 <= v <= .01 for v in knees.values()))
            passed = valid and survived and bounded and all(x <= t for x, t in zip(mae, limits))
            score = sum(min(x/t, 100.) for x, t in zip(mae, limits))/5 if valid else 100.
            score += (32-r.get('survived', 0))/32*10
            drift = r.get('position', {})
            if stationary:
                values = [drift.get(k) for k in ('p95_max_drift_m', 'mean_horizontal_speed_m_s', 'p95_max_heading_deg')]
                drift_ok = (finite_vector(values, 3) and drift.get('valid') is True
                            and drift.get('survivor_trials', 0) >= 31
                            and drift.get('measurement_seconds', 0) >= (45.9 if task == 'walk_then_stop' else 55.9))
                passed = passed and drift_ok and all(x <= t for x, t in zip(values, (.15, .02, 15.)))
                score += sum(min(x/t, 100.) for x, t in zip(values, (.15, .02, 15.)))/3 if drift_ok else 100.
            checks[name][task] = dict(passed=bool(passed), survived=r.get('survived'), mae=mae, position=drift)
            if not passed:
                failures.append(name+'/'+task)
            scores.append(score)
    rows = mujoco.get('results', [])
    if len(rows) != 2 or {r['case'] for r in rows} != {'zero_command', 'walk_then_stop'}:
        raise ValueError('Missing independent MuJoCo standing cases')
    mj_pass = True
    for r in rows:
        v = [r.get('max_drift_m'), r.get('mean_horizontal_speed_m_s')]
        valid = finite_vector(v, 2)
        ok = (r.get('completed') is True and r.get('valid') is True
              and r.get('seconds', 0) >= 60. and valid and v[0] <= .15 and v[1] <= .02
              and r.get('measurement_seconds', 0) >= (45.9 if r['case'] == 'walk_then_stop' else 55.9))
        if not ok:
            failures.append('mujoco/'+r['case'])
        mj_pass &= bool(ok)
        scores.append(sum(min(x/t, 100.) for x, t in zip(v, (.15, .02)))/2 if valid else 100.)
    regression = not mj_pass or any(not r['passed'] for name, rows in checks.items()
                                    if name != stage for r in rows.values())
    return dict(passed=not failures, regression=regression, failures=failures, checks=checks,
                mujoco_passed=mj_pass, score=sum(scores)/len(scores),
                thresholds=dict(isaac_p95_max_drift_m=.15, standing_mean_speed_m_s=.02,
                                isaac_p95_max_heading_deg=15., survived_at_least=31,
                                mujoco_max_drift_m=.15),
                method='Each scenario must pass; full 60s trials, 32 Isaac trials per stage; independent fixed-arm MuJoCo is required too')


def decide(history, stage, stage_start, completed, result):
    """Two distinct passing checkpoints to promote; two regressions to reduce difficulty."""
    same = []
    for row in reversed(history):
        if row['stage'] != stage or row['decision']['next_stage'] != stage:
            break
        same.append(row)
    prior = same[0] if same else None
    if prior and completed <= prior['completed_updates']:
        raise ValueError('A repeated checkpoint cannot count as another validation')
    next_stage, action = stage, 'hold'
    age = completed-stage_start
    index = STAGES.index(stage)
    if result['passed'] and prior and prior['gate']['passed'] and age >= 2000:
        if index < len(STAGES)-1:
            next_stage, action = STAGES[index+1], 'promote'
        else:
            action = 'full_stage_validated'
    elif (index > 0 and result['regression'] and prior and prior['gate']['regression']
          and age >= 2000):
        next_stage, action = STAGES[index-1], 'rollback_difficulty'
    # User preference: a plateau or stage age never ends training. Keep learning
    # at this difficulty, or reduce it when retention checks repeatedly fail.
    return dict(action=action, stage=stage, next_stage=next_stage, completed_updates=completed,
                stage_updates=age, passed_streak=(2 if action in ('promote', 'full_stage_validated') else int(result['passed'])))


def validate_transition(path, checkpoint, target_stage):
    from tools.teleop_curriculum import checkpoint_stage
    artifact = json.loads(Path(path).read_text())
    previous = checkpoint_stage(checkpoint)
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    decision = artifact['decision']
    if (artifact['checkpoint_sha256'] != digest or artifact['stage'] != previous
            or decision['stage'] != previous or decision['next_stage'] != target_stage
            or abs(STAGES.index(previous)-STAGES.index(target_stage)) != 1):
        raise ValueError('Curriculum decision does not authorize this exact checkpoint transition')
    expected_action = 'promote' if STAGES.index(target_stage) > STAGES.index(previous) else 'rollback_difficulty'
    if decision['action'] != expected_action:
        raise ValueError('Wrong curriculum transition direction')
    # Recompute the latest gate from the saved, hashed reports instead of trusting booleans.
    evidence = {}
    for name, item in artifact['evidence'].items():
        p = Path(item['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Curriculum evaluation evidence changed')
        if name in ('isaac', 'mujoco'):
            evidence[name] = json.loads(p.read_text())
    verified = gate(evidence['isaac'], evidence['mujoco'], previous, digest,
                    artifact['evidence']['onnx']['sha256'])
    if verified != artifact['gate']:
        raise ValueError('Saved gate differs from the evaluated evidence')
    for row in artifact['prior_history'][-1:]:
        prior_path = Path(row['decision_path'])
        if hashlib.sha256(prior_path.read_bytes()).hexdigest() != row['decision_sha256']:
            raise ValueError('Previous curriculum decision changed')
        prior = json.loads(prior_path.read_text())
        if (prior['gate'] != row['gate'] or prior['decision'] != row['decision']
                or prior['checkpoint_sha256'] == digest
                or prior['completed_updates'] >= artifact['completed_updates']):
            raise ValueError('Consecutive evaluations must refer to distinct, ordered checkpoints')
    recomputed = decide(artifact['prior_history'], previous, artifact['stage_start_updates'],
                        artifact['completed_updates'], verified)
    if recomputed != decision:
        raise ValueError('Curriculum decision does not match consecutive evaluation history')
    return decision
