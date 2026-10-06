"""Acceptance for locomotion only; pose commands are intentionally absent."""
import math

TASKS = ('stand', 'forward_030', 'forward_060', 'backward_030',
         'lateral_pos', 'lateral_neg', 'turn_pos', 'turn_neg')
PROFILES = ('bootstrap', 'controlled_arms_on', 'training_disturbances')


def evaluate_gates(report, tasks=TASKS, tracking_limits=None):
    results = {}
    for name in PROFILES:
        profiles = [p for p in report.get('profiles', []) if p['settings']['name'] == name]
        failures, checks, score = [], {}, 0.
        if len(profiles) != 1:
            results[name] = dict(passed=False, failures=['Missing unique profile'], score=None)
            continue
        if (report.get('repeats', 0) < 32 or report.get('warmup_s', 0) < 4
                or report.get('measure_s', 0) < 12 or not report.get('arm_guard_enabled')):
            failures.append('Insufficient guarded evaluation')
        rows = {r['scenario']: r for r in profiles[0]['scenarios']}
        for task in tasks:
            r = rows.get(task, {})
            mae = r.get('mae', [])
            limits = tracking_limits or ((.02, .02, .10) if task == 'stand' else (.08, .08, .15))
            valid = len(mae) == 5 and all(math.isfinite(x) and x >= 0 for x in mae)
            tracking = valid and all(x <= t for x, t in zip(mae[:3], limits))
            survival = r.get('trials', 0) >= 32 and r.get('survived', 0) >= math.ceil(r['trials'] * 31/32)
            knees = r.get('knee_clipped_fraction', {})
            bounded = len(knees) == 2 and all(math.isfinite(v) and 0 <= v <= .5 for v in knees.values())
            passed = tracking and survival and bounded and r.get('valid_samples', 0) > 0
            checks[task] = dict(passed=passed, survival_passed=survival, tracking_passed=tracking,
                                knee_bounds_passed=bounded, mae=mae, survived=r.get('survived', 0))
            if not passed:
                failures.append(task)
            score += sum(x/t for x, t in zip(mae[:3], limits)) if valid else math.inf
        results[name] = dict(passed=not failures, failures=failures, checks=checks,
                             score=score if math.isfinite(score) else None)
    return dict(passed=all(r['passed'] for r in results.values()), profiles=results)
