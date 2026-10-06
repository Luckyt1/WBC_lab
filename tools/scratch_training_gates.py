"""Simulator-free, per-task acceptance checks; never infer success from reward."""
import math

STAGES = ('locomotion', 'pose', 'combined', 'push_01', 'push_02', 'push_04', 'push_08')
BASIC = ('stand', 'forward_030', 'backward_030', 'turn_pos', 'turn_neg')
POSE = BASIC + ('lateral_pos', 'lateral_neg', 'squat_065', 'waist_yaw_pos', 'waist_yaw_neg')
FULL = POSE + ('forward_060', 'squat_050', 'crouch_walk_065', 'crouch_walk_050',
               'walk_waist_yaw', 'crouch_walk_yaw_pos', 'crouch_walk_yaw_neg')
TOLERANCES = (.08, .08, .15, .03, .10)  # vx, vy, wz, height, waist yaw


def tracking_gate(report, stage, profile_name='curriculum_stage'):
    if stage not in STAGES:
        raise ValueError(f'Unknown stage {stage}')
    required = BASIC if stage == 'locomotion' else POSE if stage == 'pose' else FULL
    failures, checks, score = [], {}, 0.
    profiles = [p for p in report.get('profiles', []) if p['settings']['name'] == profile_name]
    if len(profiles) != 1:
        return dict(passed=False, stage=stage, failures=[f'Missing unique profile {profile_name}'], checks={}, score=None)
    profile = profiles[0]
    if profile_name == 'curriculum_stage' and profile['settings'].get('stage') != stage:
        failures.append('Evaluation stage mismatch')
    if not report.get('arm_guard_enabled') or report.get('repeats', 0) < 32:
        failures.append('Requires guarded evaluation with at least 32 trials per task')
    if report.get('warmup_s', 0) < 4 or report.get('measure_s', 0) < 12:
        failures.append('Evaluation horizon too short')
    rows = {r['scenario']: r for r in profile['scenarios']}
    for name in required:
        row = rows.get(name, {})
        mae = row.get('mae', [])
        finite = len(mae) == 5 and all(isinstance(v, (float, int)) and math.isfinite(v) for v in mae)
        alive = row.get('trials', 0) >= 32 and row.get('survived', 0) >= math.ceil(row.get('trials', 0) * 31/32)
        tracked = finite and all(v <= t for v, t in zip(mae, TOLERANCES))
        # Diagnostic failure must not silently pass as an empty/NaN measurement.
        knees = row.get('knee_clipped_fraction', {})
        bounded = (len(knees) == 2 and all(isinstance(v, (float, int)) and math.isfinite(v)
                                         and 0 <= v <= .5 for v in knees.values()))
        passed = alive and tracked and bounded and row.get('valid_samples', 0) > 0
        checks[name] = dict(passed=passed, survival_passed=alive, tracking_passed=tracked,
                           knee_bounds_passed=bounded, mae=mae, knee_clipped_fraction=knees)
        if not passed:
            failures.append(f'{profile_name}/{name}')
        if finite:
            score += sum(v/t for v, t in zip(mae, TOLERANCES))
        else:
            score = math.inf
    return dict(passed=not failures, stage=stage, failures=failures, checks=checks,
                score=score if math.isfinite(score) else None,
                checkpoint_sha256=report.get('checkpoint_sha256'))


def full_tracking_gate(report):
    names = ('controlled_arms_on', 'controlled_arms_off', 'training_disturbances',
             'controlled_neutral_guarded')
    results = {name: tracking_gate(report, 'push_08', name) for name in names}
    return dict(passed=all(r['passed'] for r in results.values()), profiles=results)


def validate_promotion(report, previous_stage, next_stage, checkpoint_sha256):
    """Require evidence from the exact checkpoint, not a different successful run."""
    if previous_stage not in STAGES or next_stage not in STAGES:
        raise ValueError('A known predecessor stage is required for promotion')
    if STAGES.index(next_stage) != STAGES.index(previous_stage) + 1:
        raise ValueError('Promote only one evaluated stage at a time')
    if not checkpoint_sha256 or report.get('checkpoint_sha256') != checkpoint_sha256:
        raise ValueError('Stage report does not match the resumed checkpoint')
    gate = tracking_gate(report, previous_stage)
    if not gate['passed']:
        raise ValueError('Stage acceptance failed: ' + ', '.join(gate['failures']))
    return gate
