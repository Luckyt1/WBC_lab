"""CPU-only replay and aggregation helpers for frozen-policy arm-speed tests."""
import numpy as np


def resample_source(arm, start_frame, source_fps, dt, speed, steps):
    """Physical-time replay, holding the final frame (never silently looping)."""
    arm = np.asarray(arm)
    if arm.ndim != 2 or not len(arm) or not np.isfinite(arm).all():
        raise ValueError('arm must be a nonempty finite [frames, joints] array')
    if (not np.isfinite([source_fps, dt, speed]).all()
            or min(source_fps, dt, speed) <= 0 or steps <= 0
            or not 0 <= start_frame < len(arm)):
        raise ValueError('invalid replay timing/start/length')
    frame = start_frame + np.arange(steps) * source_fps * dt * speed
    held_tail = frame > len(arm)-1
    frame = frame.clip(0, len(arm)-1)
    lower = np.floor(frame).astype(int)
    upper = np.minimum(lower+1, len(arm)-1)
    alpha = (frame-lower)[:, None]
    return (arm[lower]*(1-alpha)+arm[upper]*alpha).astype(np.float32), held_tail


def summarize(arrays, warmup_steps, dt, moving_threshold=.05):
    """First-trial metrics; warmup failures cannot enter drift survivor statistics."""
    valid = arrays['valid'].copy()
    valid[:warmup_steps] = False
    alive_at_anchor = arrays['valid'][warmup_steps-1] & ~arrays['terminated'][warmup_steps-1]
    valid &= alive_at_anchor[None, :]
    survivors = arrays['valid'][-1] & ~arrays['terminated'][-1]
    anchor = arrays['pelvis_xy'][warmup_steps-1]
    drift = np.linalg.norm(arrays['pelvis_xy'] - anchor[None], axis=-1)
    yaw = arrays['pelvis_yaw'] - arrays['pelvis_yaw'][warmup_steps-1][None]
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

    def stats(values, mask):
        selected = np.asarray(values)[mask]
        if not selected.size:
            return dict(mean=None, p95=None, max=None, count=0)
        return dict(mean=float(selected.mean()), p95=float(np.quantile(selected, .95)),
                    max=float(selected.max()), count=int(selected.size))

    result = dict(trials=int(valid.shape[1]), survived=int(survivors.sum()),
                  survival_fraction=float(survivors.mean()),
                  measurement_sample_count=int(valid.sum()), measurement_env_seconds=float(valid.sum()*dt),
                  warmup_failures=int((~alive_at_anchor).sum()),
                  drift_m=stats(drift, valid), yaw_abs_rad=stats(abs(yaw), valid),
                  horizontal_speed_m_s=stats(arrays['horizontal_speed'], valid),
                  max_drift_survivors_m=stats(np.where(valid, drift, 0.).max(0), survivors),
                  min_height_m=float(arrays['height'][arrays['valid']].min()))
    failure_times = []
    for i in range(valid.shape[1]):
        failed = np.flatnonzero(arrays['terminated'][:, i] & arrays['valid'][:, i])
        failure_times.append(float((failed[0]+1)*dt) if len(failed) else None)
    result['first_failure_s'] = failure_times
    for key in ('amass', 'neutral', 'padded_tail', 'standing'):
        result[key + '_fraction'] = stats(arrays[key].astype(float), valid)['mean']
    for name in ('reference', 'target', 'actual'):
        for derivative, unit in (('velocity', 'rad_s'), ('acceleration', 'rad_s2')):
            key = name + '_' + derivative
            mask = valid.copy()
            if key == 'actual_acceleration':
                mask &= arrays['actual_acceleration_valid']
            elif key != 'actual_velocity':
                mask &= arrays[derivative + '_valid']
            result[key + '_abs_' + unit] = stats(abs(arrays[key]), mask)
            if derivative == 'velocity':
                moving = abs(arrays[key]).max(-1) > moving_threshold
                result[name + '_moving_fraction'] = stats(moving.astype(float), mask)['mean']
                result[name + '_moving_within_amass_fraction'] = stats(
                    moving.astype(float), mask & arrays['amass'])['mean']
    for key in ('tracking_error', 'reference_target_error'):
        result[key + '_abs_rad'] = stats(abs(arrays[key]), valid)
    return result
