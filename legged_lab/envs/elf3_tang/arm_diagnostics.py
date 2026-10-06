"""Read-only arm telemetry at the policy rate; no simulator imports or RNG use."""
import math

import torch


class ArmMotionDiagnostics:
    def __init__(self, num_envs, num_joints, dt, device, moving_threshold=.05):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        self.dt = dt
        self.moving_threshold = moving_threshold
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.previous = {name: torch.zeros(num_envs, num_joints, device=device)
                         for name in ('reference', 'target', 'actual_velocity')}
        self.previous_velocity = {name: torch.zeros_like(self.previous[name])
                                  for name in ('reference', 'target')}

    def reset(self, ids):
        # Exclude derivatives across episode boundaries, even for partial resets.
        self.age[ids] = 0

    @torch.no_grad()
    def update(self, reference, target, actual_position, actual_velocity, standing,
               amass, neutral, padded_tail):
        velocity_valid = self.age >= 1
        acceleration_valid = self.age >= 2
        sample = dict(velocity_valid=velocity_valid, acceleration_valid=acceleration_valid,
                      actual_acceleration_valid=velocity_valid, standing=standing.clone(),
                      amass=amass.clone(), neutral=neutral.clone(), padded_tail=padded_tail.clone())
        for name, position in (('reference', reference), ('target', target)):
            velocity = (position - self.previous[name]) / self.dt
            acceleration = (velocity - self.previous_velocity[name]) / self.dt
            sample[name + '_velocity'] = torch.where(velocity_valid[:, None], velocity, 0.)
            sample[name + '_acceleration'] = torch.where(acceleration_valid[:, None], acceleration, 0.)
            self.previous[name].copy_(position)
            self.previous_velocity[name].copy_(velocity)
        sample['actual_velocity'] = actual_velocity.clone()
        sample['actual_acceleration'] = torch.where(
            velocity_valid[:, None], (actual_velocity-self.previous['actual_velocity'])/self.dt, 0.)
        self.previous['actual_velocity'].copy_(actual_velocity)
        sample['tracking_error'] = target - actual_position
        sample['reference_target_error'] = reference - target
        sample['target_moving'] = (sample['target_velocity'].abs().amax(-1) > self.moving_threshold)
        sample['actual_moving'] = actual_velocity.abs().amax(-1) > self.moving_threshold
        self.age += 1
        return sample, self.reduce(sample)

    @staticmethod
    def reduce(sample):
        def mean(value, mask):
            return torch.where(mask, value, 0.).sum() / mask.sum().clamp(min=1)

        all_envs = torch.ones_like(sample['standing'])
        valid = sample['velocity_valid']
        log = {}
        for prefix, group in (('ArmMotion', all_envs), ('StandingArms', sample['standing'])):
            log[prefix + '/sample_fraction'] = group.float().mean()
            log[prefix + '/derivative_valid_fraction'] = mean(valid.float(), group)
            for name in ('amass', 'neutral', 'padded_tail', 'actual_moving'):
                log[prefix + '/' + name + '_fraction'] = mean(sample[name].float(), group)
            log[prefix + '/target_moving_fraction'] = mean(sample['target_moving'].float(), group & valid)
            for name in ('tracking_error', 'reference_target_error'):
                log[prefix + '/' + name + '_mae_rad'] = mean(sample[name].abs().mean(-1), group)
            for name in ('reference', 'target', 'actual'):
                for derivative, unit in (('velocity', 'rad_s'), ('acceleration', 'rad_s2')):
                    key = name + '_' + derivative
                    if key == 'actual_velocity':
                        mask = group
                    elif key == 'actual_acceleration':
                        mask = group & sample['actual_acceleration_valid']
                    else:
                        mask = group & sample[derivative + '_valid']
                    values = sample[key].abs()
                    log[f'{prefix}/{key}_abs_mean_{unit}'] = mean(values.mean(-1), mask)
                    # The training writer averages these per-step peaks over its logging window.
                    log[f'{prefix}/{key}_step_peak_{unit}'] = torch.where(mask[:, None], values, 0.).amax()
        return log
