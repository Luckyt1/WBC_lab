"""Position cost relative to one fixed anchor per uninterrupted standing phase."""
import math

import torch


class StandingPositionTracker:
    def __init__(self, num_envs, device, dt, settling_time=0.5, deadzone=0.01, distance_scale=0.1):
        values = (dt, settling_time, deadzone, distance_scale)
        if not all(math.isfinite(v) for v in values) or dt <= 0 or settling_time < 0 or deadzone < 0 or distance_scale <= 0:
            raise ValueError('Require finite dt/scale > 0 and settling_time/deadzone >= 0')
        self.settling_steps = max(0, math.ceil(settling_time / dt - 1e-9))
        self.deadzone = deadzone
        self.distance_scale = distance_scale
        self.anchor_xy = torch.zeros(num_envs, 2, device=device)
        self.standing_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.distance = torch.zeros(num_envs, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.anchor_xy[ids] = 0.
        self.standing_steps[ids] = 0
        self.active[ids] = False
        self.distance[ids] = 0.

    def step(self, position_xy, standing):
        """Call once after each physics transition, before resampling its command.

        Allow a fixed braking interval, then freeze the anchor even if the robot
        never slows down. Repeated zero-command samples must not renew the grace
        period. Resetting one environment invalidates only its own anchor.
        """
        entering = standing & (self.standing_steps == 0)
        self.standing_steps = torch.where(
            standing, (self.standing_steps + 1).clamp(max=self.settling_steps + 1), 0)
        capture = standing & (entering | (self.standing_steps <= self.settling_steps))
        self.anchor_xy = torch.where(capture[:, None], position_xy, self.anchor_xy)
        self.active = standing & (self.standing_steps > self.settling_steps)
        distance = (position_xy - self.anchor_xy).norm(dim=-1)
        self.distance = torch.where(self.active, distance, 0.)
        return (self.distance - self.deadzone).clamp(min=0.) / self.distance_scale


def configure_standing_position_penalty(cfg, weight=-1.0, settling_time=0.5, deadzone=0.01, distance_scale=0.1):
    """Opt in explicitly so already-running baseline evaluations keep their recipe."""
    if not math.isfinite(weight) or weight >= 0:
        raise ValueError('Standing position penalty weight must be negative and finite')
    from isaaclab.managers import RewardTermCfg
    from .standing_position_term import StandingPositionPenalty

    params = dict(settling_time=settling_time, deadzone=deadzone, distance_scale=distance_scale)
    cfg.reward.standing_position_drift = RewardTermCfg(func=StandingPositionPenalty, weight=weight, params=params)
    return dict(weight=weight, **params)
