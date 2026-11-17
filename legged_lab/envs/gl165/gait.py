"""Observable walking clock and soft foot-placement references (no pose targets)."""

import math

import torch


class AlternatingGait:
    def __init__(self, cfg, num_envs, device, dt):
        self.cfg, self.dt = cfg, dt
        if not (cfg.period_s > 0. and .5 <= cfg.stance_fraction < 1.):
            raise ValueError('Expected a positive gait period and no flight phase')
        if not (0. < cfg.contact_phase_margin < (1. - cfg.stance_fraction) / 2.):
            raise ValueError('Contact transition margin consumes the swing phase')
        if min(cfg.start_transition_s, cfg.stop_transition_s, cfg.stop_alignment_ramp_s) <= 0.:
            raise ValueError('Gait transition durations must be positive')
        if cfg.stop_alignment_delay_s < cfg.stop_transition_s:
            raise ValueError('Standing alignment must wait for the walking reference to fade')
        self.phase = torch.zeros(num_envs, device=device)
        self.activity = torch.zeros_like(self.phase)
        self.standing_time = torch.zeros_like(self.phase)
        self.refresh(torch.zeros(num_envs, 3, device=device))

    def reset(self, env_ids, command):
        # Random phase gives both legs equal opportunities to initiate a step.
        self.phase[env_ids] = torch.rand(len(env_ids), device=self.phase.device)
        self.activity[env_ids] = 0.
        standing = torch.linalg.vector_norm(command[env_ids], dim=-1) < .1
        self.standing_time[env_ids] = standing * (
            self.cfg.stop_alignment_delay_s + self.cfg.stop_alignment_ramp_s)
        self.refresh(command)

    def update(self, command):
        moving = torch.linalg.vector_norm(command, dim=-1) >= .1
        self.activity = torch.where(moving,
            self.activity + self.dt / self.cfg.start_transition_s,
            self.activity - self.dt / self.cfg.stop_transition_s).clamp(0., 1.)
        self.phase = (self.phase + (self.activity > 0.) * self.dt / self.cfg.period_s) % 1.
        self.standing_time = torch.where(moving, 0., self.standing_time + self.dt)
        self.refresh(command)

    def refresh(self, command):
        cfg = self.cfg
        self.foot_phase = (self.phase[:, None] + self.phase.new_tensor([0., .5])) % 1.
        p, d, margin = self.foot_phase, cfg.stance_fraction, cfg.contact_phase_margin
        self.swing = p >= d
        self.stance_core = (p >= margin) & (p <= d - margin)
        self.swing_core = (p >= d + margin) & (p <= 1. - margin)
        progress = ((p - d) / (1. - d)).clamp(0., 1.)
        self.clearance = cfg.swing_clearance_m * torch.sin(math.pi * progress).square()
        self.clearance *= self.activity[:, None]
        # During stance the foot travels backwards relative to the pelvis;
        # swing returns it forwards. Negative vx reverses this reference.
        shape = torch.where(self.swing, -.5 + progress.square() * (3. - 2. * progress), .5 - p / d)
        stride = command[:, 0] * cfg.period_s * d
        self.fore_aft = ((shape[:, 0] - shape[:, 1]) * stride).clamp(
            -cfg.max_fore_aft_m, cfg.max_fore_aft_m) * self.activity
        self.fore_aft_gate = ((command[:, 0].abs() - .05) / .05).clamp(0., 1.)
        self.fore_aft_gate *= (torch.linalg.vector_norm(command, dim=-1) >= .1)
        self.standing_alignment = ((self.standing_time - cfg.stop_alignment_delay_s)
                                   / cfg.stop_alignment_ramp_s).clamp(0., 1.)

    def observations(self):
        angle = 2. * math.pi * self.phase
        return torch.stack((torch.sin(angle) * self.activity, torch.cos(angle) * self.activity,
                            self.activity, self.standing_alignment), dim=-1)
