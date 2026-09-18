"""Opt-in command episodes for bounded tracking-repair experiments (Torch only)."""

import torch


class TrackingRepairSchedule:
    """Override a fixed subset of environments; leave other commands untouched.

    Each specialist command episode has 4 s preparation and 12 s tracking.
    Physical resets and command resampling both start a fresh command episode.
    No target or action is injected into the robot.
    """

    def __init__(self, generator, mode, fraction=.2, yaw_max=.2):
        if mode not in ('shallow', 'yaw'):
            raise ValueError('mode must be shallow or yaw')
        if not 0 < fraction <= 1 or not .1 <= yaw_max <= .4:
            raise ValueError('invalid specialist fraction or yaw range')
        self.gen, self.mode, self.yaw_max = generator, mode, yaw_max
        device, count = generator.command.device, len(generator.command)
        self.ids = torch.arange(round(count * fraction), device=device)
        if not len(self.ids):
            raise ValueError('at least one specialist environment is required')
        self.special = torch.zeros(count, dtype=torch.bool, device=device)
        self.special[self.ids] = True
        self.elapsed = torch.zeros(count, device=device)
        self.prepare = torch.zeros_like(generator.command)
        self.target = torch.zeros_like(generator.command)
        self._resample_original = generator._resample_command
        self._update_original = generator._update_command
        generator._resample_command = self.resample
        generator._update_command = self.update
        self.reset_special(self.ids)

    def reset_special(self, ids):
        if not len(ids):
            return
        n, device = len(ids), self.target.device
        self.elapsed[ids] = 0.
        self.prepare[ids] = 0.
        self.target[ids] = 0.
        if self.mode == 'shallow':
            # Half of command episodes start from stand, half from deeper squat.
            self.prepare[ids, 3] = torch.where(torch.rand(n, device=device) < .5, .8235, .65)
            self.target[ids, 3] = torch.empty(n, device=device).uniform_(.72, .78)
        else:
            height = torch.where(torch.rand(n, device=device) < .5, .8235, .6)
            self.prepare[ids, 3] = self.target[ids, 3] = height
            sign = torch.where(torch.rand(n, device=device) < .5, -1., 1.)
            self.target[ids, 6] = sign * torch.empty(n, device=device).uniform_(.1, self.yaw_max)
        self.gen.time_left[ids] = 16.
        self.apply(ids)

    def resample(self, env_ids):
        self._resample_original(env_ids)
        ids = torch.as_tensor(env_ids, device=self.target.device, dtype=torch.long)
        self.reset_special(ids[self.special[ids]])

    def apply(self, ids):
        self.gen.command[ids] = torch.where(
            (self.elapsed[ids] >= 4.)[:, None], self.target[ids], self.prepare[ids])
        for name in ('is_heading_env', 'is_zero_vel_yaw_env', 'is_in_place_turn_env',
                     'is_nominal_height_env', 'is_high_stance_turn_env',
                     'is_high_stance_move_turn_env', 'is_low_stance_move_env'):
            getattr(self.gen, name)[ids] = False
        self.gen.is_standing_env[ids] = True

    def update(self):
        self._update_original()
        self.elapsed[self.ids] += self.gen._env.step_dt
        self.apply(self.ids)


def shallow_precision_reward(env, std=.05):
    """Additional height precision only during specialist tracking, not preparation."""
    schedule = env.tracking_repair_schedule
    height = env.robot.data.body_pos_w[:, env.base_body_id, 2]
    error = height - env.command_generator.command[:, 3]
    return torch.exp(-(error / std).square()) * schedule.special * (schedule.elapsed >= 4.)
