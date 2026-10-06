"""Five task commands: pelvis vx/vy/wz, pelvis height, torso yaw relative to pelvis.

This module is independent of Isaac Sim so sampling and frame conventions can be
checked on CPU. ``command`` remains the three velocity channels used by shared
locomotion rewards; the policy observes ``full_command`` (all five channels).
"""

from dataclasses import dataclass, field

import torch


@dataclass
class VelocityRanges:
    lin_vel_x: tuple = (-0.5, 0.6)
    lin_vel_y: tuple = (-0.3, 0.3)
    ang_vel_z: tuple = (-0.8, 0.8)


@dataclass
class TangCommandCfg:
    ranges: VelocityRanges = field(default_factory=VelocityRanges)
    resampling_time_range: tuple = (6.0, 10.0)
    rel_standing_envs: float = 0.25
    rel_neutral_standing: float = 0.0  # opt-in extra neutral-pose examples among standing commands
    rel_lateral_envs: float = 0.0
    rel_crouch_yaw_envs: float = 0.0
    focus_start_iteration: int = 81598
    focus_end_iteration: int = 83598
    focus_hold_time_range: tuple = (10.0, 14.0)
    focus_lateral_initial_range: tuple = (.1, .2)
    focus_lateral_final_range: tuple = (.15, .3)
    focus_vx_initial_range: tuple = (.1, .2)
    focus_vx_final_range: tuple = (.2, .3)
    focus_height_initial_range: tuple = (.6, .65)
    focus_height_final_range: tuple = (.5, .6)
    focus_yaw_initial_range: tuple = (.15, .25)
    focus_yaw_final_range: tuple = (.4, .6)
    # Optional scratch-training blend: standing, turning, lateral, crouch+yaw.
    initial_mode_fractions: tuple | None = None
    # Opt-in isolated skills for the repaired scratch curriculum.
    sagittal_speed_range: tuple | None = None  # magnitude, equal forward/backward samples
    mixed_yaw_fraction: float = 0.0  # opt-in fraction of isolated sagittal samples
    mixed_yaw_range: tuple = (.02, .15)  # positive magnitude; symmetric signs
    turning_speed_range: tuple | None = None  # magnitude, equal left/right samples
    isolate_standing_pose: bool = False  # height OR waist yaw, never both together
    sampling_height_range: tuple | None = None
    sampling_waist_yaw_range: tuple | None = None
    mode_start_iteration: int = 0
    mode_end_iteration: int = 8000
    rel_turning_envs: float = 0.15  # fraction of ALL environments; remaining 60% move
    rel_zero_moving_yaw: float = 0.5
    nominal_height: float = 0.8235
    height_initial_range: tuple = (0.78, 0.8235)
    height_range: tuple = (0.50, 0.8235)
    waist_yaw_initial_range: tuple = (-0.10, 0.10)
    waist_yaw_range: tuple = (-0.60, 0.60)
    rel_nominal_height: float = 0.25
    rel_zero_waist_yaw: float = 0.25
    velocity_initial_scale: float = 0.3
    velocity_start_iteration: int = 0
    velocity_end_iteration: int = 10000
    pose_start_iteration: int = 5000
    pose_end_iteration: int = 20000
    height_rate: float = 0.12  # m/s, including stand -> squat transitions
    waist_yaw_rate: float = 0.4  # rad/s


def curriculum_progress(iteration, start, end):
    return min(max((iteration - start) / max(end - start, 1), 0.0), 1.0)


def wrap_to_pi(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quat_yaw(quat):
    """Heading of a normalized Isaac quaternion, ordered (w, x, y, z)."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def rotate_inverse(quat, vector):
    """Rotate world vectors into a body frame; quaternions are (w, x, y, z)."""
    xyz = -quat[..., 1:]
    uv = 2 * torch.cross(xyz, vector, dim=-1)
    return vector + quat[..., :1] * uv + torch.cross(xyz, uv, dim=-1)


def yaw_frame_velocity(quat, velocity):
    yaw = quat_yaw(quat)
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c * velocity[:, 0] + s * velocity[:, 1],
                        -s * velocity[:, 0] + c * velocity[:, 1]), dim=-1)


class TangCommands:
    def __init__(self, cfg, num_envs, device):
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self.iteration = 0
        fractions = (cfg.rel_standing_envs, cfg.rel_turning_envs, cfg.rel_lateral_envs, cfg.rel_crouch_yaw_envs)
        if any(f < 0 for f in fractions) or sum(fractions) > 1:
            raise ValueError("Command mode fractions must be non-negative and sum to at most 1.")
        if cfg.initial_mode_fractions is not None:
            initial = cfg.initial_mode_fractions
            if len(initial) != 4 or any(f < 0 for f in initial) or sum(initial) > 1:
                raise ValueError('Initial command mode fractions must be four non-negative fractions summing to <=1.')
        if not 0 <= cfg.rel_neutral_standing <= 1:
            raise ValueError("Neutral standing fraction must lie between zero and one.")
        if not 0 <= cfg.mixed_yaw_fraction <= 1:
            raise ValueError('Mixed yaw fraction must lie between zero and one')
        if cfg.mixed_yaw_fraction:
            lo, hi = cfg.mixed_yaw_range
            if cfg.sagittal_speed_range is None or not (0 < lo <= hi <= min(-cfg.ranges.ang_vel_z[0], cfg.ranges.ang_vel_z[1])):
                raise ValueError('Mixed yaw requires isolated sagittal sampling and bounded positive yaw magnitudes')
        for name in ('sagittal_speed_range', 'turning_speed_range'):
            bounds = getattr(cfg, name)
            if bounds is not None and not (0 < bounds[0] <= bounds[1]):
                raise ValueError(f'{name} must contain positive ordered magnitudes')
        if not 0 < cfg.height_range[0] <= cfg.nominal_height <= cfg.height_range[1]:
            raise ValueError("Nominal pelvis height must be inside the positive height range.")
        self.full_command = torch.zeros(num_envs, 5, device=device)
        self.full_command[:, 3] = cfg.nominal_height
        self.target = self.full_command.clone()
        self.time_left = torch.zeros(num_envs, device=device)
        self.is_standing_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # 0=standing, 1=turning, 2=general movement, 3=lateral, 4=crouch+yaw.
        self.sampled_mode = torch.full((num_envs,), 2, dtype=torch.long, device=device)
        self.manual_command = None

    @property
    def command(self):
        return self.full_command[:, :3]

    @property
    def wbc_command(self):
        """WBC reward layout: vx, vy, wz, height, zero roll/pitch, waist yaw.

        Keep the policy's five command channels unchanged. The two posture
        rewards use fixed upright targets, not additional sampled commands.
        """
        zeros = torch.zeros_like(self.full_command[:, :2])
        return torch.cat((self.full_command[:, :4], zeros, self.full_command[:, 4:5]), dim=-1)

    @property
    def vel_command_b(self):
        return self.command

    def pose_ranges(self):
        p = curriculum_progress(self.iteration, self.cfg.pose_start_iteration, self.cfg.pose_end_iteration)
        blend = lambda a, b: tuple(x + p * (y - x) for x, y in zip(a, b))
        return (self.cfg.sampling_height_range or blend(self.cfg.height_initial_range, self.cfg.height_range),
                self.cfg.sampling_waist_yaw_range or blend(self.cfg.waist_yaw_initial_range, self.cfg.waist_yaw_range))

    def mode_fractions(self):
        final = (self.cfg.rel_standing_envs, self.cfg.rel_turning_envs,
                 self.cfg.rel_lateral_envs, self.cfg.rel_crouch_yaw_envs)
        initial = self.cfg.initial_mode_fractions
        if initial is None:
            return final
        p = curriculum_progress(self.iteration, self.cfg.mode_start_iteration, self.cfg.mode_end_iteration)
        return tuple(a + p * (b-a) for a, b in zip(initial, final))

    def _resample(self, env_ids):
        n = len(env_ids)
        if not n:
            return
        r = torch.empty(n, device=self.device)
        if self.manual_command is not None:
            self.target[env_ids] = self.manual_command
            self.sampled_mode[env_ids] = 2
        else:
            p = curriculum_progress(self.iteration, self.cfg.velocity_start_iteration,
                                    self.cfg.velocity_end_iteration)
            scale = self.cfg.velocity_initial_scale + (1 - self.cfg.velocity_initial_scale) * p
            for axis, bounds in enumerate((self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y,
                                           self.cfg.ranges.ang_vel_z)):
                self.target[env_ids, axis] = r.uniform_(*bounds) * scale
            mode = torch.rand(n, device=self.device)
            stand_fraction, turn_fraction, lateral_fraction, crouch_fraction = self.mode_fractions()
            stand = mode < stand_fraction
            turn = (mode >= stand_fraction) & (mode < stand_fraction + turn_fraction)
            self.sampled_mode[env_ids] = 2
            self.sampled_mode[env_ids[stand]] = 0
            self.sampled_mode[env_ids[turn]] = 1
            zero_yaw = (torch.rand(n, device=self.device) < self.cfg.rel_zero_moving_yaw) & ~turn
            self.target[env_ids[zero_yaw], 2] = 0
            # Explicit turning examples prevent every small yaw command being drowned by translation.
            self.target[env_ids[turn], :2] = 0
            self.target[env_ids[stand], :3] = 0
            if self.cfg.sagittal_speed_range is not None:
                moving = ~stand & ~turn
                signs = torch.where(torch.rand(n, device=self.device) < .5, -1., 1.)
                self.target[env_ids[moving], :3] = 0
                self.target[env_ids[moving], 0] = (signs * r.uniform_(*self.cfg.sagittal_speed_range))[moving]
            if self.cfg.turning_speed_range is not None:
                signs = torch.where(torch.rand(n, device=self.device) < .5, -1., 1.)
                self.target[env_ids[turn], 2] = (signs * r.uniform_(*self.cfg.turning_speed_range))[turn]
            height_range, yaw_range = self.pose_ranges()
            self.target[env_ids, 3] = r.uniform_(*height_range)
            self.target[env_ids, 4] = r.uniform_(*yaw_range)
            upright = torch.rand(n, device=self.device) < self.cfg.rel_nominal_height
            straight = torch.rand(n, device=self.device) < self.cfg.rel_zero_waist_yaw
            self.target[env_ids[upright], 3] = self.cfg.nominal_height
            self.target[env_ids[straight], 4] = 0
            if self.cfg.rel_neutral_standing > 0:
                neutral = stand & (torch.rand(n, device=self.device) < self.cfg.rel_neutral_standing)
                self.target[env_ids[neutral], 3] = self.cfg.nominal_height
                self.target[env_ids[neutral], 4] = 0
            if self.cfg.isolate_standing_pose:
                # Pose practice only while stationary; retain neutral locomotion.
                height_only = stand & (torch.rand(n, device=self.device) < .5)
                self.target[env_ids[~height_only], 3] = self.cfg.nominal_height
                self.target[env_ids[height_only | ~stand], 4] = 0
            if self.cfg.rel_lateral_envs + self.cfg.rel_crouch_yaw_envs > 0:
                start = stand_fraction + turn_fraction
                lateral = (mode >= start) & (mode < start + lateral_fraction)
                crouch = ((mode >= start + lateral_fraction)
                          & (mode < start + lateral_fraction + crouch_fraction))
                progress = curriculum_progress(self.iteration, self.cfg.focus_start_iteration,
                                               self.cfg.focus_end_iteration)
                signs = torch.where(torch.rand(n, device=self.device) < .5, -1., 1.)
                def focus_range(name):
                    initial = getattr(self.cfg, f'focus_{name}_initial_range')
                    final = getattr(self.cfg, f'focus_{name}_final_range')
                    return tuple(a + progress * (b-a) for a, b in zip(initial, final))
                lateral_speed = r.uniform_(*focus_range('lateral')).clone()
                self.target[env_ids[lateral], :3] = 0
                self.target[env_ids[lateral], 1] = (signs*lateral_speed)[lateral]
                self.target[env_ids[lateral], 3] = self.cfg.nominal_height
                self.target[env_ids[lateral], 4] = 0
                self.sampled_mode[env_ids[lateral]] = 3
                self.target[env_ids[crouch], :3] = 0
                self.target[env_ids[crouch], 0] = r.uniform_(*focus_range('vx'))[crouch]
                self.target[env_ids[crouch], 3] = r.uniform_(*focus_range('height'))[crouch]
                self.target[env_ids[crouch], 4] = (signs*r.uniform_(*focus_range('yaw')))[crouch]
                self.sampled_mode[env_ids[crouch]] = 4
            if self.cfg.mixed_yaw_fraction:
                # Preserve all mode proportions and pure standing/turn/lateral
                # commands. With the default zero fraction, consume no RNG.
                selected = ((self.sampled_mode[env_ids] == 2)
                            & (torch.rand(n, device=self.device) < self.cfg.mixed_yaw_fraction))
                signs = torch.where(torch.rand(n, device=self.device) < .5, -1., 1.)
                self.target[env_ids[selected], 2] = (signs*r.uniform_(*self.cfg.mixed_yaw_range))[selected]
        self.full_command[env_ids, :3] = self.target[env_ids, :3]
        self.is_standing_env[env_ids] = (self.target[env_ids, :3].abs().sum(-1) < 1e-6)
        self.time_left[env_ids] = r.uniform_(*self.cfg.resampling_time_range)
        if self.manual_command is None and self.cfg.rel_lateral_envs + self.cfg.rel_crouch_yaw_envs > 0:
            focused = self.sampled_mode[env_ids] >= 3
            self.time_left[env_ids[focused]] = r.uniform_(*self.cfg.focus_hold_time_range)[focused]

    def reset(self, env_ids):
        # A reset starts near the upright pose, not at a discontinuous deep-squat target.
        self.full_command[env_ids, 3] = self.cfg.nominal_height
        self.full_command[env_ids, 4] = 0
        self._resample(env_ids)

    def compute(self, dt):
        self.time_left -= dt
        self._resample((self.time_left <= 0).nonzero(as_tuple=False).flatten())
        rates = self.full_command.new_tensor([self.cfg.height_rate, self.cfg.waist_yaw_rate]) * dt
        delta = self.target[:, 3:] - self.full_command[:, 3:]
        self.full_command[:, 3:] += torch.clamp(delta, min=-rates, max=rates)

    def set_manual_command(self, command):
        value = torch.as_tensor(command, dtype=self.full_command.dtype, device=self.device)
        if value.shape != (5,) or not torch.isfinite(value).all():
            raise ValueError("Expected finite [vx, vy, wz, pelvis_height, waist_yaw].")
        bounds = (self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z,
                  self.cfg.height_range, self.cfg.waist_yaw_range)
        low, high = value.new_tensor([b[0] for b in bounds]), value.new_tensor([b[1] for b in bounds])
        self.manual_command = torch.clamp(value, min=low, max=high)
        self._resample(torch.arange(self.num_envs, device=self.device))
