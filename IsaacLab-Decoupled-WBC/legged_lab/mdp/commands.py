# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

"""Custom command terms used by the locomotion environments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.log
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from legged_lab.utils.recovery_tracking import high_stance_commands, low_stance_commands, mix_pose_samples, small_turn_samples

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class CommandAxisCurriculumCfg:
    """Linear curriculum schedule for a single command axis."""

    start_range: tuple[float, float] = (0.0, 0.0)
    end_range: tuple[float, float] = (0.0, 0.0)
    start_iter: int = 0
    end_iter: int = 1000


@configclass
class CommandCurriculumCfg:
    """Per-axis linear curriculum on the sampled command ranges.

    Axes left as ``None`` fall back to the static ``ranges.*`` value
    (no curriculum on that axis). Set ``enable=False`` to disable the
    curriculum entirely.
    """

    enable: bool = True
    lin_vel_x: CommandAxisCurriculumCfg | None = None
    lin_vel_y: CommandAxisCurriculumCfg | None = None
    ang_vel_z: CommandAxisCurriculumCfg | None = None
    body_height: CommandAxisCurriculumCfg | None = None
    body_roll: CommandAxisCurriculumCfg | None = None
    body_pitch: CommandAxisCurriculumCfg | None = None
    body_yaw: CommandAxisCurriculumCfg | None = None


@configclass
class UniformVelocityBodyCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator with optional body commands."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    curriculum: CommandCurriculumCfg = CommandCurriculumCfg()
    """Optional linear curriculum on the sampled command range."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False."""

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """Probability of heading-based angular velocity command when heading commands are enabled."""

    rel_zero_vel_yaw_envs: float = 0.0
    """Conditional probability of exact zero yaw rate in translating, direct-yaw environments.
    Explicit in-place turning environments are excluded.

    Only the yaw-rate component is zeroed; linear velocity remains uniformly sampled from its configured ranges.
    """

    body_command: bool = False
    """Whether to generate body posture/height commands. Defaults to False."""

    rel_in_place_turn_envs: float = 0.0
    """Absolute probability of zero XY with sampled yaw; disjoint from standing."""
    in_place_small_turn_fraction: float = 0.0
    """Opt-in stratification: fraction of turns at magnitude .15 to .6 rad/s."""
    rel_zero_pose_envs: float = 0.0
    rel_single_axis_pose_envs: float = 0.0
    rel_nominal_height_envs: float = 0.0
    """Independent probability of retaining nominal height during height training."""
    nominal_body_height: float = 1.05

    rel_high_stance_turn_envs: float = 0.0
    """Absolute fraction reserved within rel_in_place_turn_envs."""
    rel_high_stance_move_turn_envs: float = 0.0
    """Absolute fraction reserved within the remaining moving environments."""
    high_stance_height_range: tuple[float, float] = (0.78, 0.8235)
    high_stance_speed_range: tuple[float, float] = (0.3, 0.6)
    high_stance_yaw_rate_range: tuple[float, float] = (0.2, 0.6)

    rel_low_stance_move_envs: float = 0.0
    """Opt-in absolute fraction taken from general movement; torso pose retained."""
    low_stance_height_range: tuple[float, float] = (0.4735, 0.60)
    low_stance_speed_range: tuple[float, float] = (0.2, 0.4)
    low_stance_yaw_rate_range: tuple[float, float] = (0.2, 0.6)
    low_stance_turn_fraction: float = 0.5

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity and body commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad)."""

        body_height: tuple[float, float] = MISSING
        """Range for the desired body height command (in meters)."""

        body_roll: tuple[float, float] = MISSING
        """Range for the desired body roll command (in radians)."""

        body_pitch: tuple[float, float] = MISSING
        """Range for the desired body pitch command (in radians)."""

        body_yaw: tuple[float, float] = MISSING
        """Range for the desired body yaw command (in radians)."""

    ranges: Ranges = Ranges()
    """Distribution ranges for the velocity and body commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker."""

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


class UniformVelocityBodyCommand(CommandTerm):
    r"""Command generator that extends Isaac Lab velocity commands with torso-pose commands."""

    cfg: UniformVelocityBodyCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityBodyCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )
        for name in ("rel_standing_envs", "rel_heading_envs", "rel_zero_vel_yaw_envs",
                     "rel_in_place_turn_envs", "rel_nominal_height_envs", "in_place_small_turn_fraction",
                     "rel_zero_pose_envs", "rel_single_axis_pose_envs",
                     "rel_high_stance_turn_envs", "rel_high_stance_move_turn_envs",
                     "rel_low_stance_move_envs", "low_stance_turn_fraction"):
            probability = float(getattr(self.cfg, name))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"The command probability '{name}' must be in [0, 1], got {probability}.")
        if self.cfg.rel_standing_envs + self.cfg.rel_in_place_turn_envs > 1.0:
            raise ValueError("Standing and in-place turn probabilities must sum to at most 1.")
        if self.cfg.rel_high_stance_turn_envs > self.cfg.rel_in_place_turn_envs:
            raise ValueError("High-stance in-place fraction must fit within the in-place turn fraction.")
        moving_fraction = 1. - self.cfg.rel_standing_envs - self.cfg.rel_in_place_turn_envs
        if self.cfg.rel_high_stance_move_turn_envs > moving_fraction + 1e-9:
            raise ValueError("High-stance moving-turn fraction must fit within the moving fraction.")
        if self.cfg.rel_high_stance_move_turn_envs + self.cfg.rel_low_stance_move_envs > moving_fraction + 1e-9:
            raise ValueError("High- and low-stance movement fractions must fit within the moving fraction.")
        if self.cfg.rel_low_stance_move_envs > 0:
            self._validate_low_stance_ranges()
        if self.cfg.rel_high_stance_turn_envs + self.cfg.rel_high_stance_move_turn_envs > 0:
            self._validate_high_stance_ranges()
        if self.cfg.rel_zero_pose_envs + self.cfg.rel_single_axis_pose_envs > 1.0:
            raise ValueError("Zero and single-axis pose probabilities must sum to at most 1.")
        if self.cfg.in_place_small_turn_fraction > 0 and self.cfg.rel_in_place_turn_envs > 0:
            lo, hi = self.cfg.ranges.ang_vel_z
            if abs(lo + hi) > 1e-6 or hi < .6 or self.cfg.curriculum.ang_vel_z is not None:
                raise ValueError("Stratified turns require symmetric yaw-rate bounds >= .6 and no yaw-rate curriculum.")
        if self.cfg.rel_nominal_height_envs > 0:
            if not self.cfg.body_command or not (
                self.cfg.ranges.body_height[0] <= self.cfg.nominal_body_height <= self.cfg.ranges.body_height[1]
            ):
                raise ValueError("Nominal-height sampling requires body commands and a height inside the configured range.")

        self.robot: Articulation = env.scene[cfg.asset_name]

        if self.cfg.body_command:
            self.vel_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        else:
            self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_heading_env)
        self.is_in_place_turn_env = torch.zeros_like(self.is_heading_env)
        self.is_nominal_height_env = torch.zeros_like(self.is_heading_env)
        self.is_high_stance_turn_env = torch.zeros_like(self.is_heading_env)
        self.is_high_stance_move_turn_env = torch.zeros_like(self.is_heading_env)
        self.is_low_stance_move_env = torch.zeros_like(self.is_heading_env)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # Current rsl-rl iteration estimate used by the per-axis curriculum schedules.
        # The base environment updates this via ``update_curriculum`` once per step.
        self.current_iteration: float = 0.0

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tDirect-yaw zero-rate probability (conditional): {self.cfg.rel_zero_vel_yaw_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity/body command in the base frame."""
        return self.vel_command_b

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self._env.base_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self._env.base_ang_vel_b[:, 2]) / max_command_step
        )

    def update_curriculum(self, iteration: float) -> None:
        """Record the current rsl-rl iteration so per-axis schedules can interpolate."""
        self.current_iteration = float(iteration)

    def curriculum_log(self) -> dict[str, float]:
        """Snapshot the schedule and effective resampling bounds for RL loggers.

        Bounds describe newly sampled commands, not the commands currently held
        by every environment. Unscheduled/disabled axes have no progress metric.
        """
        result = {
            "Curriculum/enabled": float(self.cfg.curriculum.enable),
            "Curriculum/iteration": self.current_iteration,
        }
        if (self.cfg.rel_in_place_turn_envs > 0 or self.cfg.rel_nominal_height_envs > 0
                or self.cfg.rel_high_stance_move_turn_envs > 0 or self.cfg.rel_low_stance_move_envs > 0):
            result.update({
                "CommandMix/standing_fraction": float(self.is_standing_env.float().mean()),
                "CommandMix/in_place_turn_fraction": float(self.is_in_place_turn_env.float().mean()),
                "CommandMix/nominal_height_fraction": float(self.is_nominal_height_env.float().mean()),
            })
        if self.cfg.rel_high_stance_turn_envs + self.cfg.rel_high_stance_move_turn_envs + self.cfg.rel_low_stance_move_envs > 0:
            result.update({
                "CommandMix/high_stance_turn_fraction": float(self.is_high_stance_turn_env.float().mean()),
                "CommandMix/general_in_place_turn_fraction": float(
                    (self.is_in_place_turn_env & ~self.is_high_stance_turn_env).float().mean()),
                "CommandMix/high_stance_move_turn_fraction": float(self.is_high_stance_move_turn_env.float().mean()),
                "CommandMix/general_moving_fraction": float(
                    (~self.is_standing_env & ~self.is_in_place_turn_env & ~self.is_high_stance_move_turn_env
                     & ~self.is_low_stance_move_env).float().mean()),
            })
        if self.cfg.rel_low_stance_move_envs > 0:
            low = self.is_low_stance_move_env
            result['CommandMix/low_stance_move_fraction'] = float(low.float().mean())
            result['CommandMix/low_stance_turn_fraction_of_low'] = float(
                ((self.command[:, 2].abs() > 1e-6) & low).sum() / low.sum().clamp_min(1))
            result['CommandMix/low_stance_zero_pose_fraction_of_low'] = float(
                ((self.command[:, 4:7].abs().sum(-1) < 1e-6) & low).sum() / low.sum().clamp_min(1))
        axes = ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
        if self.cfg.body_command and self.cfg.rel_zero_pose_envs + self.cfg.rel_single_axis_pose_envs > 0:
            pose_axes = (self.command[:, 4:7].abs() > 1e-6).sum(-1)
            result['CommandMix/zero_pose_fraction'] = float((pose_axes == 0).float().mean())
            result['CommandMix/single_axis_pose_fraction'] = float((pose_axes == 1).float().mean())
        if self.cfg.in_place_small_turn_fraction > 0 and self.cfg.rel_in_place_turn_envs > 0:
            turns = self.is_in_place_turn_env
            result['CommandMix/small_turn_fraction_of_turns'] = float(
                ((self.command[:, 2].abs() <= .6) & turns).sum() / turns.sum().clamp_min(1))
        if self.cfg.body_command:
            axes += ["body_height", "body_roll", "body_pitch", "body_yaw"]
        for axis in axes:
            base_range = getattr(self.cfg.ranges, axis, None)
            if base_range is None:
                continue
            low, high = self._effective_range(axis, base_range)
            result[f"Curriculum/{axis}/min"] = float(low)
            result[f"Curriculum/{axis}/max"] = float(high)
            schedule = getattr(self.cfg.curriculum, axis, None)
            if self.cfg.curriculum.enable and schedule is not None:
                if self.current_iteration <= schedule.start_iter:
                    progress = 0.0
                elif self.current_iteration >= schedule.end_iter:
                    progress = 1.0
                else:
                    span = max(int(schedule.end_iter) - int(schedule.start_iter), 1)
                    progress = (self.current_iteration - schedule.start_iter) / span
                result[f"Curriculum/{axis}/progress_percent"] = 100.0 * progress
        return result

    def _effective_range(self, axis_name: str, base_range: tuple[float, float]) -> tuple[float, float]:
        """Return the curriculum-aware sampling range for ``axis_name``.

        When the curriculum is disabled or no schedule is set for the axis, the
        static ``base_range`` is returned unchanged. Otherwise the range is
        linearly interpolated between ``start_range`` and ``end_range`` over
        ``[start_iter, end_iter]``.
        """
        if not self.cfg.curriculum.enable:
            return base_range
        axis_cfg = getattr(self.cfg.curriculum, axis_name, None)
        if axis_cfg is None:
            return base_range
        iteration = self.current_iteration
        if iteration <= axis_cfg.start_iter:
            return tuple(axis_cfg.start_range)
        if iteration >= axis_cfg.end_iter:
            return tuple(axis_cfg.end_range)
        span = max(int(axis_cfg.end_iter) - int(axis_cfg.start_iter), 1)
        progress = (iteration - axis_cfg.start_iter) / span
        low = axis_cfg.start_range[0] + (axis_cfg.end_range[0] - axis_cfg.start_range[0]) * progress
        high = axis_cfg.start_range[1] + (axis_cfg.end_range[1] - axis_cfg.start_range[1]) * progress
        return (low, high)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.is_nominal_height_env[env_ids] = False
        self.vel_command_b[env_ids, 0] = r.uniform_(*self._effective_range("lin_vel_x", self.cfg.ranges.lin_vel_x))
        self.vel_command_b[env_ids, 1] = r.uniform_(*self._effective_range("lin_vel_y", self.cfg.ranges.lin_vel_y))
        self.vel_command_b[env_ids, 2] = r.uniform_(*self._effective_range("ang_vel_z", self.cfg.ranges.ang_vel_z))
        if self.cfg.body_command:
            if hasattr(self.cfg.ranges, "body_height"):
                self.vel_command_b[env_ids, 3] = r.uniform_(
                    *self._effective_range("body_height", self.cfg.ranges.body_height)
                )
                if self.cfg.rel_nominal_height_envs > 0:
                    self.is_nominal_height_env[env_ids] = r.uniform_(0., 1.) < self.cfg.rel_nominal_height_envs
                    self.vel_command_b[env_ids, 3] = torch.where(
                        self.is_nominal_height_env[env_ids], self.cfg.nominal_body_height,
                        self.vel_command_b[env_ids, 3],
                    )
            if hasattr(self.cfg.ranges, "body_roll"):
                self.vel_command_b[env_ids, 4] = r.uniform_(
                    *self._effective_range("body_roll", self.cfg.ranges.body_roll)
                )
            if hasattr(self.cfg.ranges, "body_pitch"):
                self.vel_command_b[env_ids, 5] = r.uniform_(
                    *self._effective_range("body_pitch", self.cfg.ranges.body_pitch)
                )
            if hasattr(self.cfg.ranges, "body_yaw"):
                self.vel_command_b[env_ids, 6] = r.uniform_(
                    *self._effective_range("body_yaw", self.cfg.ranges.body_yaw)
                )
        self.is_heading_env[env_ids] = False
        if self.cfg.body_command and self.cfg.rel_zero_pose_envs + self.cfg.rel_single_axis_pose_envs > 0:
            self.vel_command_b[env_ids, 4:7] = mix_pose_samples(
                self.vel_command_b[env_ids, 4:7], self.cfg.rel_zero_pose_envs, self.cfg.rel_single_axis_pose_envs)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.rel_heading_envs
        mode_draw = r.uniform_(0.0, 1.0)
        self.is_standing_env[env_ids] = mode_draw < self.cfg.rel_standing_envs
        self.is_in_place_turn_env[env_ids] = (
            (mode_draw >= self.cfg.rel_standing_envs)
            & (mode_draw < self.cfg.rel_standing_envs + self.cfg.rel_in_place_turn_envs)
        )
        self.is_high_stance_turn_env[env_ids] = (
            (mode_draw >= self.cfg.rel_standing_envs)
            & (mode_draw < self.cfg.rel_standing_envs + self.cfg.rel_high_stance_turn_envs)
        )
        moving_start = self.cfg.rel_standing_envs + self.cfg.rel_in_place_turn_envs
        self.is_high_stance_move_turn_env[env_ids] = (
            (mode_draw >= moving_start)
            & (mode_draw < moving_start + self.cfg.rel_high_stance_move_turn_envs)
        )
        low_start = moving_start + self.cfg.rel_high_stance_move_turn_envs
        self.is_low_stance_move_env[env_ids] = (
            (mode_draw >= low_start)
            & (mode_draw < low_start + self.cfg.rel_low_stance_move_envs)
        )
        self.is_heading_env[env_ids] &= ~self.is_in_place_turn_env[env_ids]
        if self.cfg.in_place_small_turn_fraction > 0 and self.cfg.rel_in_place_turn_envs > 0:
            turn_rates = small_turn_samples(len(env_ids), self.device, self.cfg.in_place_small_turn_fraction,
                                            max_rate=self.cfg.ranges.ang_vel_z[1])
            self.vel_command_b[env_ids, 2] = torch.where(
                self.is_in_place_turn_env[env_ids], turn_rates, self.vel_command_b[env_ids, 2])

        direct_moving_envs = (~self.is_heading_env[env_ids] & ~self.is_standing_env[env_ids]
                              & ~self.is_in_place_turn_env[env_ids])
        if self.cfg.rel_zero_vel_yaw_envs <= 0.0:
            self.is_zero_vel_yaw_env[env_ids] = False
        elif self.cfg.rel_zero_vel_yaw_envs >= 1.0:
            self.is_zero_vel_yaw_env[env_ids] = direct_moving_envs
        else:
            zero_yaw_rate_draw = r.uniform_(0.0, 1.0) < self.cfg.rel_zero_vel_yaw_envs
            self.is_zero_vel_yaw_env[env_ids] = direct_moving_envs & zero_yaw_rate_draw

        if self.cfg.rel_high_stance_turn_envs + self.cfg.rel_high_stance_move_turn_envs > 0:
            selected_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            for mask, moving in ((self.is_high_stance_turn_env, False),
                                 (self.is_high_stance_move_turn_env, True)):
                ids = selected_ids[mask[selected_ids]]
                if ids.numel() == 0:
                    continue
                self.vel_command_b[ids] = high_stance_commands(
                    ids.numel(), self.device, moving, self.cfg.high_stance_height_range,
                    self.cfg.high_stance_speed_range, self.cfg.high_stance_yaw_rate_range)
                self.is_heading_env[ids] = False
                self.is_zero_vel_yaw_env[ids] = False
                self.is_nominal_height_env[ids] = False

        if self.cfg.rel_low_stance_move_envs > 0:
            selected_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            ids = selected_ids[self.is_low_stance_move_env[selected_ids]]
            if ids.numel() > 0:
                self.vel_command_b[ids] = low_stance_commands(
                    self.vel_command_b[ids], self.cfg.low_stance_height_range,
                    self.cfg.low_stance_speed_range, self.cfg.low_stance_yaw_rate_range,
                    self.cfg.low_stance_turn_fraction)
                self.is_heading_env[ids] = False
                self.is_zero_vel_yaw_env[ids] = False
                self.is_nominal_height_env[ids] = False

        # CommandTerm.reset() resamples without calling _update_command(). Apply
        # the masks here so the first post-reset observation sees effective commands.
        self._apply_command_postprocessing(env_ids)

    def _validate_low_stance_ranges(self):
        if not self.cfg.body_command:
            raise ValueError("Low-stance sampling requires body commands.")
        bounds = self.cfg.ranges
        h_lo, h_hi = self.cfg.low_stance_height_range
        v_lo, v_hi = self.cfg.low_stance_speed_range
        w_lo, w_hi = self.cfg.low_stance_yaw_rate_range
        if not (bounds.body_height[0] <= h_lo <= h_hi <= bounds.body_height[1]):
            raise ValueError("Low-stance height range must lie within body_height bounds.")
        if not (0 < v_lo <= v_hi and bounds.lin_vel_x[0] <= v_lo <= v_hi <= bounds.lin_vel_x[1]
                and bounds.lin_vel_y[0] <= -v_hi and v_hi <= bounds.lin_vel_y[1]):
            raise ValueError("Low-stance forward/left/right speeds must lie within XY bounds.")
        if not (0 < w_lo <= w_hi and bounds.ang_vel_z[0] <= -w_hi and w_hi <= bounds.ang_vel_z[1]):
            raise ValueError("Low-stance signed yaw rates must lie within ang_vel_z bounds.")
        if any(getattr(self.cfg.curriculum, name) is not None for name in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')):
            raise ValueError("Low-stance movement requires fixed velocity ranges.")
        height_schedule = self.cfg.curriculum.body_height
        if height_schedule is not None and any(
            not (limits[0] <= h_lo <= h_hi <= limits[1])
            for limits in (height_schedule.start_range, height_schedule.end_range)
        ):
            raise ValueError("Low-stance heights must fit throughout the height curriculum.")

    def _validate_high_stance_ranges(self):
        """Reject specialist commands outside the task's configured limits."""
        if not self.cfg.body_command:
            raise ValueError("High-stance sampling requires body commands.")
        h_lo, h_hi = self.cfg.high_stance_height_range
        v_lo, v_hi = self.cfg.high_stance_speed_range
        w_lo, w_hi = self.cfg.high_stance_yaw_rate_range
        bounds = self.cfg.ranges
        if not (bounds.body_height[0] <= h_lo <= h_hi <= bounds.body_height[1]):
            raise ValueError("High-stance height range must lie within body_height bounds.")
        if not (0 < w_lo <= w_hi and bounds.ang_vel_z[0] <= -w_hi and w_hi <= bounds.ang_vel_z[1]):
            raise ValueError("High-stance signed yaw rates must lie within ang_vel_z bounds.")
        if self.cfg.rel_high_stance_move_turn_envs > 0 and not (
            0 < v_lo <= v_hi and bounds.lin_vel_x[0] <= v_lo <= v_hi <= bounds.lin_vel_x[1]
            and bounds.lin_vel_y[0] <= -v_hi and v_hi <= bounds.lin_vel_y[1]
        ):
            raise ValueError("High-stance forward/left/right speeds must lie within XY bounds.")

    def _update_command(self):
        """Post-process the velocity command for every environment."""
        self._apply_command_postprocessing(slice(None))

    def _apply_command_postprocessing(self, env_ids: Sequence[int] | slice):
        """Apply heading, direct-zero-yaw, and standing masks to selected environments."""
        if self.cfg.heading_command:
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self._env.base_heading_w[env_ids]
            )
            heading_yaw_rate = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
            self.vel_command_b[env_ids, 2] = torch.where(
                self.is_heading_env[env_ids],
                heading_yaw_rate,
                self.vel_command_b[env_ids, 2],
            )

        self.vel_command_b[env_ids, 2] = torch.where(
            self.is_zero_vel_yaw_env[env_ids],
            torch.zeros_like(self.vel_command_b[env_ids, 2]),
            self.vel_command_b[env_ids, 2],
        )
        self.vel_command_b[env_ids, :3] = torch.where(
            self.is_standing_env[env_ids].unsqueeze(-1),
            torch.zeros_like(self.vel_command_b[env_ids, :3]),
            self.vel_command_b[env_ids, :3],
        )
        self.vel_command_b[env_ids, :2] = torch.where(
            self.is_in_place_turn_env[env_ids].unsqueeze(-1),
            torch.zeros_like(self.vel_command_b[env_ids, :2]),
            self.vel_command_b[env_ids, :2],
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self._env.base_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self._env.base_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
