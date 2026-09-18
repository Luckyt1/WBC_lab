"""Tensor-only HOMIE squat objectives; see docs/elf3_homie_squat_training.md.

Independent implementation of OpenHomie's height-coupled knee objective:
InternRobotics/OpenHomie, commit 33071f79f1884bb122bd45631c8c60d61dac1eb8,
HomieRL/legged_gym/legged_gym/envs/base/legged_robot.py,
_reward_deviation_knee_joint and the height-gated posture objectives.
"""

import torch


def height_coupled_knee_cost(
    knee_positions: torch.Tensor,
    joint_limits: torch.Tensor,
    height_error: torch.Tensor,
) -> torch.Tensor:
    """Nonnegative cost; limits are hard physical limits, not action scales.

    When height error is nonzero, prefer the midpoint of each knee's range.
    This is not a signed inverse-kinematics target for the commanded height.
    """
    lower, upper = joint_limits.unbind(dim=-1)
    span = (upper - lower).clamp_min(1e-6)
    midpoint_distance = ((knee_positions - lower) / span - 0.5).abs()
    return midpoint_distance.sum(dim=-1) * height_error.abs()


def standing_posture_cost(
    positions: torch.Tensor,
    default_positions: torch.Tensor,
    height_command: torch.Tensor,
    standing_height: float,
) -> torch.Tensor:
    """Keep existing L1 posture cost only at/above the standing threshold."""
    return (positions - default_positions).abs().sum(dim=-1) * (height_command >= standing_height)
