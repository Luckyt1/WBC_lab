"""Costs on raw policy targets, before global action clipping."""

import torch


def raw_target_limit_excess(
    actions: torch.Tensor,
    default_positions: torch.Tensor,
    action_scale: float | torch.Tensor,
    joint_limits: torch.Tensor,
) -> torch.Tensor:
    """Per-joint excess in radians; retain a signal arbitrarily far out of bounds.

    In-range targets, including either boundary, have zero cost. No clipping or
    upper cap is applied here: that would recreate the saturation dead zone.
    """
    target = default_positions + actions * action_scale
    lower, upper = joint_limits.unbind(-1)
    return torch.relu(lower - target) + torch.relu(target - upper)
