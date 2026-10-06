"""Joint-feasible policy action bounds shared by Tang training and inference."""

import math

import torch


def policy_action_limits(default_pos, joint_limits, action_scale, clip_actions):
    """Invert target = default + scale * action, in policy joint order.

    Intersect physical bounds with the configured normalized action cap. Zero
    must be feasible because delay/history buffers are initialized to zero.
    This runs once at environment initialization, not on every control step.
    """
    if not math.isfinite(clip_actions) or clip_actions <= 0:
        raise ValueError("clip_actions must be finite and positive")
    if not torch.isfinite(action_scale).all() or (action_scale == 0).any():
        raise ValueError("action_scale must be finite and nonzero")
    if not torch.isfinite(default_pos).all() or not torch.isfinite(joint_limits).all():
        raise ValueError("Policy joint defaults and soft limits must be finite")
    lower, upper = joint_limits.unbind(-1)
    if ((lower > default_pos) | (default_pos > upper)).any():
        raise ValueError("Policy joint defaults must lie within their soft position limits")
    a = (lower - default_pos) / action_scale
    b = (upper - default_pos) / action_scale
    return torch.minimum(a, b).clamp(min=-clip_actions), torch.maximum(a, b).clamp(max=clip_actions)
