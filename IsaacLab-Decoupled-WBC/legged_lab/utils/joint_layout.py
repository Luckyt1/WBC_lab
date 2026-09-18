"""Resolve policy joint order independently of the articulation's USD order."""

import re


def resolve_joint_layout(joint_names, action_joint_names, arm_joint_names):
    """Return policy-first joint indices and their inverse, holding extras at default."""
    ordered_names = list(action_joint_names) + list(arm_joint_names)
    if not action_joint_names:
        raise ValueError("robot.action_joint_names must specify the policy's joint order")
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("Action and arm joint names must be unique and disjoint")
    missing = set(ordered_names) - set(joint_names)
    if missing:
        raise ValueError(f"Configured joints missing from robot asset: {sorted(missing)}")
    ordered_names.extend(name for name in joint_names if name not in ordered_names)
    indices = [joint_names.index(name) for name in ordered_names]
    inverse = [indices.index(i) for i in range(len(joint_names))]
    return indices, inverse


def resolve_action_scales(action_joint_names, scales):
    """Expand regex gains in policy order; never silently scale the wrong motor."""
    values = []
    for name in action_joint_names:
        matches = [value for pattern, value in scales.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"Action scale for {name!r} must match exactly one pattern, got {len(matches)}")
        values.append(matches[0])
    return values
