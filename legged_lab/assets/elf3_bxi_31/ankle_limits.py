"""Requested bilateral ankle speed limits, in rad/s, for the restored recipe."""

ANKLE_VELOCITY_LIMITS_RAD_S = {
    "l_ankle_x_joint": 5.0,
    "r_ankle_x_joint": 5.0,
    "l_ankle_y_joint": 15.0,
    "r_ankle_y_joint": 15.0,
}


def joint_velocity_limits(joints):
    limits = {name: values["limits"]["velocity"] for name, values in joints.items()}
    missing = set(ANKLE_VELOCITY_LIMITS_RAD_S) - set(limits)
    if missing:
        raise ValueError(f"Missing ankle joints: {sorted(missing)}")
    limits.update(ANKLE_VELOCITY_LIMITS_RAD_S)
    return limits


def verify_neutral_waist(commands):
    """Reject waist-pose sampling while retaining the robot's locomotion turns."""
    if getattr(commands, "waist_rpy", None) is not None:
        raise ValueError("This restored training recipe excludes waist yaw/pitch/roll commands.")
    for name in ("waist_yaw_initial_range", "waist_yaw_range", "sampling_waist_yaw_range"):
        value = getattr(commands, name)
        if value is not None and tuple(value) != (0.0, 0.0):
            raise ValueError(f"This restored training recipe requires {name} = (0, 0).")
    if commands.rel_zero_waist_yaw != 1.0:
        raise ValueError("All waist pose commands must remain neutral.")


def verify_simulator_limits(robot):
    """Read back the PhysX solver limits before any policy learning begins."""
    import torch

    limits = robot.root_physx_view.get_dof_max_velocities()
    actual = {}
    for name, expected in ANKLE_VELOCITY_LIMITS_RAD_S.items():
        if name not in robot.joint_names:
            raise ValueError(f"Simulator is missing ankle joint {name}.")
        values = limits[:, robot.joint_names.index(name)]
        if not bool(torch.isfinite(values).all() and ((values - expected).abs() < 1e-5).all()):
            raise ValueError(f"PhysX ankle velocity limit mismatch for {name}; expected {expected} rad/s.")
        actual[name] = float(values[0])
    return actual
