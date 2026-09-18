"""Torch-only kernels for the opt-in ELF3 tracking recovery experiment."""

import torch


def low_stance_commands(command, height_range, speed_range, yaw_rate_range, turn_fraction=.5):
    """Sample low forward/left/right motion, preserving the existing torso pose.

    Half the default samples translate without turning, the rest turn with
    balanced signs. Input is not mutated; all three translation axes are drawn
    with equal probability. Backward/diagonal motion remains in the general mix.
    """
    result = command.clone()
    count, device = result.shape[0], result.device
    result[:, :3] = 0.
    result[:, 3].uniform_(*height_range)
    direction = torch.randint(3, (count,), device=device)
    speed = torch.empty(count, device=device).uniform_(*speed_range)
    result[:, 0] = torch.where(direction == 0, speed, 0.)
    result[:, 1] = torch.where(direction == 1, speed, torch.where(direction == 2, -speed, 0.))
    turn = torch.rand(count, device=device) < turn_fraction
    sign = torch.where(torch.rand(count, device=device) < .5, -1., 1.)
    yaw = torch.empty(count, device=device).uniform_(*yaw_rate_range) * sign
    result[:, 2] = torch.where(turn, yaw, 0.)
    return result


def high_stance_commands(count, device, moving, height_range, speed_range, yaw_rate_range):
    """Zero torso pose, balanced turn signs; moving samples go forward/left/right."""
    command = torch.zeros(count, 7, device=device)
    command[:, 3].uniform_(*height_range)
    sign = torch.where(torch.rand(count, device=device) < .5, -1., 1.)
    command[:, 2] = torch.empty(count, device=device).uniform_(*yaw_rate_range) * sign
    if moving:
        direction = torch.randint(3, (count,), device=device)
        speed = torch.empty(count, device=device).uniform_(*speed_range)
        command[:, 0] = torch.where(direction == 0, speed, 0.)
        command[:, 1] = torch.where(direction == 1, speed, torch.where(direction == 2, -speed, 0.))
    return command


def small_turn_samples(count, device, small_fraction=.7, small_range=(.15, .6), max_rate=1.57):
    """Sample both signs, with deliberate coverage of small nonzero yaw rates."""
    small = torch.rand(count, device=device) < small_fraction
    draw = torch.rand(count, device=device)
    magnitude = torch.where(small, small_range[0] + draw * (small_range[1] - small_range[0]),
                            small_range[1] + draw * (max_rate - small_range[1]))
    sign = torch.where(torch.rand(count, device=device) < .5, -1., 1.)
    return magnitude * sign


def mix_pose_samples(pose, zero_fraction, single_fraction):
    """Keep the sampled ranges, masking zero/single-axis tasks independently of motion."""
    draw = torch.rand(pose.shape[0], device=pose.device)
    axis = torch.randint(3, (pose.shape[0],), device=pose.device)
    single = (draw >= zero_fraction) & (draw < zero_fraction + single_fraction)
    active = torch.ones_like(pose, dtype=torch.bool)
    active[draw < zero_fraction] = False
    active[single] = torch.nn.functional.one_hot(axis[single], 3).bool()
    return pose * active


def yaw_rate_tracking(command, measured, min_std=.15, max_std=.3, command_scale=.5):
    width = (command_scale * command.abs()).clamp(min=min_std, max=max_std)
    return torch.exp(-((measured - command) / width).square())


def angular_tracking(measured, command, std):
    delta = measured - command
    error = torch.atan2(torch.sin(delta), torch.cos(delta))
    return torch.exp(-(error / std).square())


def zero_motion(command, threshold=.1):
    return command[:, :2].norm(dim=-1) + command[:, 2].abs() <= threshold


def smooth_stance(contact_time, command, ramp_s=.15):
    return torch.tanh(contact_time.amin(dim=-1) / ramp_s) * zero_motion(command)


def standing_foot_lift(contact_time, command):
    return (contact_time <= 0).float().mean(dim=-1) * zero_motion(command)
