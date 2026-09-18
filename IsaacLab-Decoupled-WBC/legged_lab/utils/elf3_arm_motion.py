"""Joint-space transfer of G1 AMASS arm motion to the ELF3 arm chains.

This transfers joint motion, not Cartesian hand trajectories between robots with
different link lengths. The nominal shoulder mounting tilt is compensated before
applying ELF3 limits. Both robots use positive pitch=Y, roll=X, yaw=Z axes.
"""

import math

import torch


ELF3_AMASS_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("l", "r")
    for joint in ("shoulder_y", "shoulder_x", "shoulder_z", "elbow_y", "wrist_x", "wrist_y", "wrist_z")
]


def map_g1_arms_to_elf3(poses: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    """Map [..., 14] AMASS columns 22:36 into the configured ELF3 order.

    G1 shoulder orientation: Rx(a) Ry(pitch) Rx(roll-a) Rz(yaw),
    a=+16 degrees on the left and -16 on the right (nominal USD mount).
    ELF3 shoulder orientation: Ry(pitch) Rx(roll) Rz(yaw).
    Elbow and wrist axes and their zero-pose directions already correspond.
    """
    if poses.shape[-1] != 14:
        raise ValueError("AMASS requires 14 arm columns")
    if len(joint_names) != 14 or set(joint_names) != set(ELF3_AMASS_JOINT_NAMES):
        raise ValueError("amass_elf3 requires all 14 named ELF3 arm joints")
    mapped = poses.clone()
    for offset, angle in ((0, math.radians(16)), (7, math.radians(-16))):
        p, r, y = poses[..., offset:offset + 3].unbind(-1)
        cp, sp = p.cos(), p.sin()
        cb, sb = (r - angle).cos(), (r - angle).sin()
        cy, sy = y.cos(), y.sin()
        ca, sa = math.cos(angle), math.sin(angle)
        # Only the five entries needed to extract the Y-X-Z Euler angles.
        r02 = sp * cb
        r12 = -ca * sb - sa * cp * cb
        r22 = -sa * sb + ca * cp * cb
        r10 = ca * cb * sy + sa * sp * cy - sa * cp * sb * sy
        r11 = ca * cb * cy - sa * sp * sy - sa * cp * sb * cy
        mapped[..., offset] = torch.atan2(r02, r22)
        mapped[..., offset + 1] = torch.asin((-r12).clamp(-1, 1))
        mapped[..., offset + 2] = torch.atan2(r10, r11)
    indices = [ELF3_AMASS_JOINT_NAMES.index(name) for name in joint_names]
    return mapped[..., indices]


class Elf3ArmMotion:
    """Mapped and bounded AMASS targets, with optional ramp and rate limiting."""

    def __init__(self, joint_names, defaults, limits, dt, scale=1.0, ramp_s=1.0, max_velocity=2.0):
        if not all(math.isfinite(x) for x in (dt, scale, ramp_s)):
            raise ValueError("Arm motion parameters must be finite")
        if dt <= 0 or not 0 <= scale <= 1 or ramp_s < 0:
            raise ValueError("Arm motion requires dt>0, 0<=scale<=1, ramp_s>=0")
        if max_velocity is not None and (not math.isfinite(max_velocity) or max_velocity <= 0):
            raise ValueError("max_velocity must be positive and finite, or None to disable")
        map_g1_arms_to_elf3(defaults, joint_names)  # validate the named layout
        self.joint_names = list(joint_names)
        self.defaults = defaults.clone()
        self.limits = limits.clone()
        self.dt, self.scale, self.ramp_s = dt, scale, ramp_s
        self.max_delta = None if max_velocity is None else max_velocity * dt
        self.target = defaults.clamp(limits[..., 0], limits[..., 1]).clone()
        self.start = self.target.clone()
        self.elapsed = torch.zeros(defaults.shape[0], device=defaults.device)
        self.clipped_fraction = torch.zeros((), device=defaults.device)

    def reset(self, env_ids, positions):
        self.target[env_ids] = positions.clamp(self.limits[env_ids, :, 0], self.limits[env_ids, :, 1])
        self.start[env_ids] = self.target[env_ids]
        self.elapsed[env_ids] = 0

    def step(self, poses):
        desired = map_g1_arms_to_elf3(poses, self.joint_names)
        desired = self.defaults + self.scale * (desired - self.defaults)
        clipped = desired.clamp(self.limits[..., 0], self.limits[..., 1])
        self.clipped_fraction = (desired != clipped).float().mean()
        self.elapsed += self.dt
        desired = clipped
        if self.ramp_s > 0:
            alpha = (self.elapsed / self.ramp_s).clamp(0, 1)
            alpha = alpha.square() * (3 - 2 * alpha)
            desired = self.start + alpha[:, None] * (clipped - self.start)
        if self.max_delta is None:
            self.target.copy_(desired)
        else:
            self.target += (desired - self.target).clamp(-self.max_delta, self.max_delta)
        return self.target
