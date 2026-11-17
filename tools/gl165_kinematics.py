"""URDF-derived arm kinematics for GL-165 IK data generation (CPU only)."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from legged_lab.assets.gl165.model import ARM_JOINT_NAMES, DEFAULT_POSITIONS


class ArmChain:
    def __init__(self, urdf, side):
        root = ET.parse(urdf).getroot()
        self.names = ARM_JOINT_NAMES[:7] if side == "l" else ARM_JOINT_NAMES[7:]
        self.origins, self.rotations, self.axes, self.limits = [], [], [], []
        parent = "body_y_link"
        for name in self.names:
            joint = root.find(f"joint[@name='{name}']")
            if joint is None or joint.find("parent").get("link") != parent:
                raise ValueError(f"Invalid arm chain at {name}")
            parent = joint.find("child").get("link")
            origin = joint.find("origin")
            self.origins.append(np.fromstring(origin.get("xyz", "0 0 0"), sep=" "))
            self.rotations.append(Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix())
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            self.axes.append(axis / np.linalg.norm(axis))
            limit = joint.find("limit")
            self.limits.append([float(limit.get("lower")) + .02, float(limit.get("upper")) - .02])
        self.limits = np.array(self.limits)
        self.home = np.array([DEFAULT_POSITIONS[n] for n in self.names])
        self.sign = 1 if side == "l" else -1

    def fk(self, q):
        position, rotation = np.zeros(3), np.eye(3)
        points = []
        for angle, origin, basis, axis in zip(q, self.origins, self.rotations, self.axes):
            position = position + rotation @ origin
            rotation = rotation @ basis @ Rotation.from_rotvec(axis * angle).as_matrix()
            points.append(position.copy())
        return position, rotation, np.array(points)

    def solve(self, position, rotation, initial=None):
        def residual(q):
            actual_p, actual_r, _ = self.fk(q)
            return np.r_[actual_p - position,
                         .15 * Rotation.from_matrix(rotation.T @ actual_r).as_rotvec(),
                         .001 * (q - self.home)]

        result = least_squares(residual, self.home if initial is None else initial,
                               bounds=(self.limits[:, 0], self.limits[:, 1]),
                               max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        actual_p, actual_r, _ = self.fk(result.x)
        error = np.linalg.norm(actual_p - position)
        angle = np.linalg.norm(Rotation.from_matrix(rotation.T @ actual_r).as_rotvec())
        if not result.success or error > .005 or angle > .05:
            return None
        return result.x, error, angle

    def sample(self, rng):
        # Solve Cartesian poses generated inside a bounded, reachable workspace.
        # These bounds deliberately keep the hand in front and outside the torso.
        for _ in range(1000):
            seed = self.home + rng.uniform(-1, 1, 7) * np.array([.65, .40, .65, .55, .6, .3, .3])
            seed[1] = self.sign * rng.uniform(.15, .7)
            seed = np.clip(seed, self.limits[:, 0], self.limits[:, 1])
            position, rotation, _ = self.fk(seed)
            if position[0] < .07 or self.sign * position[1] < .16:
                continue
            solved = self.solve(position, rotation)
            if solved is None:
                continue
            q, error, angle = solved
            # Check the whole nominal interpolation, not just its endpoint.
            # This is a workspace screen, not a full mesh collision certificate.
            safe = True
            for fraction in np.linspace(0, 1, 33):
                p, _, points = self.fk(self.home + fraction * (q - self.home))
                if p[0] < .05 or np.any(self.sign * points[3:, 1] < .14):
                    safe = False
                    break
            if safe:
                return q, position, rotation, error, angle
        raise RuntimeError("Could not sample a valid GL-165 arm IK target")

    def sample_overhead(self, rng, min_height, max_height):
        """Generate torso-relative wrist goals above the head, within URDF limits.

        The caller must screen complete two-arm transitions against robot geometry.
        """
        for _ in range(1000):
            seed = np.array([rng.uniform(-2.6, -1.8), self.sign * rng.uniform(.12, .55),
                             rng.uniform(-.35, .35), rng.uniform(.08, .65),
                             rng.uniform(-.4, .4), rng.uniform(-.25, .25), rng.uniform(-.25, .25)])
            seed = np.clip(seed, self.limits[:, 0], self.limits[:, 1])
            position, rotation, _ = self.fk(seed)
            if not (min_height <= position[2] <= max_height and .07 <= position[0] <= .35
                    and .17 <= self.sign * position[1] <= .45):
                continue
            solved = self.solve(position, rotation, initial=seed)
            if solved is not None:
                q, error, angle = solved
                return q, position, rotation, error, angle
        raise RuntimeError('Could not sample an overhead GL165 wrist pose')

    def sample_lateral(self, rng, height_range):
        """Side-reaching wrist poses, with height measured from the shoulder.

        Full robot collision checks are performed by the bank builder.
        """
        for _ in range(1000):
            seed = np.array([rng.uniform(-.8, .8), self.sign * rng.uniform(.8, 2.2),
                             rng.uniform(-1.6, 1.6), rng.uniform(-.2, 1.1),
                             rng.uniform(-.4, .4), rng.uniform(-.25, .25), rng.uniform(-.25, .25)])
            seed = np.clip(seed, self.limits[:, 0], self.limits[:, 1])
            position, rotation, _ = self.fk(seed)
            delta = position - self.origins[0]
            if not (-.12 <= delta[0] <= .16 and .35 <= self.sign * delta[1] <= .53
                    and height_range[0] <= delta[2] <= height_range[1]):
                continue
            solved = self.solve(position, rotation, initial=seed)
            if solved is None:
                continue
            q, error, angle = solved
            actual, actual_rotation, _ = self.fk(q)
            delta = actual - self.origins[0]
            if (-.12 <= delta[0] <= .16 and .35 <= self.sign * delta[1] <= .53
                    and height_range[0] <= delta[2] <= height_range[1]):
                return q, actual, actual_rotation, error, angle
        raise RuntimeError('Could not sample a lateral GL165 wrist pose')
