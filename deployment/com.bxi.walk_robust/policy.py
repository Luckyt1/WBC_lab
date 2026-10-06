"""Offline ELF3 actor adapter; load an explicitly supplied ONNX model."""

from pathlib import Path
import hashlib
import json

import numpy as np

from bxi_example_py_elf3.framework.inference import JointPolicy, ModelSpec, PolicyJointContract
from bxi_example_py_elf3.framework.inference.runtime import default_runtime
from bxi_example_py_elf3.framework.joints import JointLayout


ASSETS = Path(__file__).resolve().parent / "assets"
CONTROL = json.loads((ASSETS / "control_parameters.json").read_text())
OBS_NAMES = tuple(CONTROL["policy_joint_names"] + CONTROL["arm_joint_names"])
OUTPUT_NAMES = tuple(CONTROL["deployment_joint_names"])


def pelvis_imu(quat_wxyz, torso_omega, waist_q, waist_dq):
    """Convert torso IMU to waist_z_link, including relative joint velocity.

    The checked ELF3 URDF/MJCF chain is torso -> Ry -> Rx -> Rz, with no
    fixed rotations. Quaternion maps torso-local vectors to world (wxyz).
    """
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("expected finite torso quaternion wxyz")
    norm = np.linalg.norm(quat)
    if norm < 1e-6:
        raise ValueError("torso quaternion has zero norm")
    w, x, y, z = quat / norm
    gravity_torso = np.array([2 * (w*y - x*z), -2 * (y*z + w*x),
                              2 * (x*x + y*y) - 1])
    cy, cx, cz = np.cos(waist_q)
    sy, sx, sz = np.sin(waist_q)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rotation = ry @ rx @ rz
    relative_omega = (np.array([0, waist_dq[0], 0])
                      + ry[:, 0] * waist_dq[1]
                      + (ry @ rx)[:, 2] * waist_dq[2])
    return rotation.T @ (torso_omega + relative_omega), rotation.T @ gravity_torso


class WalkRobustPolicy(JointPolicy):
    joint_contract = PolicyJointContract(
        observation=JointLayout.create(OBS_NAMES, label="walk_robust_encoders"),
        action=JointLayout.create(OUTPUT_NAMES, label="elf3_31"),
    )

    def __init__(self, assets=ASSETS, *, onnx=None):
        super().__init__()
        assets = Path(assets)
        self.info = json.loads((assets / "model_info.json").read_text())
        for filename, key in (("control_parameters.json", "control_parameters_sha256"),):
            if hashlib.sha256((assets / filename).read_bytes()).hexdigest() != self.info[key]:
                raise ValueError(f"pinned model asset checksum mismatch: {filename}")
        parameters = json.loads((assets / "control_parameters.json").read_text())
        if parameters != CONTROL:
            raise ValueError("control parameters do not match the joint contract")
        if onnx is None:
            raise ValueError('Supply an exported ONNX model explicitly; this repository ships no pretrained policy')
        onnx = Path(onnx)
        joints = [parameters["joints"][name] for name in OUTPUT_NAMES]
        self.default = np.array([j["default_position"] for j in joints], dtype=np.float32)
        self.kps = np.array([j["kp"] for j in joints], dtype=np.float32)
        self.kds = np.array([j["kd"] for j in joints], dtype=np.float32)
        self.scale = np.array([j["action_scale"] for j in joints[:15]], dtype=np.float32)
        hard = np.array([[j["limits"]["lower"], j["limits"]["upper"]]
                         for j in joints[:15]], dtype=np.float32)
        midpoint = hard.mean(axis=1)
        half_range = (hard[:, 1] - hard[:, 0]) * 0.5 * self.info["soft_joint_pos_limit_factor"]
        self.target_lower, self.target_upper = midpoint - half_range, midpoint + half_range
        self.action_lower = np.maximum((self.target_lower - self.default[:15]) / self.scale, -100)
        self.action_upper = np.minimum((self.target_upper - self.default[:15]) / self.scale, 100)
        self.previous_action = np.zeros(15, dtype=np.float32)
        self.history = np.zeros((10, 84), dtype=np.float32)
        self.observation = np.zeros((1, 840), dtype=np.float32)
        self.initialized = False
        self.backend = default_runtime().open_backend(
            ModelSpec.onnx(onnx, providers=("CPUExecutionProvider",)),
            backend="onnxruntime",
        )
        if self.backend.input_shape("obs") != (1, 840) or self.backend.output_shape("actions") != (1, 15):
            self.backend.close()
            raise ValueError("walk_robust requires ONNX obs[1,840] -> actions[1,15]")
        self.publish_output(self.default, self.kps, self.kds)

    def reset(self, frame=None):
        self.previous_action.fill(0)
        self.history.fill(0)
        self.initialized = False
        self.publish_output(self.default, self.kps, self.kds)

    def current_observation(self, frame):
        joints = self.bind_joints(frame)
        q, dq = joints.position, joints.velocity
        omega = np.asarray(frame.angular_velocity)
        command = np.zeros(3) if frame.command is None else np.asarray(frame.command)
        if (omega.shape != (3,) or command.shape != (3,)
                or not all(np.isfinite(a).all() for a in (q, dq, omega, command))):
            raise ValueError("non-finite or malformed encoder/IMU/command input")
        ang_vel, gravity = pelvis_imu(frame.quat_wxyz, omega, q[:3], dq[:3])
        bounds = np.asarray(self.info["command_bounds"])
        command = np.clip(command, bounds[:, 0], bounds[:, 1])
        return np.concatenate((ang_vel, gravity, command,
                               [self.info["height_command"], self.info["waist_yaw_command"]],
                               q[:15] - self.default[:15], dq[:15], self.previous_action,
                               q[15:] - self.default[15:29], dq[15:])).astype(np.float32)

    def step(self, frame, dt, *, advance=True):
        if advance and not np.isclose(dt, self.info["control_period_s"], atol=1e-6):
            raise ValueError(f"walk_robust needs 50 Hz inference; received dt={dt}")
        current = self.current_observation(frame)
        candidate = self.observation.reshape(10, 84)
        if self.initialized:
            candidate[:-1] = self.history[1:]
            candidate[-1] = current
        else:
            candidate[:] = current
        # Keep unclipped history, exactly as Isaac Lab's CircularBuffer does.
        pending = candidate.copy() if advance else None
        np.clip(self.observation, -100, 100, out=self.observation)
        outputs = self.backend.run({"obs": self.observation})
        bounded = self.decode_into(outputs)
        if advance:
            self.history[:] = pending
            self.previous_action[:] = bounded
            self.initialized = True
        return self.output

    def decode_into(self, outputs):
        raw = np.asarray(outputs["actions"])
        if raw.shape != (1, 15) or not np.isfinite(raw).all():
            raise ValueError("actor returned invalid actions")
        bounded = np.clip(raw[0], self.action_lower, self.action_upper)
        targets = self.default.copy()
        targets[:15] = np.clip(self.default[:15] + self.scale * bounded,
                               self.target_lower, self.target_upper)
        self.publish_output(targets, self.kps, self.kds)
        return bounded

    def close(self):
        self.backend.close()
