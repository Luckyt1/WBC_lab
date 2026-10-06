"""Offline MuJoCo smoke test; no ROS nodes or hardware communication.

Run with the framework Python package installed/on PYTHONPATH.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from bxi_example_py_elf3.framework.inference import InferenceFrame
from bxi_example_py_elf3.framework.joints import JointStateView

MOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOD.parents[1]))
from tools.fixed_pelvis_height import apply_onnx_height
spec = importlib.util.spec_from_file_location("walk_robust_policy", MOD / "policy.py")
policy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_module)


class Simulation:
    def __init__(self, xml, *, onnx=None):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)
        self.policy = policy_module.WalkRobustPolicy(onnx=onnx)
        self.layout = self.policy.joint_contract.action
        joints = [self.model.joint(name) for name in self.layout.names]
        self.qids = np.array([int(j.qposadr[0]) for j in joints])
        self.dqids = np.array([int(j.dofadr[0]) for j in joints])
        self.actids = np.array([self.model.actuator(name).id for name in self.layout.names])
        self.pelvis = self.model.body("waist_z_link").id

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = self.policy.default
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.policy.reset()

    def frame(self, command):
        return InferenceFrame(
            joints=JointStateView(self.layout, self.data.qpos[self.qids].copy(),
                                  self.data.qvel[self.dqids].copy()),
            quat_wxyz=self.data.sensor("torso_quat").data.copy(),
            angular_velocity=self.data.sensor("torso_ang_vel").data.copy(),
            command=np.array(command, dtype=np.float32),
        )

    def advance(self, command):
        output = self.policy.step(self.frame(command), 0.02).joints
        for _ in range(4):
            torque = output.kp * (output.position - self.data.qpos[self.qids]) - output.kd * self.data.qvel[self.dqids]
            limits = self.model.actuator_ctrlrange[self.actids]
            self.data.ctrl[self.actids] = np.clip(torque, limits[:, 0], limits[:, 1])
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def run(self, command, seconds, viewer=None, push=False):
        self.reset()
        height, velocity = [], []
        object_velocity = np.empty(6)
        for step in range(round(seconds / 0.02)):
            start = time.monotonic()
            self.data.xfrc_applied[:] = 0
            # 80 N lateral force for 0.2 s at t=6 s (not a complete robustness evaluation).
            if push and 300 <= step < 310:
                self.data.xfrc_applied[self.pelvis, 1] = 80
            self.advance(command if step >= 100 else (0, 0, 0))
            height.append(float(self.data.xpos[self.pelvis, 2]))
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                     self.pelvis, object_velocity, 1)
            if step >= 200:
                velocity.append(object_velocity[[3, 4, 2]].copy())
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0, 0.02 - (time.monotonic() - start)))
            if height[-1] < 0.5 or not np.isfinite(self.data.qpos).all():
                break
        return {"command": list(command), "push": push, "elapsed_s": self.data.time,
                "completed": bool(len(height) == round(seconds / 0.02) and min(height) >= 0.5),
                "minimum_pelvis_height": min(height),
                "mean_velocity_after_4s": np.mean(velocity, axis=0).tolist() if velocity else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--viewer", action="store_true", help="show forward walking in MuJoCo")
    args = parser.parse_args()
    sim = Simulation(args.xml, onnx=args.onnx)
    try:
        height = apply_onnx_height(sim.policy, args.onnx)
        if args.viewer:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = sim.pelvis
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -15
                results = [sim.run((0.3, 0, 0), args.seconds, viewer=viewer)]
        else:
            cases = [(0, 0, 0), (0.3, 0, 0), (-0.3, 0, 0),
                     (0, 0.2, 0), (0, -0.2, 0), (0, 0, 0.5), (0, 0, -0.5)]
            results = [sim.run(cmd, args.seconds) for cmd in cases]
            results.append(sim.run((0.3, 0, 0), args.seconds, push=True))
        report = {"onnx": str(args.onnx), "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
                  "height_command": height, "physics_step": 0.005,
                  "inference_hz": 50, "results": results}
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        return 0 if all(result["completed"] for result in results) else 1
    finally:
        sim.policy.close()


if __name__ == "__main__":
    raise SystemExit(main())
