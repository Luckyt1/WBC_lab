"""Create GL-165-specific reach/hold/return data, compatible with IkArmMotion."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from legged_lab.assets.gl165.model import ARM_JOINT_NAMES, DEFAULT_POSITIONS, asset_dir, ik_bank_dir
from tools.gl165_kinematics import ArmChain
from tools.gl165_sources import sha256


def build_bank(asset, output, train_count=256, test_count=64, seed=42):
    asset, output = Path(asset), Path(output)
    if train_count < 1 or test_count < 1:
        raise ValueError("Both train and test splits must be nonempty")
    model = json.loads((asset / "model.json").read_text())
    urdf = asset / "source/urdf/gl165.urdf"
    if sha256(urdf) != model['training_urdf_sha256']:
        raise ValueError("Prepared URDF changed after conversion")
    chains = [ArmChain(urdf, side) for side in ("l", "r")]
    home = np.array([DEFAULT_POSITIONS[n] for n in ARM_JOINT_NAMES])
    # Quintic time scaling gives zero endpoint velocity and acceleration.
    t = np.linspace(0, 1, 129)
    blend = t**3 * (10 - 15 * t + 6 * t**2)
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(robot="gl165", frame="body_y_link", end_effectors=["lwrist_y_link", "rwrist_y_link"],
                    source_urdf_sha256=model['source_urdf_sha256'], training_urdf_sha256=sha256(urdf),
                    arm_joint_names=ARM_JOINT_NAMES, neutral_positions=home.tolist(), seed=seed,
                    bank_sha256={}, targets=[],
                    recipe=dict(name="gl165_reach_hold_return_v1", arrival_gate=False,
                                neutral_fraction=.15, hard_fraction_of_active=0.0,
                                prepare_s=[.5, 1.0], reach_s=[1.5, 2.5], hold_s=[1.0, 2.0],
                                return_s=[1.5, 2.5], rest_s=[.5, 1.0],
                                max_reference_velocity_rad_s=2.0,
                                max_reference_acceleration_rad_s2=8.0),
                    note="URDF pose IK endpoints with smooth joint-space paths; torso-relative, no live waist compensation. "
                         "Nominal workspace screening plus PhysX self-collisions; not the ELF3 v3 OBB/Ruckig controller.")
    for split, count in (("train", train_count), ("test", test_count)):
        paths, durations, ids, groups = [], [], [], []
        for i in range(count):
            samples = [chain.sample(rng) for chain in chains]
            goal = np.concatenate([sample[0] for sample in samples])
            group = ("both", "left", "right")[i % 3]
            if group == "left":
                goal[7:] = home[7:]
            elif group == "right":
                goal[:7] = home[:7]
            forward = home + blend[:, None] * (goal - home)
            paths.append(np.stack((forward, forward[::-1])))
            # Max first/second derivative of the quintic profile: 1.875, 5.774.
            delta = np.max(np.abs(goal - home))
            duration = max(1.5, 1.875 * delta / 2.0, np.sqrt(5.774 * delta / 8.0))
            durations.append([duration, duration])
            identity = f"gl165_{split}_{i:05d}"
            ids.append(identity)
            groups.append(group)
            poses = [chain.fk(q) for chain, q in zip(chains, goal.reshape(2, 7))]
            manifest['targets'].append(dict(
                id=identity, split=split, group=group,
                positions=[p.tolist() for p, _, _ in poses],
                orientations_wxyz=[Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]].tolist() for _, r, _ in poses],
                ik_position_residual_m=max(sample[3] for sample in samples),
                ik_orientation_residual_rad=max(sample[4] for sample in samples)))
        path = output / f"{split}.npz"
        np.savez_compressed(path, ids=np.array(ids), groups=np.array(groups),
                            paths=np.array(paths, dtype=np.float32), hard=np.zeros(count, dtype=bool),
                            min_durations=np.array(durations, dtype=np.float32))
        manifest['bank_sha256'][split] = sha256(path)
        print(f"GL165_IK_SPLIT_COMPLETE {split} {count}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=asset_dir())
    parser.add_argument("--output", type=Path, default=ik_bank_dir())
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_bank(args.asset, args.output, args.train_count, args.test_count, args.seed)
