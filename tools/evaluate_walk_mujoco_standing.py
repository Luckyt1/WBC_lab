"""Long zero-command and walk-to-stop checks using the actual deployment adapter.

CPU only; requires the framework package on PYTHONPATH plus MuJoCo/ONNX Runtime.
No ROS messages are published. The supplied ONNX must have the same 840/15
observation/action contract as com.bxi.walk_robust.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def drift_metrics(positions, start, dt):
    samples = np.asarray(positions)[start:]
    if len(samples) < 2:
        return {'valid': False}
    offsets = samples - samples[0]
    speed = np.linalg.norm(np.diff(samples, axis=0), axis=1) / dt
    return {'valid': True, 'measurement_seconds': (len(samples) - 1) * dt,
            'net_drift_m': float(np.linalg.norm(offsets[-1])),
            'max_drift_m': float(np.linalg.norm(offsets, axis=1).max()),
            'end_xy_offset_m': offsets[-1].tolist(),
            'mean_horizontal_speed_m_s': float(speed.mean()),
            'p95_horizontal_speed_m_s': float(np.quantile(speed, .95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mod', type=Path, required=True)
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--terrain', choices=('flat', 'source'), default='flat',
                        help='Flat plane for Isaac comparisons; source retains XML obstacles.')
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--max-drift', type=float, default=.15)
    parser.add_argument('--max-speed', type=float, default=.02)
    args = parser.parse_args()
    if args.seconds < 30 or min(args.max_drift, args.max_speed) <= 0:
        parser.error('use at least 30 seconds and positive thresholds')
    spec = importlib.util.spec_from_file_location('deployment_sim', args.mod / 'tools/validate_sim.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sim = module.Simulation(args.xml, onnx=args.onnx)
    from mujoco_evaluation_terrain import configure_terrain, first_obstacle_contact
    terrain = configure_terrain(sim.model, args.terrain)
    from fixed_pelvis_height import apply_onnx_height
    height_command = apply_onnx_height(sim.policy, args.onnx)
    results = []
    count = round(args.seconds / .02)
    try:
        for case in ('zero_command', 'walk_then_stop'):
            sim.reset()
            positions, heights = [], []
            obstacle_contact = None
            for step in range(count):
                command = (.3, 0, 0) if case == 'walk_then_stop' and 100 <= step < 500 else (0, 0, 0)
                sim.advance(command)
                if obstacle_contact is None:
                    obstacle_contact = first_obstacle_contact(sim.model, sim.data)
                positions.append(sim.data.xpos[sim.pelvis, :2].copy())
                heights.append(float(sim.data.xpos[sim.pelvis, 2]))
                if heights[-1] < .5 or not np.isfinite(sim.data.qpos).all():
                    break
            start = 700 if case == 'walk_then_stop' else 200
            metrics = drift_metrics(positions, start, .02)
            completed = len(positions) == count and min(heights) >= .5
            passed = (completed and metrics['valid']
                      and metrics['max_drift_m'] <= args.max_drift
                      and metrics['mean_horizontal_speed_m_s'] <= args.max_speed)
            result = dict(case=case, seconds=len(positions)*.02, measurement_start_s=start*.02,
                          min_height_m=min(heights), completed=completed, passed=passed,
                          first_obstacle_contact=obstacle_contact, **metrics)
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        sim.policy.close()
    report = {'onnx': str(args.onnx.resolve()),
              'onnx_sha256': hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
              'xml_sha256': hashlib.sha256(args.xml.read_bytes()).hexdigest(),
              'physics_dt': .005, 'policy_dt': .02,
              'height_command': height_command,
              'terrain': terrain,
              'thresholds': {'max_drift_m': args.max_drift, 'mean_speed_m_s': args.max_speed},
              'arms': 'fixed_training_default', 'noise': False, 'push': False,
              'results': results, 'passed': all(x['passed'] for x in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
