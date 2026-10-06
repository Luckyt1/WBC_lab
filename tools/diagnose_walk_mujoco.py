"""Record gait, heading and drift of an explicitly selected ONNX (CPU, no ROS).

Use the same deployment adapter and XML as evaluate_walk_mujoco_standing.py.
Noise/push are separate diagnostic interventions, not Isaac acceptance gates.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np

DT = .02
CASES = {
    'zero': ((0., 0., 0.), 60., False, False),
    'walk_stop': ((.3, 0., 0.), 60., False, False),
    'forward': ((.3, 0., 0.), 60., False, False),
    'fast': ((.6, 0., 0.), 20., False, False),
    'backward': ((-.3, 0., 0.), 20., False, False),
    'turn_pos': ((0., 0., .5), 20., False, False),
    'turn_neg': ((0., 0., -.5), 20., False, False),
    'zero_noise': ((0., 0., 0.), 60., True, False),
    'zero_push': ((0., 0., 0.), 60., False, True),
}
DEFAULT_CASES = list(CASES)
CASES.update({
    'arc_forward_pos': ((.3, 0., .1), 20., False, False),
    'arc_forward_neg': ((.3, 0., -.1), 20., False, False),
    'arc_backward_pos': ((-.3, 0., .1), 20., False, False),
    'arc_backward_neg': ((-.3, 0., -.1), 20., False, False),
})


def complete_bouts(mask, dt=DT):
    """True-run durations; omit intervals truncated by either window boundary."""
    edges = np.flatnonzero(np.diff(np.asarray(mask, dtype=int))) + 1
    return [(b-a)*dt for a, b in zip(edges[:-1], edges[1:]) if mask[a]]


def contact_metrics(contact, dt=DT):
    output = []
    for side in range(2):
        c = contact[:, side]
        air = complete_bouts(~c, dt)
        output.append(dict(duty=float(c.mean()),
                           raw_touchdowns_per_s=float(np.sum(c[1:] & ~c[:-1]) / (len(c)*dt)),
                           median_air_s=float(np.median(air)) if air else None,
                           p95_air_s=float(np.quantile(air, .95)) if air else None,
                           air_bouts_at_least_80ms=int(sum(t >= .08-1e-9 for t in air))))
    return dict(left=output[0], right=output[1],
                double_support=float(contact.all(-1).mean()),
                single_support=float((contact.sum(-1) == 1).mean()),
                flight=float((~contact.any(-1)).mean()))


def summarize(a, name):
    start = 700 if name == 'walk_stop' else 200
    positions = a['position'][start:, :2]
    heading = np.unwrap(a['yaw'])[start:]
    velocity = a['velocity_yaw'][start:]
    offset = positions - positions[0]
    contact = a['contact'][start:]
    yaw0 = heading[0]
    initial_frame = offset @ np.array([[np.cos(yaw0), -np.sin(yaw0)],
                                      [np.sin(yaw0), np.cos(yaw0)]])
    r = dict(completed=bool(len(a['time']) == round(CASES[name][1]/DT)
                            and a['position'][:, 2].min() >= .5
                            and np.isfinite(a['position']).all()),
             elapsed_s=float(a['time'][-1]),
             minimum_height_m=float(a['position'][:, 2].min()),
             measurement_start_s=start*DT,
             measurement_duration_s=float(a['time'][-1]-a['time'][start]),
             mean_velocity_yaw=velocity.mean(0).tolist(),
             velocity_mae=np.abs(velocity-a['command'][start:]).mean(0).tolist(),
             requested_velocity_mae=np.abs(velocity-a.get('requested_command', a['command'])[start:]).mean(0).tolist(),
             yaw_change_deg=float(np.rad2deg(heading[-1]-heading[0])),
             max_heading_excursion_deg=float(np.rad2deg(np.abs(heading-heading[0]).max())),
             end_xy_offset_m=offset[-1].tolist(),
             max_planar_offset_m=float(np.linalg.norm(offset, axis=-1).max()),
             max_cross_track_initial_heading_m=float(np.abs(initial_frame[:, 1]).max()),
             mean_planar_speed_m_s=float(np.linalg.norm(np.diff(positions, axis=0), axis=-1).mean()/DT),
             contact=contact_metrics(contact),
             p95_foot_clearance_m=np.quantile(a['foot_clearance'][start:], .95, axis=0).tolist(),
             action_clipped_fraction=a['clipped'][start:].mean(0).tolist())
    for foot, label in enumerate(('left', 'right')):
        mask = contact[:, foot]
        v = a['contact_point_speed'][start:, foot][mask]
        r['contact'][label]['mean_contact_point_speed_m_s'] = float(v.mean()) if len(v) else None
    if name == 'zero_push':
        speed = np.linalg.norm(velocity[:, :2], axis=-1)
        after = a['time'][start:] >= 12.
        r['after_12s_mean_speed_m_s'] = float(speed[after].mean())
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mod', type=Path, required=True)
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES), default=DEFAULT_CASES)
    parser.add_argument('--include-mixed-yaw', action='store_true')
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--noise-component', choices=('all', 'gyro', 'gravity', 'joint_pos', 'joint_vel'), default='all')
    parser.add_argument('--joint-velocity-filter-alpha', type=float, default=1.,
                        help='Simulation-only EMA of noisy actor joint velocities; 1 disables filtering.')
    parser.add_argument('--heading-hold-gain', type=float, default=0.,
                        help='Simulation-only heading feedback during zero-turn translation, using simulator yaw.')
    args = parser.parse_args()
    if args.include_mixed_yaw:
        args.cases = list(dict.fromkeys(args.cases+[name for name in CASES if name.startswith('arc_')]))
    if not 0 < args.joint_velocity_filter_alpha <= 1 or not 0 <= args.heading_hold_gain <= 2:
        parser.error('Filter alpha must be in (0,1]; heading gain in [0,2]')
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('deployment_sim', args.mod/'tools/validate_sim.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sim = module.Simulation(args.xml, onnx=args.onnx)
    from fixed_pelvis_height import apply_onnx_height
    height_command = apply_onnx_height(sim.policy, args.onnx)
    feet = np.array([sim.model.body(s+'_ankle_x_link').id for s in ('l', 'r')])
    foot_geoms = [np.flatnonzero((sim.model.geom_bodyid == b) & (sim.model.geom_contype > 0)) for b in feet]
    # The supplied XML uses capsules for all sole collision geometry.
    assert all(sim.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for gs in foot_geoms for g in gs)
    original_observation = sim.policy.current_observation
    original_decode = sim.policy.decode_into
    scales = np.zeros(84, dtype=np.float32)
    scales[:3], scales[3:6] = .2, .05
    scales[11:26], scales[26:41], scales[56:70], scales[70:84] = .01, 1.5, .01, 1.5
    component_mask = np.zeros(84, dtype=bool)
    slices = {'all': [slice(0, 84)], 'gyro': [slice(0, 3)], 'gravity': [slice(3, 6)],
              'joint_pos': [slice(11, 26), slice(56, 70)],
              'joint_vel': [slice(26, 41), slice(70, 84)]}
    for selected in slices[args.noise_component]:
        component_mask[selected] = True
    scales *= component_mask
    noise_on = False
    rng = np.random.default_rng(args.seed)
    parity_max = np.zeros(2)
    clipped = np.zeros(15, dtype=bool)
    velocity_columns = np.r_[26:41, 70:84]
    filtered_velocity = None

    def observation(frame):
        nonlocal filtered_velocity
        current = original_observation(frame)
        rotation = sim.data.xmat[sim.pelvis].reshape(3, 3)
        vel = np.empty(6)
        mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_XBODY, sim.pelvis, vel, 1)
        parity_max[:] = np.maximum(parity_max, [np.max(np.abs(current[:3]-vel[:3])),
                                                np.max(np.abs(current[3:6]-rotation.T @ [0, 0, -1]))])
        if noise_on:
            current = current + rng.uniform(-1, 1, 84).astype(np.float32)*scales
        if args.joint_velocity_filter_alpha < 1:
            v = current[velocity_columns]
            filtered_velocity = (v.copy() if filtered_velocity is None else
                                 filtered_velocity + args.joint_velocity_filter_alpha*(v-filtered_velocity))
            current[velocity_columns] = filtered_velocity
        return current

    def decode(outputs):
        raw = outputs['actions'][0]
        clipped[:] = (raw < sim.policy.action_lower) | (raw > sim.policy.action_upper)
        return original_decode(outputs)

    sim.policy.current_observation = observation
    sim.policy.decode_into = decode
    report = dict(onnx=str(args.onnx.resolve()), onnx_sha256=hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
                  xml=str(args.xml.resolve()), xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),
                  policy_py_sha256=hashlib.sha256((args.mod/'policy.py').read_bytes()).hexdigest(),
                  seed=args.seed, physics_dt=.005, policy_dt=DT, arms='fixed_training_default',
                  height_command=height_command,
                  noise='uniform independent, matching current Isaac actor noise scales',
                  noise_component=args.noise_component,
                  joint_velocity_filter_alpha=args.joint_velocity_filter_alpha,
                  heading_hold_gain=args.heading_hold_gain,
                  heading_feedback='simulator pelvis yaw; idealized experiment, requires a real heading estimate for deployment',
                  push='zero_push only: overwrite root vx=.8, vy=.8 m/s at t=8s; no mass/friction randomization',
                  contact='total positive normal foot-ground force >1N at policy sample; raw events may include chatter',
                  contact_point_speed='max horizontal rigid-body velocity at active ground contact points',
                  clearance='lowest point of sole collision capsules relative to z=0', results={})
    try:
        for name in args.cases:
            command, seconds, noise_on, push = CASES[name]
            rng = np.random.default_rng(args.seed)
            sim.reset()
            filtered_velocity = None
            heading_reference = None
            hist = {k: [] for k in ('time', 'position', 'yaw', 'velocity_yaw', 'command', 'contact',
                                  'requested_command', 'normal_force', 'foot_clearance', 'contact_point_speed', 'joint_pos', 'clipped')}
            for step in range(round(seconds/DT)):
                requested = command if step >= 100 and (name != 'walk_stop' or step < 500) else (0., 0., 0.)
                cmd = np.array(requested, dtype=float)
                if args.heading_hold_gain > 0 and cmd[2] == 0 and np.linalg.norm(cmd[:2]) > 0:
                    rotation = sim.data.xmat[sim.pelvis].reshape(3, 3)
                    current_yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                    if heading_reference is None:
                        heading_reference = current_yaw
                    error = np.arctan2(np.sin(heading_reference-current_yaw), np.cos(heading_reference-current_yaw))
                    bounds = sim.policy.info['command_bounds'][2]
                    cmd[2] = np.clip(args.heading_hold_gain*error, *bounds)
                else:
                    heading_reference = None
                if push and step == 400:
                    sim.data.qvel[:2] = .8
                    mujoco.mj_forward(sim.model, sim.data)
                sim.advance(cmd)
                rotation = sim.data.xmat[sim.pelvis].reshape(3, 3)
                yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                vel = np.empty(6)
                mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_XBODY, sim.pelvis, vel, 0)
                velocity = np.array([np.cos(yaw)*vel[3]+np.sin(yaw)*vel[4],
                                     -np.sin(yaw)*vel[3]+np.cos(yaw)*vel[4], vel[2]])
                force, point_speed = np.zeros(2), np.zeros(2)
                for cidx in range(sim.data.ncon):
                    c = sim.data.contact[cidx]
                    bodies = sim.model.geom_bodyid[[c.geom1, c.geom2]]
                    if 0 not in bodies:
                        continue
                    for fi, b in enumerate(feet):
                        if b not in bodies:
                            continue
                        cf = np.empty(6)
                        mujoco.mj_contactForce(sim.model, sim.data, cidx, cf)
                        force[fi] += max(0., cf[0])
                        if cf[0] > .1:
                            bv = np.empty(6)
                            mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_XBODY, b, bv, 0)
                            pv = bv[3:] + np.cross(bv[:3], c.pos-sim.data.xpos[b])
                            point_speed[fi] = max(point_speed[fi], np.linalg.norm(pv[:2]))
                clearance = [min(sim.data.geom_xpos[g, 2]-sim.model.geom_size[g, 0]
                                 -sim.model.geom_size[g, 1]*abs(sim.data.geom_xmat[g].reshape(3, 3)[2, 2])
                                 for g in gs) for gs in foot_geoms]
                values = dict(time=sim.data.time, position=sim.data.xpos[sim.pelvis].copy(), yaw=yaw,
                              velocity_yaw=velocity, command=cmd, requested_command=requested,
                              contact=force > 1., normal_force=force,
                              foot_clearance=clearance, contact_point_speed=point_speed,
                              joint_pos=sim.data.qpos[sim.qids].copy(), clipped=clipped.copy())
                for k, value in values.items():
                    hist[k].append(value)
                if sim.data.xpos[sim.pelvis, 2] < .5 or not np.isfinite(sim.data.qpos).all():
                    break
            arrays = {k: np.asarray(v) for k, v in hist.items()}
            np.savez_compressed(args.output/(name+'.npz'), **arrays)
            if len(arrays['time']) <= (700 if name == 'walk_stop' else 200):
                result = dict(completed=False, failure_before_measurement=True)
            else:
                result = summarize(arrays, name)
            report['results'][name] = result
            report['imu_parity_max_abs_error'] = dict(angular_velocity=float(parity_max[0]), gravity=float(parity_max[1]))
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(dict(case=name, **result)), flush=True)
    finally:
        sim.policy.close()


if __name__ == '__main__':
    main()
