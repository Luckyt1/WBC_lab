"""Preview GL165's training arm IK disturbance in MuJoCo, with torso/legs fixed."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from legged_lab.assets.gl165.model import asset_dir, ik_bank_dir
from tools.gl165_mujoco import Gl165ArmPreview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=asset_dir())
    parser.add_argument('--bank', type=Path, default=ik_bank_dir())
    parser.add_argument('--arm-group', choices=('both', 'left', 'right', 'all'), default='all')
    parser.add_argument('--target-region', choices=('all', 'low', 'overhead', 'lateral'), default='all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=0., help='Duration; 0 runs until the window closes')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.headless and args.seconds <= 0:
        parser.error('--headless requires --seconds > 0')
    sim = Gl165ArmPreview(args.asset, args.bank, args.arm_group, args.seed, args.target_region)
    print('GL165_MUJOCO_READY', json.dumps(dict(mode='arm IK preview; torso and legs fixed',
          arm_group=args.arm_group, bank=str(args.bank), xml=str(sim.xml_path))), flush=True)
    samples, endpoints, phases = [], [], set()
    state = dict(paused=False, reset=False)

    def keypress(key):
        if key == 32:
            state['paused'] = not state['paused']
        elif key == ord('R'):
            state['reset'] = True

    def run(viewer=None):
        step = 0
        while (viewer is None or viewer.is_running()) and (args.seconds <= 0 or step * sim.dt < args.seconds):
            started = time.monotonic()
            if state['reset']:
                sim.reset()
                state['reset'] = False
            if not state['paused']:
                sim.step()
                phases.add(int(sim.arms.phase[0]))
                if int(sim.arms.phase[0]) == 2 and sim.continuous:
                    endpoints.append(sim.metrics())
                if step % 50 == 0:
                    sample = sim.metrics()
                    samples.append(sample)
                    if step % 250 == 0:
                        print('GL165_MUJOCO_STATE', json.dumps(sample), flush=True)
                step += 1
            if viewer is not None:
                metrics = sim.metrics()
                showing_goal = sim.continuous or metrics['phase'] in ('reach', 'hold')
                with viewer.lock():
                    viewer.user_scn.ngeom = 2 if showing_goal else 0
                    for i, position in enumerate(sim.target_positions() if showing_goal else []):
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([.018, .018, .018]), position, np.eye(3).ravel(),
                            np.array([.2, .7, 1., .8]) if i == 0 else np.array([1., .6, .15, .8]))
                left_error, right_error = (100 * e for e in metrics['wrist_position_error_m'])
                if sim.continuous:
                    error_text = f'Next target: {metrics["target_region"]} | distance L/R: {left_error:.1f}/{right_error:.1f} cm\n'
                    if endpoints:
                        error_text += 'Last endpoint error L/R: ' + '/'.join(
                            f'{100 * e:.1f}' for e in endpoints[-1]['wrist_position_error_m']) + ' cm\n'
                else:
                    error_text = (f'Hold error L / R: {left_error:.1f} / {right_error:.1f} cm\n'
                                  if metrics['phase'] == 'hold' else 'Arrival error is measured during hold\n')
                viewer.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    'GL165 | TRAINING ARM IK',
                    f"Phase: {metrics['phase']} | cycles: {metrics['completed_cycles']}\n"
                    f"{args.arm_group} arms | torso / waist / legs fixed\n"
                    'Blue / orange dots: next wrist goals\n' + error_text +
                    'Space: pause | R: new target | mouse: orbit / zoom')])
                viewer.sync()
                time.sleep(max(0., sim.dt - (time.monotonic() - started)))
        return step

    if args.headless:
        steps = run()
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keypress,
                                         show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 45, -15, 3.3
            viewer.cam.lookat[:] = (0., 0., 1.05)
            viewer.opt.geomgroup[3] = 0
            steps = run(viewer)
    report = dict(mode='fixed torso/legs, dynamic arms, training IK replay', seconds=steps * sim.dt,
                  arm_group=args.arm_group, target_region=args.target_region, phases_seen=sorted(phases),
                  endpoints=endpoints, samples=samples)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('GL165_MUJOCO_COMPLETE', json.dumps({k: v for k, v in report.items() if k not in ('samples', 'endpoints')}), flush=True)


if __name__ == '__main__':
    main()
