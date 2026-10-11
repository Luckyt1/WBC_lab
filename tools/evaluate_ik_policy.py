"""Frozen online-IK MuJoCo validation, using exactly the training arm supervisor.

No training updates, model deployment, ROS or hardware connection. All unseen
targets are kept in a separate split. Prior failures are reported separately as
regression cases, never relabelled as generalization evidence.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

# Install dependencies in the dedicated IK environment; do not inject packages
# from a different Python environment.
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from elf3_ik_core import new_solver, SOURCE
from fixed_pelvis_height import apply_onnx_height
from mujoco_evaluation_terrain import configure_terrain

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legged_lab.paths import deployment_framework, deployment_mod, mujoco_xml
package = types.ModuleType('elf3_cpu')
package.__path__ = [str(ROOT/'legged_lab/envs/elf3_tang')]
sys.modules[package.__name__] = package
from elf3_cpu.wbc.arm_guard import ArmCollisionGuard
from elf3_cpu.wbc.arm_gravity import ArmGravity
from elf3_cpu.wbc.arm_pose_tracker import ArmPoseTracker


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def select_targets(manifest, full=False):
    pool=manifest['targets']+manifest.get('rejected_targets',[])
    test = [t for t in pool if t['split']=='test']
    regression = [t for t in pool if t['split']=='train' and t['hard']]
    if full: return test+regression
    # Fixed selection made without consulting checkpoint outcomes.
    chosen = []
    for group in sorted({t['group'] for t in test}):
        chosen += sorted([t for t in test if t['group']==group],key=lambda t:t['id'])[:4]
    known = ('004_both_overhead','020_left_workspace_10','027_right_workspace_05',
             '034_mirrored_boundary_00','039_mirrored_boundary_07','046_asymmetric_010',
             '055_asymmetric_056','062_asymmetric_088')
    priority = [t for t in regression if any(t['id'].endswith(k) for k in known)]
    remainder = [t for t in sorted(regression,key=lambda t:t['id']) if t not in priority]
    return chosen+(priority+remainder)[:12]


class IkPolicyTrial:
    def __init__(self, target, duration, onnx, mod, xml, bank):
        torch.set_num_threads(1)
        sys.path.insert(0,str(deployment_framework()))
        adapter = load_module('ik_validation_adapter',Path(mod)/'tools/validate_sim.py')
        self.sim = sim = adapter.Simulation(xml, onnx=onnx)
        self.terrain = configure_terrain(sim.model,'flat')
        self.height = apply_onnx_height(sim.policy,onnx)
        sim.reset()
        self.module,self.solver,self.limiter,_ = new_solver()
        if tuple(sim.layout.names[15:29])!=self.module.ARM_JOINTS: raise ValueError('Arm layout mismatch')
        manifest = json.loads((Path(bank)/'manifest.json').read_text())
        limits = torch.tensor(manifest['joint_limits']).unsqueeze(0)
        names = [sim.model.body(i).name for i in range(sim.model.nbody)]
        recipe = manifest['recipe']
        self.guard = ArmCollisionGuard(self.module.ARM_JOINTS,names,limits,'cpu',margin=recipe['guard_margin'],max_velocity=recipe['max_velocity'],
                                      geometry_mode=recipe.get('guard_geometry','capsule_v1'))
        self.pose_tracker = ArmPoseTracker(manifest,'cpu') if recipe.get('waist_pose_correction') else None
        self.cached_path = None
        self.requested_duration=float(duration)
        self.return_duration=3.
        self.v3=manifest['version']>=3
        if self.pose_tracker is not None:
            data=np.load(Path(bank)/(target['split']+'.npz'),allow_pickle=False)
            matches=np.flatnonzero(data['ids']==target['id'])
            if len(matches):
                self.cached_path=data['paths'][matches[0]].copy()
                if 'min_durations' in data:
                    duration=max(float(duration),float(data['min_durations'][matches[0],0]))
                    self.return_duration=max(3.,float(data['min_durations'][matches[0],1]))
        self.executor=None
        if self.v3:
            from elf3_cpu.wbc.predictive_arm import PredictiveArmExecutor
            self.executor=PredictiveArmExecutor(self.guard,torch.tensor(sim.policy.default[None,15:29],dtype=torch.float32),recipe,names)
        self.reference_limits_max=np.zeros(3)
        self.arrival_gated=recipe.get('arrival_gate',False) and self.cached_path is not None
        self.arrival_time=None;self.arrival_dwell=0.;self.reach_timed_out=False
        self.arrival_timeout=recipe.get('arrival_timeout_s',3.)
        self.arrival_dwell_required=recipe.get('arrival_dwell_s',.1)
        self.arrival_speed_limit=recipe.get('arrival_joint_velocity_rad_s',float('inf'))
        self.gravity = ArmGravity('cpu')
        self.torso = names.index('torso_link')
        self.wrists = [names.index(s+'_wrist_z_link') for s in ('l','r')]
        self.arm_bodies = {i for i,n in enumerate(names) if any(p in n for p in ('shoulder','elbow','wrist'))}
        self.target,self.duration = target,float(duration)
        self.goal_p,self.goal_q = np.array(target['positions']),np.array(target['orientations_wxyz'])
        self.initial_p,self.initial_q = self.wrist_pose()
        self.slerps = None
        self.last_reference = sim.policy.default[15:29].copy()
        self.return_start = 2.+self.duration+6.
        self.hold_start = 2.+self.duration
        if self.arrival_gated:
            self.return_start+=self.arrival_timeout
        self.total = self.return_start+self.return_duration+2.
        self.step_index = 0
        self.first_xy = sim.data.xpos[sim.pelvis,:2].copy()
        self.previous_xy = self.first_xy.copy()
        self.hold_errors,self.hold_speeds = [],[]
        self.reasons,self.contacts = Counter(),Counter()
        self.max_drift = self.max_tilt = self.max_step = 0.
        self.min_height = float(sim.data.xpos[sim.pelvis,2])
        self.guard_no_safe = self.guard_corrected = 0.
        self.fall = None
        self.trace = []

    def wrist_pose(self):
        sim = self.sim
        R = sim.data.xmat[sim.pelvis].reshape(3,3)
        p = np.array([R.T@(sim.data.xpos[i]-sim.data.xpos[sim.pelvis]) for i in self.wrists])
        q = np.array([Rotation.from_matrix(R.T@sim.data.xmat[i].reshape(3,3)).as_quat()[[3,0,1,2]] for i in self.wrists])
        return p,q

    @torch.inference_mode()
    def step(self):
        sim = self.sim
        t = self.step_index*.02
        if t>=2 and self.slerps is None:
            self.initial_p,self.initial_q = self.wrist_pose()
            self.slerps = [Slerp([0,1],Rotation.from_quat(np.array([self.initial_q[s],self.goal_q[s]])[:,[1,2,3,0]])) for s in (0,1)]
            self.first_xy = sim.data.xpos[sim.pelvis,:2].copy()
        if t<2: self.phase='prepare';alpha=0.
        elif t<2+self.duration: self.phase='reach';alpha=(t-2)/self.duration
        elif t<self.return_start: self.phase='hold';alpha=1.
        elif t<self.return_start+self.return_duration: self.phase='return';alpha=1-(t-self.return_start)/self.return_duration
        else: self.phase='rest';alpha=0.
        if self.arrival_gated and self.arrival_time is None and t>=2+self.duration and not self.reach_timed_out:
            if t < 2+self.duration+self.arrival_timeout:
                self.phase='reach';alpha=1.
            else:
                self.reach_timed_out=True;self.return_start=t;self.total=t+self.return_duration+2.
                self.phase='return';alpha=1.
        alpha=alpha*alpha*(3-2*alpha)
        p = self.initial_p+alpha*(self.goal_p-self.initial_p)
        q = self.initial_q if self.slerps is None else np.array([f(alpha).as_quat()[[3,0,1,2]] for f in self.slerps])
        desired = sim.policy.default[15:29].copy()
        if t>=2:
            if self.cached_path is not None:
                leg=1 if self.phase in ('return','rest') else 0
                fraction=(t-self.return_start)/self.return_duration if leg else (t-2)/self.duration
                frame=np.clip(fraction,0,1)*(self.cached_path.shape[1]-1)
                lo=int(frame);hi=min(lo+1,self.cached_path.shape[1]-1)
                guide=torch.tensor(((1-frame+lo)*self.cached_path[leg,lo]+(frame-lo)*self.cached_path[leg,hi])[None],dtype=torch.float32)
                def tensor(x): return torch.tensor(x[None],dtype=torch.float32)
                desired=self.pose_tracker.correct(guide,tensor(sim.data.xpos[sim.pelvis]),tensor(sim.data.xquat[sim.pelvis]),
                    tensor(sim.data.xpos[self.torso]),tensor(sim.data.xquat[self.torso])).numpy()[0]
                if self.phase=='hold' or (self.phase=='reach' and t>=2+self.duration):
                    p,q=self.goal_p,self.goal_q
                else:
                    pp,qq=self.pose_tracker.guide_pose(guide);p,q=pp.numpy()[0],qq.numpy()[0]
                self.reasons['validated_guide_with_waist_correction']+=1
            elif not self.v3:
                self.solver.update_measured(tuple(sim.layout.names),sim.data.qpos[sim.qids])
                result = self.solver.solve(p,q)
                self.reasons[result.reason]+=1
                desired = self.last_reference.copy()
                if result.positions is not None:
                    desired = self.limiter._rate_limited_arm_reference(result.positions)
            else:
                self.reasons['planning_failed_hold_neutral']+=1
        actual = torch.tensor(sim.data.qpos[sim.qids[15:29]],dtype=torch.float32).unsqueeze(0)
        body_pos=torch.tensor(sim.data.xpos,dtype=torch.float32).unsqueeze(0)
        body_quat=torch.tensor(sim.data.xquat,dtype=torch.float32).unsqueeze(0)
        desired_tensor=torch.tensor(desired,dtype=torch.float32).unsqueeze(0)
        if self.executor is not None:
            arm_target=self.executor.project(desired_tensor,actual,body_pos,body_quat,self.torso).numpy()[0]
            keys=('IK/reference_velocity_max_rad_s','IK/reference_acceleration_max_rad_s2','IK/reference_jerk_max_rad_s3')
            self.reference_limits_max=np.maximum(self.reference_limits_max,[float(self.executor.metrics[k]) for k in keys])
        else:
            arm_target = self.guard.project(desired_tensor,
                torch.tensor(self.last_reference,dtype=torch.float32).unsqueeze(0),actual,
                body_pos,body_quat,self.torso).numpy()[0]
        self.max_step = max(self.max_step,float(np.abs(arm_target-self.last_reference).max()))
        self.last_reference = arm_target.copy()
        self.limiter._last_published_arm_positions = arm_target.copy()
        self.guard_no_safe += float(self.guard.metrics['ArmGuard/no_safe_candidate_fraction'])
        self.guard_corrected += float(self.guard.metrics['ArmGuard/corrected_arm_fraction'])
        gravity = sim.data.xmat[self.torso].reshape(3,3).T@np.array([0.,0.,-9.81])
        ff = self.gravity.compute(actual,torch.tensor(gravity,dtype=torch.float32).unsqueeze(0)).numpy()[0]
        output = sim.policy.step(sim.frame((0.,0.,0.)),.02).joints
        desired_all = output.position.copy();desired_all[15:29]=arm_target
        limits = sim.model.actuator_ctrlrange[sim.actids]
        for _ in range(4):
            torque = output.kp*(desired_all-sim.data.qpos[sim.qids])-output.kd*sim.data.qvel[sim.dqids]
            torque[15:29] += ff
            sim.data.ctrl[sim.actids] = np.clip(torque,limits[:,0],limits[:,1])
            mujoco.mj_step(sim.model,sim.data)
        mujoco.mj_forward(sim.model,sim.data)
        actual_p,actual_q = self.wrist_pose()
        pos_error = float(np.linalg.norm(actual_p-p,axis=1).max())
        rot_error = float((Rotation.from_quat(actual_q[:,[1,2,3,0]]).inv()*Rotation.from_quat(q[:,[1,2,3,0]])).magnitude().max())
        xy = sim.data.xpos[sim.pelvis,:2].copy()
        height = float(sim.data.xpos[sim.pelvis,2])
        up = float(sim.data.xmat[sim.pelvis].reshape(3,3)[2,2])
        drift = float(np.linalg.norm(xy-self.first_xy))
        if t>=2:
            self.max_drift=max(self.max_drift,drift)
            for c in sim.data.contact:
                a,b=(int(sim.model.geom_bodyid[i]) for i in (c.geom1,c.geom2))
                if a and b and a!=b and c.dist<-.001 and (a in self.arm_bodies or b in self.arm_bodies):
                    self.contacts[' / '.join(sorted((sim.model.body(a).name,sim.model.body(b).name)))]+=1
        if self.arrival_gated and self.arrival_time is None and not self.reach_timed_out and t>=2+self.duration:
            settled=np.abs(sim.data.qvel[sim.dqids[15:29]]).max()<=self.arrival_speed_limit
            self.arrival_dwell=self.arrival_dwell+.02 if pos_error<=.02 and rot_error<=.1 and settled else 0.
            if self.arrival_dwell>=self.arrival_dwell_required:
                self.arrival_time=t+.02;self.hold_start=self.arrival_time
                self.return_start=self.hold_start+6.;self.total=self.return_start+self.return_duration+2.
        if self.phase=='hold' and t>=self.hold_start+1.:
            self.hold_errors.append([pos_error,rot_error])
            self.hold_speeds.append(float(np.linalg.norm(xy-self.previous_xy)/.02))
        self.previous_xy=xy
        self.min_height=min(self.min_height,height)
        self.max_tilt=max(self.max_tilt,float(np.rad2deg(np.arccos(np.clip(up,-1,1)))))
        if self.step_index%5==0:
            self.trace.append(np.r_[sim.data.time,sim.data.qpos.copy(),p.ravel(),pos_error,rot_error,drift])
        self.step_index+=1
        if height<.5 or up<.5 or not np.isfinite(sim.data.qpos).all():
            self.fall=dict(time_s=float(sim.data.time),phase=self.phase,height_m=height,up_z=up)
        return self.fall is not None or self.step_index>=round(self.total/.02)

    def report(self):
        error = np.percentile(self.hold_errors,95,axis=0).tolist() if self.hold_errors else None
        speed = float(np.mean(self.hold_speeds)) if self.hold_speeds else None
        reached = error is not None and error[0]<=.02 and error[1]<=.1 and not (self.v3 and self.cached_path is None)
        stable = self.fall is None and self.max_drift<=.15 and speed is not None and speed<=.02
        return dict(target_id=self.target['id'],split=self.target['split'],group=self.target['group'],
            duration_s=self.requested_duration,effective_reach_duration_s=self.duration,
            effective_return_duration_s=self.return_duration,completed=self.fall is None,fall=self.fall,
            elapsed_s=float(self.sim.data.time),maximum_drift_m=self.max_drift,minimum_height_m=self.min_height,
            maximum_tilt_deg=self.max_tilt,hold_wrist_error_p95=error,hold_speed_mean_m_s=speed,
            reached=bool(reached),stable=bool(stable),passed=bool(reached and stable and not self.contacts),
            self_contacts=dict(self.contacts),maximum_reference_step_rad=self.max_step,
            guard_no_safe_fraction=self.guard_no_safe/self.step_index,
            guard_corrected_fraction=self.guard_corrected/self.step_index,ik_reasons=dict(self.reasons),
            validated_path_available=self.cached_path is not None,arrival_time_s=self.arrival_time,
            reach_timed_out=self.reach_timed_out,guard_geometry=self.guard.geometry_mode,
            planning_failed=self.v3 and self.cached_path is None,
            maximum_reference_velocity_acceleration_jerk=self.reference_limits_max.tolist())

    def close(self):
        self.sim.policy.close()


def run_case(arguments):
    target,duration,onnx,mod,xml,bank,out = arguments
    trial = IkPolicyTrial(target,duration,onnx,mod,xml,bank)
    try:
        while not trial.step(): pass
        report = trial.report()
        path = Path(out)/f'{target["id"]}_reach{duration:g}'
        Path(str(path)+'.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(str(path)+'.npz',trace=np.asarray(trial.trace),qpos_width=trial.sim.model.nq)
        return report
    finally:
        trial.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--onnx',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mod',type=Path,default=deployment_mod())
    parser.add_argument('--xml',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--targets',nargs='+',help='Explicit diagnostic target IDs; does not change frozen default pool')
    parser.add_argument('--durations',nargs='+',type=float,help='Requested reach times for explicit diagnostics')
    args = parser.parse_args()
    if args.xml is None:
        args.xml = mujoco_xml()
    manifest = json.loads((args.bank/'manifest.json').read_text())
    selected = select_targets(manifest,args.full)
    if args.targets:
        pool={t['id']:t for t in manifest['targets']+manifest.get('rejected_targets',[])}
        selected=[pool[i] for i in args.targets]
    if args.smoke: selected=selected[:1]
    durations = args.durations or ([.5] if args.smoke else [.5,3.,6.])
    args.output.mkdir(parents=True,exist_ok=True)
    cases = args.output/'cases';cases.mkdir(exist_ok=True)
    arguments = [(t,d,args.onnx,args.mod,args.xml,args.bank,cases) for t in selected for d in durations]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_case,a) for a in arguments]):
            r = future.result();results.append(r)
            print(json.dumps(dict(completed=len(results),total=len(arguments),target=r['target_id'],passed=r['passed'])),flush=True)
    summaries = {}
    for split,name in [('test','unseen'),('train','regression')]:
        rows = [r for r in results if r['split']==split]
        summaries[name] = dict(trials=len(rows),survived=sum(r['completed'] for r in rows),
            reached=sum(r['reached'] and r['completed'] for r in rows),stable=sum(r['stable'] for r in rows),
            passed=sum(r['passed'] for r in rows))
    protocol=('predictive_bounded_reference_v3' if manifest['version']>=3 else
              'validated_ik_guide_live_waist_correction_v2' if manifest['version']>=2 else 'online_original_ik_shared_guard_v1')
    report = dict(protocol=protocol,bank=str(args.bank.resolve()),
        bank_sha256=manifest['bank_sha256'],onnx_sha256=hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        source=manifest['source'],xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        terrain='flat',selected_target_ids=[t['id'] for t in selected],durations=durations,
        execution=manifest['recipe'],summary=summaries,results=results,
        limitations='Finite single-initial-state simulation coverage; guard samples paths, not a continuous collision proof; no hardware validation.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    main()
