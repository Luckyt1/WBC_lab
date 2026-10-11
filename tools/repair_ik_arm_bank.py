"""Build a versioned, execution-validated IK bank without changing old inputs.

Try original paths first, then FK waypoint guides and alternate IK seeds. Every
accepted path must preserve the original wrist goal, pass the shared collision
geometry and actual MuJoCo self-contact check, and reach and return accurately.
Rejected goals remain in the manifest and evaluation pool, never count as passes.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

from evaluate_ik_policy import ArmCollisionGuard
from elf3_cpu.wbc.arm_guard import arm_frames, rotate, quat_mul
from elf3_ik_core import new_solver, forward_targets
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import pinocchio as pin


class PathBuilder:
    def __init__(self, manifest, xml):
        torch.set_num_threads(1)
        self.m = manifest
        self.module, self.solver, self.limiter, _ = new_solver()
        self.default = np.asarray(manifest['default_arm_q'])
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[2] = .85
        self.ids = [self.model.joint(n).qposadr[0] for n in manifest['arm_joint_names']]
        self.data.qpos[self.ids] = self.default
        mujoco.mj_forward(self.model, self.data)
        self.names = [self.model.body(i).name for i in range(self.model.nbody)]
        self.torso = self.names.index('torso_link')
        self.pos = torch.tensor(self.data.xpos.copy(),dtype=torch.float32)
        self.rot = torch.tensor(self.data.xquat.copy(),dtype=torch.float32)
        self.guard = ArmCollisionGuard(manifest['arm_joint_names'],self.names,
            torch.tensor(manifest['joint_limits']),'cpu',margin=.003,max_velocity=6.,geometry_mode='obb_v2')
        self.arm_bodies = {i for i,n in enumerate(self.names) if any(p in n for p in ('shoulder','elbow','wrist'))}
        self.frame_ids = [self.names.index(s+'_'+p+'_link') for s in ('l','r') for p in ('shoulder_z','wrist_x','wrist_y','wrist_z')]

    def clearances(self, path):
        q = torch.tensor(np.asarray(path).reshape(-1,14),dtype=torch.float32)
        pos,rot = self.pos.repeat(len(q),1,1),self.rot.repeat(len(q),1,1)
        p,r = arm_frames(q.reshape(-1,2,7)); p=p.reshape(-1,8,3);r=r.reshape(-1,8,4)
        tq=rot[:,self.torso,None,:]
        pos[:,self.frame_ids]=pos[:,self.torso,None,:]+rotate(tq,p)
        rot[:,self.frame_ids]=quat_mul(tq,r)
        c,r=self.guard.obstacle_frames(pos,rot,self.torso)
        return self.guard.clearance(q.reshape(-1,1,2,7),c,r).numpy()[:,0]

    def error(self, q, goal):
        p,r=forward_targets(self.solver,q)
        return [float(np.linalg.norm(p-np.asarray(goal['positions']),axis=1).max()),
                float((Rotation.from_quat(r[:,[1,2,3,0]]).inv()*Rotation.from_quat(np.asarray(goal['orientations_wxyz'])[:,[1,2,3,0]])).magnitude().max())]

    def physical_clear(self,path):
        for q in path:
            self.data.qpos[self.ids]=q
            mujoco.mj_forward(self.model,self.data)
            for c in self.data.contact:
                a,b=(int(self.model.geom_bodyid[i]) for i in (c.geom1,c.geom2))
                if a and b and a!=b and c.dist < -.0001 and (a in self.arm_bodies or b in self.arm_bodies):
                    return False
        return True

    def valid(self,path,goal):
        if not np.isfinite(path).all() or np.abs(np.diff(path,axis=0)).max()>.120001: return None
        error=self.error(path[-1],goal)
        if error[0]>.005 or error[1]>.03: return None
        # Validate intermediate samples as well as the control points.
        sampled=np.concatenate((path,.5*(path[1:]+path[:-1])))
        clearance=float(self.clearances(sampled).min())
        if clearance < .003: return None
        if not self.physical_clear(sampled): return None
        return dict(endpoint_error=error,minimum_nominal_clearance_m=clearance,
                    physical_self_penetration_over_point1mm=False)

    def guide(self, waypoints):
        points=np.array(waypoints)
        lengths=np.abs(np.diff(points,axis=0)).max(1)
        cumulative=np.r_[0,np.cumsum(lengths)]; cumulative/=max(cumulative[-1],1e-12)
        t=np.linspace(0,1,126); t=t*t*(3-2*t)
        path=np.stack([np.interp(t,cumulative,points[:,j]) for j in range(14)],axis=1)
        return np.concatenate((path,np.repeat(path[-1:],25,axis=0)))

    def solve_guide(self, guide):
        # A collision-checked guide selects the redundant IK branch. Seed each
        # upstream solve on that guide, then limit relative to the last command.
        # This avoids null-space drift back into the blocked elbow configuration.
        reference=guide[0].copy(); self.limiter._last_published_arm_positions=reference.copy()
        path=[reference]
        for waypoint in guide[1:]:
            p,r=forward_targets(self.solver,waypoint)
            self.solver.update_measured(self.module.ARM_JOINTS,waypoint)
            result=self.solver.solve(p,r)
            if result.positions is None: return None
            reference=self.limiter._rate_limited_arm_reference(result.positions).astype(float)
            path.append(reference.copy())
        return np.array(path)

    def repair(self,target,old):
        neutral_p,neutral_q=forward_targets(self.solver,self.default)
        home=dict(positions=neutral_p,orientations_wxyz=neutral_q)
        # Nominal endpoint residuals must be measured after rate limiting.
        a,b=self.valid(old[0],target),self.valid(old[1],home)
        if a and b:
            return old,dict(method='original_cartesian_ik',outbound=a,inbound=b)
        seed=int.from_bytes(hashlib.sha256(target['id'].encode()).digest()[:4],'little')
        rng=np.random.default_rng(seed)
        gp,gq=np.array(target['positions']),np.array(target['orientations_wxyz'])
        def endpoints():
            yield np.asarray(target['witness_arm_q'])
            yield old[0,-1]
            for k in range(96):
                start = (self.guard.neutral.numpy().ravel() if k==0 else
                         rng.uniform(self.solver._soft_lower,self.solver._soft_upper))
                self.solver.update_measured(self.module.ARM_JOINTS,start)
                iterations=self.solver.iterations
                self.solver.iterations=96  # bounded offline multistart search
                result=self.solver.solve(gp,gq)
                self.solver.iterations=iterations
                if result.success: yield result.positions.astype(float)
        for endpoint in endpoints():
            if max(self.error(endpoint,target)[0]/.005,self.error(endpoint,target)[1]/.03)>1: continue
            if self.clearances([endpoint]).min()<.003: continue
            side=np.array([-.7,.85,0,1.2,0,0,0]*2);side[8]*=-1
            forward=np.array([-1.5,.6,0,1.,0,0,0]*2);forward[8]*=-1
            for via in ([],[self.guard.neutral.numpy().ravel()],[side],[forward]):
                guide=self.guide([self.default]+via+[endpoint])
                if self.clearances(guide).min()<.004: continue
                path=self.solve_guide(guide)
                if path is None: continue
                a=self.valid(path,target)
                if not a: continue
                back=self.solve_guide(self.guide([path[-1]]+via[::-1]+[self.default]))
                if back is None: continue
                b=self.valid(back,home)
                if b:
                    return np.stack((path,back)),dict(method='planned_branch_upstream_ik_waypoints',outbound=a,inbound=b)
        return None,dict(method='rejected',reason='no_verified_reaching_path_within_search_budget',
                         old_endpoint_error=self.error(old[0,-1],target))


BUILDER=None
def initialize(manifest,xml):
    global BUILDER
    BUILDER=PathBuilder(manifest,xml)


def build(args):
    target,path=args
    repaired,report=BUILDER.repair(target,path)
    return target['id'],repaired,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--xml',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--only',nargs='*')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.source/'manifest.json').read_text())
    paths={}
    for split in ('train','test'):
        data=np.load(args.source/(split+'.npz'))
        paths.update(zip(data['ids'],data['paths']))
    selected=[t for t in manifest['targets'] if args.only is None or t['id'] in args.only]
    results={}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(manifest,args.xml)) as pool:
        for future in as_completed([pool.submit(build,(t,paths[t['id']])) for t in selected]):
            ident,path,report=future.result();results[ident]=(path,report)
            print(json.dumps(dict(completed=len(results),total=len(selected),id=ident,**report)),flush=True)
    (args.output/'validation.json').write_text(json.dumps({k:v[1] for k,v in results.items()},indent=2)+'\n')
    accepted=[];rejected=[]
    for t in selected:
        path,report=results[t['id']]
        t['execution_validation']=report
        if path is None: rejected.append(t)
        else: accepted.append(t)
    for split in ('train','test'):
        subset=[t for t in accepted if t['split']==split]
        if not subset: continue
        np.savez_compressed(args.output/(split+'.npz'),paths=np.stack([results[t['id']][0] for t in subset]).astype(np.float32),
            ids=np.array([t['id'] for t in subset]),hard=np.array([t['hard'] for t in subset]),groups=np.array([t['group'] for t in subset]))
    manifest.update(version=2,targets=accepted,rejected_targets=rejected,parent_bank=str(args.source.resolve()),
        parent_manifest_sha256=hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
        counts=dict(train=sum(t['split']=='train' for t in accepted),test=sum(t['split']=='test' for t in accepted),
                    hard=sum(t['hard'] for t in accepted),rejected=len(rejected)))
    manifest['recipe'].update(guard_geometry='obb_v2',guard_margin=.003,arrival_gate=True,
        arrival_position_m=.02,arrival_orientation_rad=.1,arrival_dwell_s=.1,arrival_timeout_s=3.,max_episode_s=28.,
        waist_pose_correction=True)
    _,solver,_,_=new_solver()
    pin.framesForwardKinematics(solver.model,solver.data,pin.neutral(solver.model))
    transform=solver.data.oMf[solver._anchor_frame].inverse()*solver.data.oMf[solver.model.getFrameId('torso_link')]
    manifest['nominal_torso_in_pelvis']=dict(position=transform.translation.tolist(),
        orientation_wxyz=Rotation.from_matrix(transform.rotation).as_quat()[[3,0,1,2]].tolist())
    manifest['generation']['validation']='Both legs: FK endpoint <=5mm/0.03rad after limiter, OBB >=3mm at control points/midpoints, MuJoCo self-penetration <=0.1mm; up to 96 alternate original-IK seeds with 96 iterations, and collision-checked joint guides seeding original IK at Cartesian waypoints.'
    manifest['generation']['nominal_feedback']='planned joint guide selects IK seed; previous rate-limited command sets velocity limit; live waist correction and physical feedback guard during execution'
    manifest['limitations']='Finite nominal-waist collision and endpoint checks, not a continuous path or hardware proof. Live body geometry may block a path; arrival timeout is logged. Rejected goals stay in evaluation and are not successes.'
    manifest['bank_sha256']={s:hashlib.sha256((args.output/(s+'.npz')).read_bytes()).hexdigest() for s in ('train','test') if (args.output/(s+'.npz')).exists()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',**manifest['counts'])),flush=True)


if __name__=='__main__':main()
