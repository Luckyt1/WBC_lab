"""Shared Isaac/MuJoCo v3 executor: bounded references and predictive braking.

Ruckig limits commanded derivatives, not measured physical acceleration. Collision
prediction uses sampled horizons and constant obstacle velocity in the torso frame;
it is not a continuous-time or arbitrary-disturbance safety guarantee.
"""
import numpy as np
import torch
from .arm_guard import arm_frames, rotate, quat_mul, inverse
from .arm_pose_tracker import rotation_error


class PredictiveArmExecutor:
    def __init__(self, guard, defaults, recipe, body_names, dt=.02):
        try:
            from . import _arm_ruckig
        except ImportError as exc:
            raise RuntimeError('Build local limiter first: python tools/build_arm_ruckig.py') from exc
        self.native=_arm_ruckig
        self.guard,self.device,self.dt=guard,defaults.device,dt
        self.vmax,self.amax,self.jmax=(float(recipe[k]) for k in ('max_velocity','max_acceleration','max_jerk'))
        self.times=np.asarray(recipe['prediction_times_s'],dtype=np.float64)
        if abs(self.times[0]-dt)>1e-9 or np.any(np.diff(self.times)<=0): raise ValueError('Invalid prediction times')
        self.extra=float(recipe['dynamic_extra_margin_m'])
        self.state=np.zeros((len(defaults),3,14),dtype=np.float64)
        self.state[:,0]=defaults.detach().cpu().numpy()
        self.fresh=torch.ones(len(defaults),dtype=torch.bool,device=self.device)
        self.prev_centers=self.prev_rotations=None
        self.obstacle_arm=[]
        arm_names=[f'{s}_{part}_link' for s in ('l','r') for part in ('shoulder_z','wrist_x','wrist_y','wrist_z')]
        for a,name in enumerate(arm_names):
            bid=body_names.index(name)
            if bid in guard.body_ids: self.obstacle_arm.append((guard.body_ids.index(bid),a))
        self.dynamic=torch.tensor([body_names[i]!='torso_link' for i in guard.body_ids],device=self.device)
        self.metrics={}

    def reset(self,ids,actual):
        indices=ids.detach().cpu().numpy()
        self.state[indices]=0.;self.state[indices,0]=actual.detach().cpu().numpy()
        self.fresh[ids]=True

    def _calculate(self,state,goal,brake=False):
        out,extrema,status=self.native.calculate(state,goal,self.times,self.vmax,self.amax,self.jmax,brake,4)
        if np.any(status<0):
            raise RuntimeError(f'Arm trajectory solver failed: {np.unique(status).tolist()}')
        return out,extrema

    def _clearance(self,preview,centers,rotations,linear,angular,actual,per_horizon=False):
        n,h=preview.shape[:2];g=self.guard
        times=torch.as_tensor(self.times,device=self.device,dtype=actual.dtype)
        q=preview[:,:,0].reshape(n*h,2,7)
        cp=centers[:,None]+times[None,:,None,None]*linear[:,None]
        angle=times[None,:,None]*angular.norm(dim=-1)[:,None]
        axis=angular/angular.norm(dim=-1,keepdim=True).clamp(min=1e-8)
        dq=torch.cat(((angle*.5).cos()[...,None],axis[:,None]*(angle*.5).sin()[...,None]),-1)
        rp=quat_mul(dq,rotations[:,None].expand(-1,h,-1,-1)).clone()
        cp=cp.reshape(n*h,-1,3).clone();rp=rp.reshape(n*h,-1,4)
        # Opposite-arm obstacles must use the SAME simultaneous candidate, not
        # stale measured poses of the other arm.
        p,r=arm_frames(q);p=p+rotate(r,g.arm_centers.expand_as(p))
        p=p.reshape(n*h,8,3);r=r.reshape(n*h,8,4)
        for obstacle,arm in self.obstacle_arm: cp[:,obstacle]=p[:,arm];rp[:,obstacle]=r[:,arm]
        speed=linear.norm(dim=-1)+.15*angular.norm(dim=-1)
        extra=self.extra*(speed/.4).clamp(0,1)*self.dynamic
        extra=extra[:,None].expand(-1,h,-1).reshape(n*h,-1)
        clear=g.box_clearance_batched(q[:,None],cp,rp,extra)[:,0].reshape(n,h,2).amin(-1)
        if per_horizon: return clear
        clear=clear.amin(-1)
        # Also check the physical-to-next-command bridge at current geometry.
        midpoint=.5*(actual+preview[:,0,0])
        bridge=g.box_clearance_batched(midpoint.reshape(n,1,2,7),centers,rotations)[:,0].amin(-1)
        return torch.minimum(clear,bridge)

    @torch.no_grad()
    def project(self,desired,actual,body_pos,body_quat,torso_id):
        g=self.guard;n=len(desired)
        limits=g.limits.reshape(-1,14,2).expand(n,-1,-1)
        goal=desired.clamp(limits[...,0],limits[...,1]).detach().cpu().numpy().astype(np.float64)
        centers,rotations=g.obstacle_frames(body_pos,body_quat,torso_id)
        if self.prev_centers is None:
            self.prev_centers=centers.clone();self.prev_rotations=rotations.clone()
        linear=((centers-self.prev_centers)/self.dt).clamp(-2.,2.)
        angular=(rotation_error(rotations,self.prev_rotations)/self.dt).clamp(-6.,6.)
        linear[self.fresh]=0.;angular[self.fresh]=0.;self.fresh[:]=False
        self.prev_centers.copy_(centers);self.prev_rotations.copy_(rotations)
        preview,extrema=self._calculate(self.state,goal)
        tensor=torch.as_tensor(preview,device=self.device,dtype=desired.dtype)
        clearance=self._clearance(tensor,centers,rotations,linear,angular,actual)
        bounds=torch.as_tensor(extrema,device=self.device,dtype=desired.dtype)
        bounded=((bounds[...,0]>=limits[...,0]-1e-6)&(bounds[...,1]<=limits[...,1]+1e-6)).all(-1)
        blocked=(clearance<g.margin)|~bounded
        ids=blocked.nonzero().flatten();unsafe=torch.zeros(n,dtype=torch.bool,device=self.device)
        recovering=torch.zeros_like(unsafe)
        if len(ids):
            cpuids=ids.cpu().numpy()
            stop,stop_ext=self._calculate(self.state[cpuids],goal[cpuids],brake=True)
            stop_tensor=torch.as_tensor(stop,device=self.device,dtype=desired.dtype)
            stop_clear=self._clearance(stop_tensor,centers[ids],rotations[ids],linear[ids],angular[ids],actual[ids])
            sb=torch.as_tensor(stop_ext,device=self.device,dtype=desired.dtype)
            stop_bounded=((sb[...,0]>=limits[ids,:,0]-1e-6)&(sb[...,1]<=limits[ids,:,1]+1e-6)).all(-1)
            unsafe[ids]=(stop_clear<g.margin)|~stop_bounded
            trapped=unsafe[ids].nonzero().flatten()
            if len(trapped):
                # Reset poses or moving obstacles can already violate the guard
                # margin. A pure brake would stay trapped forever. Try a bounded
                # outward retreat, accept only if its sampled separation is no
                # worse than braking and its final preview improves separation.
                # This is explicitly recovery from an unsafe set, not certified
                # collision-free motion.
                global_ids=ids[trapped];ci=global_ids.cpu().numpy()
                retreat=g.neutral.reshape(1,14).expand(len(trapped),-1)
                retreat=retreat.clamp(limits[global_ids,:,0],limits[global_ids,:,1]).cpu().numpy().astype(np.float64)
                escape,escape_ext=self._calculate(self.state[ci],retreat)
                et=torch.as_tensor(escape,device=self.device,dtype=desired.dtype)
                args=(centers[global_ids],rotations[global_ids],linear[global_ids],angular[global_ids],actual[global_ids])
                ec=self._clearance(et,*args,per_horizon=True)
                bc=self._clearance(stop_tensor[trapped],*args,per_horizon=True)
                eb=torch.as_tensor(escape_ext,device=self.device,dtype=desired.dtype)
                bounded_escape=((eb[...,0]>=limits[global_ids,:,0]-1e-6)&(eb[...,1]<=limits[global_ids,:,1]+1e-6)).all(-1)
                improve=(ec>=bc-1e-4).all(-1)&(ec[:,-1]>bc[:,-1]+1e-4)&bounded_escape
                selected=trapped[improve];local=improve.cpu().numpy()
                if len(selected):
                    stop[selected.cpu().numpy()]=escape[local]
                    stop_clear[selected]=self._clearance(et[improve],*(a[improve] for a in args))
                    recovering[global_ids[improve]]=True
            # Never teleport to a projected target: braking keeps v/a/j continuous.
            preview[cpuids]=stop;clearance[ids]=stop_clear
        previous_a=self.state[:,2].copy()
        self.state=preview[:,0].copy()
        result=torch.as_tensor(self.state[:,0],device=self.device,dtype=desired.dtype)
        jerk=np.abs((self.state[:,2]-previous_a)/self.dt).max()
        self.metrics={
            'ArmGuard/corrected_arm_fraction':blocked.float().mean(),
            'ArmGuard/target_correction_mae':(result-desired).abs().mean(),
            'ArmGuard/reference_error_mae':(result-desired).abs().mean(),
            'ArmGuard/no_safe_candidate_fraction':unsafe.float().mean(),
            'ArmGuard/selected_clearance_mean':clearance.mean(),
            'ArmGuard/predictive_brake_fraction':blocked.float().mean(),
            'ArmGuard/unresolved_prediction_fraction':unsafe.float().mean(),
            'ArmGuard/outward_recovery_fraction':recovering.float().mean(),
            'IK/reference_velocity_max_rad_s':desired.new_tensor(np.abs(self.state[:,1]).max()),
            'IK/reference_acceleration_max_rad_s2':desired.new_tensor(np.abs(self.state[:,2]).max()),
            'IK/reference_jerk_max_rad_s3':desired.new_tensor(jerk)}
        g.metrics=self.metrics
        return result
