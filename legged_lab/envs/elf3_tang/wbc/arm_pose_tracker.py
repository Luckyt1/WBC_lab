"""Small batched IK correction of a validated guide for the measured waist pose.

The offline upstream IK chooses a continuous arm branch. This bounded DLS stage
keeps that guide's pelvis-relative wrist pose when the live waist moves. Joint
limits, velocity limits and collision projection still run after correction.
"""
import torch
from .arm_guard import axis_quat, quat_mul, rotate, inverse


def kinematics(q):
    """Wrist pose and spatial geometric Jacobian, (...,2,7), torso frame."""
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    shoulder = q.new_tensor([[0.,.178,.087],[0.,-.178,.087]]).expand(*q.shape[:-1],3)
    ry = axis_quat(q[...,0],1)
    rx = quat_mul(ry,axis_quat(q[...,1],0))
    upper = quat_mul(rx,axis_quat(q[...,2],2))
    elbow = shoulder+rotate(upper,q.new_tensor([0.,0.,-.256]).expand_as(shoulder))
    er = quat_mul(upper,axis_quat(q[...,3],1))
    wrist = elbow+rotate(er,q.new_tensor([.256,0.,0.]).expand_as(shoulder))
    wx = quat_mul(er,axis_quat(q[...,4],0))
    wy = quat_mul(wx,axis_quat(q[...,5],1))
    wz = quat_mul(wy,axis_quat(q[...,6],2))
    def axis(rot,i): return rotate(rot,eye[i].expand_as(shoulder))
    axes=torch.stack((eye[1].expand_as(shoulder),axis(ry,0),axis(rx,2),axis(upper,1),
                      axis(er,0),axis(wx,1),axis(wy,2)),-1)
    origins=torch.stack((shoulder,shoulder,shoulder,elbow,wrist,wrist,wrist),-1)
    linear=torch.cross(axes,wrist[...,None]-origins,dim=-2)
    return wrist,wz,torch.cat((linear,axes),-2)


def rotation_error(target, actual):
    q=quat_mul(target,inverse(actual))
    q=torch.where(q[...,:1]<0,-q,q)
    norm=q[...,1:].norm(dim=-1,keepdim=True)
    return q[...,1:]*(2*torch.atan2(norm,q[...,:1].clamp(min=0)) / norm.clamp(min=1e-8))


class ArmPoseTracker:
    def __init__(self,manifest,device):
        nominal=manifest['nominal_torso_in_pelvis']
        self.p=torch.tensor(nominal['position'],device=device,dtype=torch.float32)
        self.r=torch.tensor(nominal['orientation_wxyz'],device=device,dtype=torch.float32)
        self.limits=torch.tensor(manifest['joint_limits'],device=device,dtype=torch.float32).reshape(2,7,2)
        self.eye=torch.eye(6,device=device)
        self.metrics={}

    def guide_pose(self,guide):
        p,r,_=kinematics(guide.reshape(-1,2,7))
        return self.p+rotate(self.r.expand_as(r),p),quat_mul(self.r,r)

    @torch.no_grad()
    def correct(self,guide,pelvis_p,pelvis_q,torso_p,torso_q):
        target_p,target_q=self.guide_pose(guide)
        p=rotate(inverse(torso_q)[:,None,:],pelvis_p[:,None,:]+rotate(pelvis_q[:,None,:],target_p)-torso_p[:,None,:])
        r=quat_mul(inverse(torso_q)[:,None,:],quat_mul(pelvis_q[:,None,:],target_q))
        q=guide.reshape(-1,2,7).clone()
        for _ in range(3):
            actual_p,actual_r,J=kinematics(q)
            error=torch.cat((p-actual_p,rotation_error(r,actual_r)),-1)
            solution,_=torch.linalg.solve_ex(J@J.transpose(-1,-2)+.001*self.eye,error[...,None],check_errors=False)
            dq=J.transpose(-1,-2)@solution
            q=(q+.8*dq[...,0].clamp(-.15,.15)).clamp(self.limits[...,0],self.limits[...,1])
        actual_p,actual_r,_=kinematics(q)
        self.metrics={'IK/correction_position_residual_m':(p-actual_p).norm(dim=-1).amax(-1).mean(),
                      'IK/correction_orientation_residual_rad':rotation_error(r,actual_r).norm(dim=-1).amax(-1).mean()}
        return q.reshape(-1,14)
