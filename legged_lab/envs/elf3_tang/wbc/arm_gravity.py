"""Batched version of the pinned deployment arm gravity model (no ROS)."""
import json
from pathlib import Path
import torch
from .arm_guard import axis_quat, quat_mul, rotate


class ArmGravity:
    def __init__(self, device):
        path = Path(__file__).resolve().parents[3]/'assets/elf3_ik_upstream/gravity_constants.json'
        c = json.loads(path.read_text())
        self.origins = torch.tensor([c['_JOINT_ORIGINS_LEFT'], c['_JOINT_ORIGINS_RIGHT']], device=device)
        self.coms = torch.tensor([c['_LINK_COMS_LEFT'], c['_LINK_COMS_RIGHT']], device=device)
        self.masses = torch.tensor([c['_LINK_MASSES_LEFT'], c['_LINK_MASSES_RIGHT']], device=device)
        self.axes = torch.tensor(c['_JOINT_AXES'], device=device)
        self.descendants = torch.ones(7,7,device=device).triu()
        self.limits = .8*torch.tensor([45.,45.,21.,45.,21.,21.,21.]*2,device=device)

    def compute(self, positions, gravity_in_torso):
        q = positions.reshape(-1,2,7)
        n = len(q)
        rotation = q.new_zeros(n,2,4); rotation[...,0] = 1
        pos = q.new_zeros(n,2,3)
        joints, axes, centers = [], [], []
        for j, axis in enumerate((1,0,2,1,0,1,2)):
            pos = pos+rotate(rotation,self.origins[:,j].expand(n,-1,-1))
            joints.append(pos)
            axes.append(rotate(rotation,self.axes[j].expand(n,2,-1)))
            rotation = quat_mul(rotation,axis_quat(q[...,j],axis))
            centers.append(pos+rotate(rotation,self.coms[:,j].expand(n,-1,-1)))
        joints, axes, centers = (torch.stack(v,2) for v in (joints,axes,centers))
        displacement = centers[:,:,None,:,:]-joints[:,:,:,None,:]
        force = self.masses[None,:,:,None]*gravity_in_torso[:,None,None,:]
        moment = torch.cross(displacement,force[:,:,None,:,:].expand_as(displacement),dim=-1)
        moment = (moment*self.descendants[None,None,:,:,None]).sum(-2)
        return (-(moment*axes).sum(-1)).reshape(n,14).clamp(-self.limits,self.limits)
