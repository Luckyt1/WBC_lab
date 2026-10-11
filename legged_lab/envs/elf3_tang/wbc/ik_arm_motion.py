"""GPU replay of procedural IK paths with explicit complete motion phases.

Only the selected split is loaded. Hard sampling uses predeclared TRAIN failures;
held-out scores never change this sampler. Limits and the collision projector run
on live geometry after this path generator. No additional actor inputs required.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class IkArmMotion:
    def __init__(self, bank, split, names, defaults, dt):
        bank = Path(bank)
        self.manifest = json.loads((bank/'manifest.json').read_text())
        if list(names) != self.manifest['arm_joint_names']:
            raise ValueError('IK bank joint order differs from robot')
        path = bank/(split+'.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest['bank_sha256'][split]:
            raise ValueError('IK bank changed')
        if split not in ('train', 'test') or abs(dt-.02)>1e-8:
            raise ValueError('IK recipe requires train/test split and 50 Hz control')
        self.dt, self.split, self.device = dt, split, defaults.device
        data = np.load(path, allow_pickle=False)
        self.ids = list(data['ids'])
        self.group_names = sorted(set(data['groups']))
        self.groups = torch.tensor([self.group_names.index(g) for g in data['groups']], device=self.device)
        self.paths = torch.as_tensor(data['paths'].copy(), device=self.device)
        self.min_durations = (torch.as_tensor(data['min_durations'].copy(),device=self.device)
                              if 'min_durations' in data else None)
        self.hard = torch.as_tensor(data['hard'].copy(), device=self.device)
        targets = {t['id']:t for t in self.manifest['targets']}
        if any(targets[i]['split'] != split for i in self.ids):
            raise ValueError('Train/test leakage in IK bank')
        self.goal_positions = torch.tensor([targets[i]['positions'] for i in self.ids], device=self.device)
        self.goal_orientations = torch.tensor([targets[i]['orientations_wxyz'] for i in self.ids], device=self.device)
        self.arrival_gate = self.manifest['recipe'].get('arrival_gate',False)
        weights = torch.ones(len(self.ids), device=self.device)
        if split == 'train':
            normal = ~self.hard
            nonempty = sum(bool(((self.groups==g)&normal).any()) for g in range(len(self.group_names)))
            fraction = self.manifest['recipe']['hard_fraction_of_active'] if self.hard.any() else 0.
            for g in range(len(self.group_names)):
                members = (self.groups==g)&normal
                weights[members] = (1-fraction)/nonempty/members.sum().clamp(min=1)
            weights[self.hard] = fraction/self.hard.sum().clamp(min=1)
        self.weights = weights/weights.sum()
        self.defaults = defaults.clone()
        self.target = defaults.clone()
        self.requested_target = defaults.clone()
        self.start = defaults.clone()
        n = len(defaults)
        self.index = torch.zeros(n, dtype=torch.long, device=self.device)
        self.elapsed = torch.zeros(n, device=self.device)
        self.durations = torch.ones(n, 5, device=self.device)
        self.phase = torch.zeros(n, dtype=torch.long, device=self.device)
        self.neutral = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.sample_counts = torch.zeros(len(self.ids), device=self.device)
        self.completed_cycles = torch.zeros((), device=self.device)
        self.metrics = {}
        self.phase_elapsed = torch.zeros(n,device=self.device)
        self.arrival_dwell = torch.zeros(n,device=self.device)
        self.position_error = torch.full((n,),torch.inf,device=self.device)
        self.orientation_error = self.position_error.clone()
        self.home_error = self.position_error.clone()
        self.arm_speed = self.position_error.clone()
        self.cycle_finished = torch.zeros(n,dtype=torch.bool,device=self.device)
        self.attempts = torch.zeros((),device=self.device)
        self.arrivals = torch.zeros_like(self.attempts)
        self.reach_timeouts = torch.zeros_like(self.attempts)
        self.return_timeouts = torch.zeros_like(self.attempts)
        if self.arrival_gate:
            from .arm_pose_tracker import ArmPoseTracker
            tracker=ArmPoseTracker(self.manifest,self.device)
            self.home_p,self.home_q=tracker.guide_pose(self.defaults)

    def reset(self, ids, actual):
        if len(ids)==0: return
        self.index[ids] = torch.multinomial(self.weights, len(ids), replacement=True)
        self.sample_counts.scatter_add_(0, self.index[ids], torch.ones(len(ids), device=self.device))
        recipe = self.manifest['recipe']
        ranges = self.defaults.new_tensor([recipe[s+'_s'] for s in ('prepare','reach','hold','return','rest')])
        self.durations[ids] = ranges[:, 0]+torch.rand(len(ids),5,device=self.device)*(ranges[:,1]-ranges[:,0])
        if self.min_durations is not None:
            self.durations[ids,1]=torch.maximum(self.durations[ids,1],self.min_durations[self.index[ids],0])
            self.durations[ids,3]=torch.maximum(self.durations[ids,3],self.min_durations[self.index[ids],1])
        self.neutral[ids] = torch.rand(len(ids), device=self.device)<recipe['neutral_fraction'] if self.split=='train' else False
        self.start[ids] = self.target[ids] = actual
        self.elapsed[ids] = 0
        self.phase[ids] = 0
        self.phase_elapsed[ids] = 0
        self.arrival_dwell[ids] = 0
        self.position_error[ids] = self.orientation_error[ids] = self.home_error[ids] = torch.inf
        self.arm_speed[ids]=torch.inf
        self.cycle_finished[ids] = False

    def _interpolate(self, leg, fraction):
        frame = fraction.clamp(0,1)*(self.paths.shape[2]-1)
        lo = frame.long()
        hi = (lo+1).clamp(max=self.paths.shape[2]-1)
        a = (frame-lo)[:,None]
        return (1-a)*self.paths[self.index, leg, lo]+a*self.paths[self.index, leg, hi]

    def step(self,remaining_s=None):
        if self.arrival_gate:
            return self._step_gated(remaining_s)
        complete = self.elapsed >= self.durations.sum(-1)
        ids = complete.nonzero().flatten()
        self.completed_cycles += complete.sum()
        self.reset(ids, self.target[ids])
        ends = self.durations.cumsum(-1)
        self.phase = (self.elapsed[:,None]>=ends[:,:-1]).sum(-1)
        out_fraction = (self.elapsed-ends[:,0])/self.durations[:,1]
        in_fraction = (self.elapsed-ends[:,2])/self.durations[:,3]
        desired = self._interpolate(0, out_fraction)
        desired = torch.where((self.phase>=3)[:,None], self._interpolate(1,in_fraction),desired)
        # Prepare/return rest are smooth connections, never discontinuous resets.
        a = (self.elapsed/self.durations[:,0]).clamp(0,1)
        a = a*a*(3-2*a)
        prepared = self.start+a[:,None]*(self.defaults-self.start)
        desired = torch.where((self.phase==0)[:,None], prepared,desired)
        a = ((self.elapsed-ends[:,3])/self.durations[:,4]).clamp(0,1)
        a = a*a*(3-2*a)
        final = self.paths[self.index,1,-1]
        desired = torch.where((self.phase==4)[:,None],final+a[:,None]*(self.defaults-final),desired)
        self.requested_target.copy_(torch.where(self.neutral[:,None],self.defaults,desired))
        self.elapsed += self.dt
        active = ~self.neutral
        self.metrics = {'IK/neutral_fraction':self.neutral.float().mean(),
                        'IK/hard_active_fraction':(self.hard[self.index]&active).sum()/active.sum().clamp(min=1),
                        'IK/unique_target_coverage':(self.sample_counts>0).float().mean(),
                        'IK/completed_cycles':self.completed_cycles.clone()}
        for i, name in enumerate(('prepare','reach','hold','return','rest')):
            self.metrics['IK/phase_'+name+'_fraction'] = ((self.phase==i)&active).float().mean()
        for i, name in enumerate(self.group_names):
            self.metrics['IK/group_'+name+'_fraction'] = ((self.groups[self.index]==i)&active).float().mean()
        return self.requested_target

    def _step_gated(self,remaining_s):
        recipe=self.manifest['recipe']
        old=self.phase.clone()
        elapsed=self.phase_elapsed
        active=~self.neutral
        starting=(old==0)&(elapsed>=self.durations[:,0])
        self.attempts+=(starting&active).sum()
        in_reach=(old==1)&(elapsed>=self.durations[:,1])
        arrived=(self.position_error<=recipe['arrival_position_m'])&(self.orientation_error<=recipe['arrival_orientation_rad'])
        if 'arrival_joint_velocity_rad_s' in recipe:
            arrived &= self.arm_speed<=recipe['arrival_joint_velocity_rad_s']
        self.arrival_dwell=torch.where(in_reach&arrived,self.arrival_dwell+self.dt,torch.zeros_like(elapsed))
        reached=in_reach&((self.arrival_dwell>=recipe['arrival_dwell_s'])|self.neutral)
        failed=in_reach&~reached&(elapsed>=self.durations[:,1]+recipe['arrival_timeout_s'])
        self.arrivals+=(reached&active).sum();self.reach_timeouts+=(failed&active).sum()
        held=(old==2)&(elapsed>=self.durations[:,2])
        in_return=(old==3)&(elapsed>=self.durations[:,3])
        returned=in_return&(self.home_error<=1.)
        return_failed=in_return&~returned&(elapsed>=self.durations[:,3]+recipe['arrival_timeout_s'])
        self.return_timeouts+=(return_failed&active).sum()
        complete=(old==4)&(elapsed>=self.durations[:,4])&~self.cycle_finished
        self.completed_cycles+=complete.sum();self.cycle_finished|=complete
        # One complete attempt per episode. Stay at neutral after return rather
        # than starting another task that could be truncated by the episode end.
        self.phase[starting]=1;self.phase[reached]=2;self.phase[failed|held]=3
        self.phase[returned|return_failed]=4
        changed=self.phase!=old
        self.phase_elapsed[changed]=0.
        out=self._interpolate(0,self.phase_elapsed/self.durations[:,1])
        out=torch.where((self.phase==2)[:,None],self.paths[self.index,0,-1],out)
        back=self._interpolate(1,self.phase_elapsed/self.durations[:,3])
        out=torch.where((self.phase==3)[:,None],back,out)
        a=(self.phase_elapsed/self.durations[:,0]).clamp(0,1);a=a*a*(3-2*a)
        out=torch.where((self.phase==0)[:,None],self.start+a[:,None]*(self.defaults-self.start),out)
        a=(self.phase_elapsed/self.durations[:,4]).clamp(0,1);a=a*a*(3-2*a)
        final=self.paths[self.index,1,-1]
        out=torch.where((self.phase==4)[:,None],final+a[:,None]*(self.defaults-final),out)
        self.requested_target.copy_(torch.where(self.neutral[:,None],self.defaults,out))
        self.phase_elapsed+=self.dt;self.elapsed+=self.dt
        self.metrics={'IK/neutral_fraction':self.neutral.float().mean(),
            'IK/hard_active_fraction':(self.hard[self.index]&active).sum()/active.sum().clamp(min=1),
            'IK/unique_target_coverage':(self.sample_counts>0).float().mean(),
            'IK/completed_cycles':self.completed_cycles.clone(),
            'IK/reach_attempts':self.attempts.clone(),'IK/reached_targets':self.arrivals.clone(),
            'IK/reach_timeouts':self.reach_timeouts.clone(),'IK/return_timeouts':self.return_timeouts.clone(),
            'IK/arrival_rate':self.arrivals/(self.arrivals+self.reach_timeouts).clamp(min=1),
            'IK/completed_wait_fraction':self.cycle_finished.float().mean()}
        for i,name in enumerate(('prepare','reach','hold','return','rest')):
            self.metrics['IK/phase_'+name+'_fraction']=((self.phase==i)&active).float().mean()
        for i,name in enumerate(self.group_names):
            self.metrics['IK/group_'+name+'_fraction']=((self.groups[self.index]==i)&active).float().mean()
        return self.requested_target

    def tracking_metrics(self, actual_wrist_positions,actual_wrist_orientations=None,actual_arm_velocity=None):
        error = (actual_wrist_positions-self.goal_positions[self.index]).norm(dim=-1).amax(-1)
        if actual_arm_velocity is not None:
            self.arm_speed.copy_(actual_arm_velocity.abs().amax(-1))
        if self.arrival_gate:
            from .arm_pose_tracker import rotation_error
            if actual_wrist_orientations is None: raise ValueError('Arrival gating requires wrist orientation feedback')
            self.position_error.copy_(error)
            self.orientation_error.copy_(rotation_error(self.goal_orientations[self.index],actual_wrist_orientations).norm(dim=-1).amax(-1))
            recipe=self.manifest['recipe']
            if 'arrival_joint_velocity_rad_s' in recipe:
                if actual_arm_velocity is None: raise ValueError('v3 arrival requires actual joint velocity')
                self.arm_speed.copy_(actual_arm_velocity.abs().amax(-1))
            self.home_error.copy_(torch.maximum((actual_wrist_positions-self.home_p).norm(dim=-1).amax(-1)/recipe['arrival_position_m'],
                rotation_error(self.home_q,actual_wrist_orientations).norm(dim=-1).amax(-1)/recipe['arrival_orientation_rad']))
        holding = (self.phase==2)&~self.neutral
        return {'IK/hold_active_envs':holding.sum().float(),
                'IK/hold_wrist_position_error_m':torch.where(holding,error,0.).sum()/holding.sum().clamp(min=1),
                'IK/hold_position_within_2cm_fraction':((error<.02)&holding).sum()/holding.sum().clamp(min=1)}
