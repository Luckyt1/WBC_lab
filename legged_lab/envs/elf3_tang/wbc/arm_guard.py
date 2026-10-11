"""Batched arm-target projection using conservative bounds of the BXI meshes.

This supervisor preserves PhysX collisions and termination conditions. It checks
sampled paths against current body poses, not a continuous-time safety proof.
The mesh-derived capsule/box approximations deliberately err toward clearance.
"""
import json
from pathlib import Path
import torch


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2), -1)


def rotate(q, v):
    xyz = q[..., 1:]
    t = 2 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz.expand_as(v), t, dim=-1)


def inverse(q):
    return q * q.new_tensor([1., -1., -1., -1.])


def axis_quat(angle, axis):
    q = angle.new_zeros(*angle.shape, 4)
    q[..., 0] = (angle * .5).cos()
    q[..., axis + 1] = (angle * .5).sin()
    return q


def arm_frames(q):
    """Return shoulder-z and wrist x/y/z frames, relative to torso, (...,2,7)."""
    shoulder = q.new_tensor([[0., .178, .087], [0., -.178, .087]]).expand(*q.shape[:-1], 3)
    upper = quat_mul(quat_mul(axis_quat(q[..., 0], 1), axis_quat(q[..., 1], 0)), axis_quat(q[..., 2], 2))
    elbow = shoulder + rotate(upper, q.new_tensor([0., 0., -.256]).expand_as(shoulder))
    elbow_rot = quat_mul(upper, axis_quat(q[..., 3], 1))
    wrist = elbow + rotate(elbow_rot, q.new_tensor([.256, 0., 0.]).expand_as(shoulder))
    wrist_x = quat_mul(elbow_rot, axis_quat(q[..., 4], 0))
    wrist_y = quat_mul(wrist_x, axis_quat(q[..., 5], 1))
    wrist_z = quat_mul(wrist_y, axis_quat(q[..., 6], 2))
    return torch.stack((shoulder, wrist, wrist, wrist), -2), torch.stack((upper, wrist_x, wrist_y, wrist_z), -2)


class ArmCollisionGuard:
    def __init__(self, joint_names, body_names, limits, device, margin=.012, max_velocity=3., dt=.02,
                 geometry_mode='capsule_v1'):
        expected = [f'{s}_{j}_joint' for s in ('l', 'r') for j in
                    ('shoulder_y', 'shoulder_x', 'shoulder_z', 'elbow_y', 'wrist_x', 'wrist_y', 'wrist_z')]
        if list(joint_names) != expected:
            raise ValueError('Arm guard requires named left/right BXI arm order')
        self.device, self.margin, self.max_delta = device, margin, max_velocity * dt
        if geometry_mode not in ('capsule_v1', 'obb_v2'):
            raise ValueError('Unknown arm guard geometry mode')
        self.geometry_mode = geometry_mode
        self.limits = limits.reshape(-1, 2, 7, 2)
        path = Path(__file__).resolve().parents[3] / 'assets/elf3_bxi_31/arm_guard_geometry.json'
        geometry = json.loads(path.read_text())
        arm_names = [f'{s}_{part}_link' for s in ('l', 'r') for part in ('shoulder_z', 'wrist_x', 'wrist_y', 'wrist_z')]
        self.arm_centers = torch.tensor([geometry['boxes'][n]['center'] for n in arm_names], device=device).reshape(2,4,3)
        self.arm_half = torch.tensor([geometry['boxes'][n]['half_extent'] for n in arm_names], device=device).reshape(2,4,3)
        points, radii = [], []
        # A chain of overlapping spheres encloses each entire capsule.
        for name in arm_names:
            cap = geometry['arm_capsules'][name]
            a, b = torch.tensor(cap['start']), torch.tensor(cap['end'])
            samples = torch.linspace(0, 1, 5)[:, None]
            points.append(a + samples * (b-a))
            spacing = (b-a).norm().item() / 4
            radii.append((cap['radius']**2 + (spacing/2)**2)**.5)
        self.points = torch.stack(points).reshape(2, 4, 5, 3).to(device)
        self.radii = torch.tensor(radii, device=device).reshape(2, 4, 1).expand(2, 4, 5)
        obstacle_names = [n for n in geometry['boxes'] if n in body_names]
        self.body_ids = [body_names.index(n) for n in obstacle_names]
        self.centers = torch.tensor([geometry['boxes'][n]['center'] for n in obstacle_names], device=device)
        self.half = torch.tensor([geometry['boxes'][n]['half_extent'] for n in obstacle_names], device=device)
        self.include = torch.ones(2, len(obstacle_names), dtype=torch.bool, device=device)
        for side, s in enumerate(('l', 'r')):
            for j, n in enumerate(obstacle_names):
                if n.startswith(s+'_') and n in arm_names:
                    self.include[side, j] = False
        self.neutral = torch.tensor([[.15, .55, 0., .8, 0., 0., 0.],
                                     [.15, -.55, 0., .8, 0., 0., 0.]], device=device)
        self.metrics = {}

    def points_in_torso(self, q):
        pos, rot = arm_frames(q)
        v = self.points.expand(*q.shape[:-2], 2, 4, 5, 3)
        return pos[..., None, :] + rotate(rot[..., None, :], v)

    def obstacle_frames(self, body_pos, body_quat, torso_id):
        torso_inv = inverse(body_quat[:, torso_id])
        q = quat_mul(torso_inv[:, None, :], body_quat[:, self.body_ids])
        p = rotate(torso_inv[:, None, :], body_pos[:, self.body_ids] - body_pos[:, torso_id, None, :])
        center = p + rotate(q, self.centers.expand_as(p))
        return center, q

    def clearance(self, q, centers, rotations):
        if self.geometry_mode == 'obb_v2':
            return self.box_clearance(q, centers, rotations)
        points = self.points_in_torso(q)  # N,C,2,4,5,3
        minimum = q.new_full(q.shape[:-1], torch.inf)  # N,C,2
        for j in range(len(self.body_ids)):
            local = rotate(inverse(rotations[:, j])[:, None, None, None, None, :],
                           points - centers[:, j, None, None, None, None, :])
            # Signed box distance: negative inside, Euclidean outside.
            d = local.abs() - self.half[j]
            signed = d.clamp(min=0).norm(dim=-1) + d.amax(-1).clamp(max=0)
            distance = (signed - self.radii).flatten(-2).amin(-1)
            distance = torch.where(self.include[:, j], distance, torch.inf)
            minimum = torch.minimum(minimum, distance)
        return minimum

    def box_clearance(self, q, centers, rotations, extra_margin=None):
        """Conservative separation of mesh-enclosing oriented boxes.

        Projection gaps along either box's face normals are lower bounds on
        Euclidean clearance. A gap >= margin certifies that separation; missing
        edge-cross axes can only reject a safe pair, never accept an unsafe one.
        Unlike a capsule, these boxes do not extend past the mesh's end faces.
        """
        pos, rot = arm_frames(q)
        a = pos + rotate(rot, self.arm_centers.expand_as(pos))
        axes = q.new_tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        minimum = q.new_full(q.shape[:-1], torch.inf)
        for j in range(len(self.body_ids)):
            bq = rotations[:,j,None,None,None,:].expand_as(rot)
            rel = quat_mul(inverse(rot), bq)
            # Columns of R are B's axes in A's local coordinates.
            R = rotate(rel[...,None,:], axes.expand(*rel.shape[:-1],3,3)).transpose(-1,-2)
            t = rotate(inverse(rot), centers[:,j,None,None,None,:]-a)
            ga = t.abs()-self.arm_half-(R.abs()*self.half[j]).sum(-1)
            tb = (R*t[...,None]).sum(-2)
            gb = tb.abs()-self.half[j]-(R.abs()*self.arm_half[...,None]).sum(-2)
            gap = torch.maximum(ga.amax(-1), gb.amax(-1)).amin(-1)
            if extra_margin is not None:
                gap = gap-extra_margin[:,j,None,None]
            minimum = torch.minimum(minimum, torch.where(self.include[:,j],gap,torch.inf))
        return minimum

    def box_clearance_batched(self,q,centers,rotations,extra_margin=None):
        """Same six-axis OBB bound, with obstacle axes evaluated in one batch."""
        pos,rot=arm_frames(q)
        a=pos+rotate(rot,self.arm_centers.expand_as(pos))
        ar=rot[...,None,:]
        br=rotations[:,None,None,None,:,:]
        rel=quat_mul(inverse(ar),br)
        axes=q.new_tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        R=rotate(rel[...,None,:],axes.expand(*rel.shape[:-1],3,3)).transpose(-1,-2)
        t=rotate(inverse(ar),centers[:,None,None,None,:,:]-a[...,None,:])
        ah=self.arm_half[None,None,:,:,None,:]
        bh=self.half[None,None,None,None,:,:]
        ga=t.abs()-ah-(R.abs()*bh[...,None,:]).sum(-1)
        tb=(R*t[...,None]).sum(-2)
        gb=tb.abs()-bh-(R.abs()*ah[...,None]).sum(-2)
        gap=torch.maximum(ga.amax(-1),gb.amax(-1)).amin(-2)
        if extra_margin is not None: gap=gap-extra_margin[:,None,None,:]
        return torch.where(self.include[None,None],gap,torch.inf).amin(-1)

    @torch.no_grad()
    def project(self, desired, previous, actual, body_pos, body_quat, torso_id):
        if self.geometry_mode != 'obb_v2':
            return self._project_all(desired,previous,actual,body_pos,body_quat,torso_id)[0]
        # Most verified references are already clear. Check those once; only
        # build the nine fallback candidates for environments actually blocked.
        shape=desired.shape
        d,p,a=(q.reshape(-1,2,7) for q in (desired,previous,actual))
        limits=self.limits.expand(len(d),-1,-1,-1)
        goals=d.clamp(limits[...,0],limits[...,1])
        result=p+(goals-p).clamp(-self.max_delta,self.max_delta)
        initial=result.clone()
        centers,rotations=self.obstacle_frames(body_pos,body_quat,torso_id)
        clearance=torch.minimum(self.clearance(result[:,None],centers,rotations),
                                self.clearance((.5*(a+result))[:,None],centers,rotations))[:,0]
        bad=(clearance<self.margin).any(-1).nonzero().flatten()
        no_safe=torch.zeros_like(clearance,dtype=torch.bool)
        if len(bad):
            fallback,selected,unsafe=self._project_all(d[bad],p[bad],a[bad],body_pos[bad],body_quat[bad],torso_id,limits[bad])
            result[bad]=fallback.reshape(-1,2,7);clearance[bad]=selected;no_safe[bad]=unsafe
        correction=(result-initial).abs()
        self.metrics={
            'ArmGuard/corrected_arm_fraction':(correction.amax(-1)>1e-4).float().mean(),
            'ArmGuard/target_correction_mae':correction.mean(),
            'ArmGuard/reference_error_mae':(result-d).abs().mean(),
            'ArmGuard/no_safe_candidate_fraction':no_safe.float().mean(),
            'ArmGuard/selected_clearance_mean':clearance.mean()}
        return result.reshape(shape)

    @torch.no_grad()
    def _project_all(self, desired, previous, actual, body_pos, body_quat, torso_id, limits=None):
        desired, previous, actual = (q.reshape(-1, 2, 7) for q in (desired, previous, actual))
        limits=self.limits if limits is None else limits
        low, high = limits[..., 0], limits[..., 1]
        candidates = [desired]
        for amount in (.2, .4, .7):
            candidate = desired.clone()
            candidate[..., 1] += desired.new_tensor([amount, -amount])
            candidates.append(candidate)
        for alpha in (.75, .5, .25, 0.):
            candidates.append(self.neutral + alpha * (desired-self.neutral))
        candidates.append(previous)
        goals = torch.stack(candidates, 1).clamp(low[:, None], high[:, None])
        next_q = previous[:, None] + (goals-previous[:, None]).clamp(-self.max_delta, self.max_delta)
        centers, rotations = self.obstacle_frames(body_pos, body_quat, torso_id)
        end_clearance = self.clearance(next_q, centers, rotations)
        midpoint = .5 * (actual[:, None] + next_q)
        clearance = torch.minimum(end_clearance, self.clearance(midpoint, centers, rotations))
        cost = (goals - desired[:, None]).square().mean(-1)
        safe = clearance >= self.margin
        best_safe = torch.where(safe, cost, torch.inf).argmin(1)
        # If moving obstacles or reset geometry leave no safe local candidate,
        # choose the largest clearance and log it; never silently label it safe.
        best = torch.where(safe.any(1), best_safe, end_clearance.argmax(1))
        gather = best[:, None, :, None].expand(-1, 1, 2, 7)
        result = next_q.gather(1, gather).squeeze(1)
        selected = clearance.gather(1, best[:, None]).squeeze(1)
        correction = (result-next_q[:, 0]).abs()
        self.metrics = {
            'ArmGuard/corrected_arm_fraction': (correction.amax(-1) > 1e-4).float().mean(),
            'ArmGuard/target_correction_mae': correction.mean(),
            'ArmGuard/reference_error_mae': (result-desired).abs().mean(),
            'ArmGuard/no_safe_candidate_fraction': (~safe.any(1)).float().mean(),
            'ArmGuard/selected_clearance_mean': selected.mean(),
        }
        return result.reshape(-1, 14),selected,~safe.any(1)
