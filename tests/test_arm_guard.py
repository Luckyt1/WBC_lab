import importlib
import torch
from test_tang_commands import commands

guard = importlib.import_module('tang_cpu.wbc.arm_guard')


def test_arm_forward_frames_match_zero_urdf_chain_and_pitch():
    q = torch.zeros(3, 2, 7)
    p, r = guard.arm_frames(q)
    torch.testing.assert_close(p[0, :, 0], torch.tensor([[0., .178, .087], [0., -.178, .087]]))
    torch.testing.assert_close(p[0, :, 1], torch.tensor([[.256, .178, -.169], [.256, -.178, -.169]]))
    torch.testing.assert_close(r[..., 0], torch.ones_like(r[..., 0]))
    q[..., 0] = torch.pi/2
    p, r = guard.arm_frames(q)
    torch.testing.assert_close(p[0, :, 1], torch.tensor([[-.256, .178, -.169], [-.256, -.178, -.169]]), atol=1e-6, rtol=1e-5)


def test_guard_projects_collision_and_preserves_limits_and_rate():
    names = [f'{s}_{j}_joint' for s in ('l', 'r') for j in
             ('shoulder_y', 'shoulder_x', 'shoulder_z', 'elbow_y', 'wrist_x', 'wrist_y', 'wrist_z')]
    limits = torch.tensor([-2., 2.]).expand(2, 14, 2).clone()
    g = guard.ArmCollisionGuard(names, ['torso_link'], limits, 'cpu')
    previous = g.neutral.flatten().expand(2, -1).clone()
    target = previous.clone()
    target[:, 1] = -.3
    target[:, 8] = .3
    pos = torch.zeros(2, 1, 3)
    quat = torch.tensor([1., 0., 0., 0.]).expand(2, 1, 4)
    for _ in range(40):
        q = g.project(target, previous, previous, pos, quat, 0)
        assert (q-previous).abs().max() <= g.max_delta + 1e-6
        assert q.abs().max() <= 2.
        assert torch.isfinite(q).all()
        previous = q
    centers, rot = g.obstacle_frames(pos, quat, 0)
    safe = g.clearance(q.reshape(2, 1, 2, 7), centers, rot)
    requested = g.clearance(target.reshape(2, 1, 2, 7), centers, rot)
    assert (safe > requested).all()
    assert all(torch.isfinite(v) for v in g.metrics.values())
