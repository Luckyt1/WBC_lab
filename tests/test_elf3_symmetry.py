"""Physical reflection, transition alignment and joint PPO/auxiliary updates."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import pytest
import torch

from test_tang_commands import commands

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'rsl_rl')]
sym = importlib.import_module('tang_cpu.symmetry')
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from tools.run_teleop_curriculum import training_command


def peer(name):
    return 'r_'+name[2:] if name.startswith('l_') else 'l_'+name[2:] if name.startswith('r_') else name


def test_mirrored_joint_targets_produce_mirrored_urdf_forward_kinematics():
    asset = ROOT/'legged_lab/assets/elf3_bxi_31'
    p = json.loads((asset/'control_parameters.json').read_text())
    from legged_lab.paths import robot_asset_dir
    urdf = robot_asset_dir()/'source'/p['urdf_relative_path']
    if not urdf.is_file():
        pytest.skip('Optional external robot URDF is not installed; set ELF3_ASSET_DIR')
    root = ET.parse(urdf).getroot()
    joints = root.findall('joint')
    names = sym.POLICY_JOINTS+sym.ARM_JOINTS
    q = torch.randn(1, len(names), generator=torch.Generator().manual_seed(91), dtype=torch.float64)*.4
    mirrored = sym.mirror_joints(q, names)
    S = np.diag([1., -1., 1.])

    def fk(values):
        angles = dict(zip(names, values[0].tolist()))
        child_links = {j.find('child').get('link') for j in joints}
        base, = set(l.get('name') for l in root.findall('link'))-child_links
        transforms = {base: np.eye(4)}
        pending = list(joints)
        while pending:
            count = len(pending)
            for j in list(pending):
                parent, child = j.find('parent').get('link'), j.find('child').get('link')
                if parent not in transforms:
                    continue
                origin = j.find('origin')
                T = np.eye(4)
                T[:3,:3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' ')).as_matrix()
                T[:3, 3] = np.fromstring(origin.get('xyz'), sep=' ')
                if j.get('type') != 'fixed':
                    a = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                    angle = angles.get(j.get('name'), 0.)
                    K = np.array([[0.,-a[2],a[1]],[a[2],0.,-a[0]],[-a[1],a[0],0.]])
                    T[:3,:3] = T[:3,:3]@(np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K))
                transforms[child] = transforms[parent]@T
                pending.remove(j)
            assert len(pending) < count
        return transforms

    a, b = fk(q), fk(mirrored)
    for name, T in a.items():
        if name not in {n.replace('_joint', '_link') for n in names}:
            continue  # Fixed camera/IMU mounting frames are not motor control frames.
        U = b[peer(name)]
        np.testing.assert_allclose(U[:3, 3], S@T[:3, 3], atol=1e-10)
        np.testing.assert_allclose(U[:3, :3], S@T[:3, :3]@S, atol=1e-10)
    assert list(sym.POLICY_JOINTS) == p['policy_joint_names']
    assert list(sym.ARM_JOINTS) == p['arm_joint_names']
    defaults = torch.tensor([[p['joints'][n]['default_position'] for n in names]])
    torch.testing.assert_close(defaults, sym.mirror_joints(defaults, names))


def test_observation_reflection_preserves_time_height_and_swaps_complete_side_state():
    x = torch.arange(2*10*91, dtype=torch.float).reshape(2, 910)
    y = sym.mirror_observations(x, 'critic').reshape(2, 10, 91)
    frames = x.reshape(2, 10, 91)
    torch.testing.assert_close(y[..., [0,2,4,7,8,10,85,90]], -frames[..., [0,2,4,7,8,10,85,90]])
    torch.testing.assert_close(y[..., [1,3,5,6,9,84,86,89]], frames[..., [1,3,5,6,9,84,86,89]])
    torch.testing.assert_close(y[..., 87:89], frames[..., [88,87]])
    # Left hip pitch receives right hip pitch, left hip roll receives minus right.
    torch.testing.assert_close(y[..., 14], frames[..., 20])
    torch.testing.assert_close(y[..., 15], -frames[..., 21])
    # Both pose and velocity of the right shoulder must accompany its mirror.
    torch.testing.assert_close(y[..., 56], frames[..., 63])
    torch.testing.assert_close(y[..., 71], -frames[..., 78])
    torch.testing.assert_close(sym.mirror_observations(y.reshape_as(x), 'critic'), x)
    actor = frames[..., :84].reshape(2, 840)
    torch.testing.assert_close(sym.mirror_observations(actor), y[..., :84].reshape(2, 840))
    with pytest.raises(ValueError): sym.mirror_observations(torch.zeros(2, 560))


def targets(n):
    return dict(velocity=torch.randn(n,3), contact=torch.randint(0,2,(n,2)).float(),
        applied_action=torch.randn(n,29), current_joint_velocity=torch.randn(n,15),
        next_joint_velocity=torch.randn(n,15), current_angular_velocity=torch.randn(n,3),
        next_angular_velocity=torch.randn(n,3), valid=torch.arange(n)%2 == 0)


def test_auxiliary_reflection_uses_correct_vector_types_and_retains_terminal_mask():
    t = targets(4); u = sym.mirror_auxiliary(t)
    torch.testing.assert_close(u['velocity'], t['velocity']*torch.tensor([1.,-1.,1.]))
    torch.testing.assert_close(u['next_angular_velocity'], t['next_angular_velocity']*torch.tensor([-1.,1.,-1.]))
    torch.testing.assert_close(u['contact'], t['contact'][:, [1,0]])
    assert torch.equal(u['valid'], t['valid'])
    for k, v in sym.mirror_auxiliary(u).items(): torch.testing.assert_close(v, t[k])
    augmented = sym.augment_auxiliary(t)
    for k,v in t.items():
        torch.testing.assert_close(augmented[k][:4], v)
        torch.testing.assert_close(augmented[k][4:], u[k])


def build_ppo(augment=True, label_function=True):
    policy = ActorCritic(840,910,15,actor_hidden_dims=[32,16],critic_hidden_dims=[32,16],init_noise_std=.3)
    cfg = sym.symmetry_config()
    cfg.update(data_augmentation_func=sym.augment_observations_actions, _env=None,
               use_data_augmentation=augment)
    if label_function: cfg['auxiliary_augmentation_func'] = sym.augment_auxiliary
    else: cfg.pop('auxiliary_augmentation_func')
    return PPO(policy, num_learning_epochs=2, num_mini_batches=2, device='cpu', symmetry_cfg=cfg,
               auxiliary_cfg=dict(mode='dynamics',feature_dim=16,joint_dim=15,action_dim=29,
                                  warmup_updates=0,ramp_updates=1))


@pytest.mark.parametrize('augment',[False,True])
def test_joint_ppo_symmetry_auxiliary_update_and_shuffled_label_alignment(augment):
    ppo = build_ppo(augment); ppo.auxiliary.updates.fill_(1)
    ppo.init_storage('rl', 4, 3, [840], [910], [15])
    for step in range(3):
        obs = torch.randn(4,840)*.1
        t = targets(4)
        # Height slot is invariant; gravity-y changes sign like linear y velocity.
        obs[:,9] = t['velocity'][:,0]
        obs[:,4] = t['velocity'][:,1]
        ppo.act(obs,torch.randn(4,910)*.1)
        ppo.process_env_step(torch.randn(4),torch.zeros(4,dtype=torch.bool),{'auxiliary':t})
    ppo.compute_returns(torch.randn(4,910)*.1)
    original = ppo.auxiliary.loss
    calls = []
    def checked(policy, obs, target):
        torch.testing.assert_close(obs[:,9], target['velocity'][:,0])
        torch.testing.assert_close(obs[:,4], target['velocity'][:,1])
        calls.append(len(obs))
        return original(policy, obs, target)
    ppo.auxiliary.loss = checked
    before = deepcopy(ppo.policy.state_dict())
    logs = ppo.update()
    assert calls and set(calls) == {12 if augment else 6}
    assert logs['symmetry'] > 0 and logs['aux_total'] > 0
    assert ppo.auxiliary.updates.item() == 2
    assert all(torch.isfinite(p).all() for p in ppo.policy.parameters())
    assert not torch.equal(before['actor.0.weight'], ppo.policy.state_dict()['actor.0.weight'])
    assert ppo.auxiliary_optimizer.state


def test_cannot_augment_observations_without_auxiliary_labels():
    with pytest.raises(ValueError, match='label augmentation'):
        build_ppo(label_function=False)


def test_curriculum_continuation_carries_symmetry_and_height():
    for checkpoint in (None, Path('/run/model_5000.pt')):
        argv = training_command('python', '/venv/python', '/logs/run', 'quiet', 1000, 4096,
                                .8, checkpoint=checkpoint, symmetry=True)
        assert '--elf3_symmetry' in argv
        assert argv[argv.index('--pelvis_height')+1] == '0.8'
