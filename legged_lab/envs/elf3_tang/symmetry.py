"""Sagittal reflection for the BXI actor, critic and dynamics supervision.

All history frames and external arm states are reflected together. Polar
vectors use (x,-y,z); axial vectors use (-x,y,-z). Joint axes are verified
against the frozen BXI URDF by tests, not inferred from array positions.
"""
import torch


POLICY_JOINTS = tuple(['waist_y_joint', 'waist_x_joint', 'waist_z_joint'] + [
    f'{side}_{joint}_joint' for side in ('l', 'r')
    for joint in ('hip_y', 'hip_x', 'hip_z', 'knee_y', 'ankle_y', 'ankle_x')])
ARM_JOINTS = tuple(f'{side}_{joint}_joint' for side in ('l', 'r')
                  for joint in ('shoulder_y', 'shoulder_x', 'shoulder_z', 'elbow_y',
                                'wrist_x', 'wrist_y', 'wrist_z'))


def joint_map(names):
    peers = [('r_' + n[2:] if n.startswith('l_') else
              'l_' + n[2:] if n.startswith('r_') else n) for n in names]
    indices = [names.index(n) for n in peers]
    signs = [1. if n.endswith('_y_joint') else -1. for n in names]
    return indices, signs


def mirror_joints(value, names):
    if value.shape[-1] != len(names):
        raise ValueError('Unexpected joint vector dimension')
    indices, signs = joint_map(names)
    return value[..., indices] * value.new_tensor(signs)


def mirror_observations(obs, obs_type='policy'):
    width = {'policy': 84, 'critic': 91}[obs_type]
    if obs.ndim != 2 or obs.shape[-1] != width * 10:
        raise ValueError('ELF3 symmetry requires 10 frames of 84 actor / 91 critic observations')
    x = obs.reshape(-1, 10, width)
    y = x.clone()
    y[..., :3] = x[..., :3] * x.new_tensor([-1., 1., -1.])
    y[..., 3:6] = x[..., 3:6] * x.new_tensor([1., -1., 1.])
    y[..., 6:11] = x[..., 6:11] * x.new_tensor([1., -1., -1., 1., -1.])
    for start in (11, 26, 41):
        y[..., start:start+15] = mirror_joints(x[..., start:start+15], POLICY_JOINTS)
    for start in (56, 70):
        y[..., start:start+14] = mirror_joints(x[..., start:start+14], ARM_JOINTS)
    if obs_type == 'critic':
        y[..., 84:87] = x[..., 84:87] * x.new_tensor([1., -1., 1.])
        y[..., 87:89] = x[..., [88, 87]]
        y[..., 90] = -x[..., 90]  # measured waist yaw; height stays unchanged
    return y.reshape_as(obs)


def augment_observations_actions(obs, actions, env=None, obs_type='policy'):
    """RSL-RL convention: original samples, followed by their reflections."""
    return (None if obs is None else torch.cat((obs, mirror_observations(obs, obs_type))),
            None if actions is None else torch.cat((actions, mirror_joints(actions, POLICY_JOINTS))))


def mirror_auxiliary(targets):
    result = {}
    for name, value in targets.items():
        if name == 'velocity':
            result[name] = value * value.new_tensor([1., -1., 1.])
        elif name == 'contact':
            result[name] = value[..., [1, 0]]
        elif name == 'applied_action':
            result[name] = mirror_joints(value, POLICY_JOINTS + ARM_JOINTS)
        elif name in ('current_joint_velocity', 'next_joint_velocity'):
            result[name] = mirror_joints(value, POLICY_JOINTS)
        elif name in ('current_angular_velocity', 'next_angular_velocity'):
            result[name] = value * value.new_tensor([-1., 1., -1.])
        elif name == 'valid':
            result[name] = value.clone()
        else:
            raise ValueError(f'No reflection defined for auxiliary target {name}')
    return result


def augment_auxiliary(targets, env=None):
    mirrored = mirror_auxiliary(targets)
    return {k: torch.cat((v, mirrored[k])) for k, v in targets.items()}


def symmetry_config():
    return dict(use_data_augmentation=True, use_mirror_loss=True, mirror_loss_coeff=.1,
                data_augmentation_func='legged_lab.envs.elf3_tang.symmetry:augment_observations_actions',
                auxiliary_augmentation_func='legged_lab.envs.elf3_tang.symmetry:augment_auxiliary')


def validate_environment(env, empirical_normalization=False):
    """Fail before training if the live robot no longer matches this reflection."""
    if empirical_normalization:
        raise ValueError('ELF3 reflection requires fixed observation scaling')
    if (not env.cfg.observe_arm_state or env.cfg.scene.height_scanner.enable_height_scan
            or env.cfg.scene.depth_camera.enable_depth_camera):
        raise ValueError('ELF3 symmetry requires encoder-only 84/91-frame observations')
    data = env.robot.data
    for ids, names in ((env.policy_joint_ids, POLICY_JOINTS), (env.arm_joint_ids, ARM_JOINTS)):
        if tuple(data.joint_names[i] for i in ids) != names:
            raise ValueError('ELF3 symmetry joint layout mismatch')
        for attr in ('default_joint_pos', 'default_joint_vel'):
            value = getattr(data, attr)[:, ids]
            torch.testing.assert_close(value, mirror_joints(value, names), atol=1e-6, rtol=0)
    indices, _ = joint_map(POLICY_JOINTS)
    torch.testing.assert_close(env.action_scale, env.action_scale[indices])
    low, high = env.policy_action_low, env.policy_action_high
    a, b = mirror_joints(low, POLICY_JOINTS), mirror_joints(high, POLICY_JOINTS)
    torch.testing.assert_close(low, torch.minimum(a, b), atol=1e-5, rtol=0)
    torch.testing.assert_close(high, torch.maximum(a, b), atol=1e-5, rtol=0)
    # feet_cfg indexes the contact sensor, whose body ordering differs from the articulation.
    feet = tuple(env.contact_sensor.body_names[i] for i in env.feet_cfg.body_ids)
    if len(feet) != 2 or set(feet) != {'l_ankle_x_link', 'r_ankle_x_link'}:
        raise ValueError('Unexpected foot-contact order')
    actor, extras = env.get_observations()
    mirror_observations(actor)
    mirror_observations(extras['observations']['critic'], 'critic')
