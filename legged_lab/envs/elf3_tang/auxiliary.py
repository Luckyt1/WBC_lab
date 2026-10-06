"""Simulator truth labels; these never enter deployment actor observations."""
from .commands import rotate_inverse


def capture_state(env):
    data = env.robot.data
    quat = data.body_quat_w[:, env.pelvis_body_id]
    return {
        'velocity': rotate_inverse(quat, data.body_lin_vel_w[:, env.pelvis_body_id]).detach().clone(),
        'contact': (env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids] > 0).float().clone(),
        'joint_velocity': data.joint_vel[:, env.policy_joint_ids].detach().clone(),
        'angular_velocity': rotate_inverse(quat, data.body_ang_vel_w[:, env.pelvis_body_id]).detach().clone(),
    }


def transition_targets(before, after, applied_action, done):
    return dict(velocity=before['velocity'], contact=before['contact'],
                applied_action=applied_action.detach().clone(),
                next_joint_velocity=after['joint_velocity'], next_angular_velocity=after['angular_velocity'],
                current_joint_velocity=before['joint_velocity'], current_angular_velocity=before['angular_velocity'],
                valid=(~done).detach().clone())
