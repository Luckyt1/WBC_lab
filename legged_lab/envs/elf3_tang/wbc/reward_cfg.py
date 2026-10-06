"""WBC elf3_g1_pelvis rewards, weights and anatomical selectors."""
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from . import rewards


@configclass
class WBCRewardCfg:
    track_lin_vel_xy_exp = RewTerm(func=rewards.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={'std': 0.5})
    track_ang_vel_z_exp = RewTerm(func=rewards.track_ang_vel_z_world_exp, weight=1.0, params={'std': 0.5})
    lin_vel_z_l2 = RewTerm(func=rewards.lin_vel_z_l2, weight=-1.0, params={})
    ang_vel_xy_l2 = RewTerm(func=rewards.ang_vel_xy_l2, weight=-0.1, params={})
    energy = RewTerm(func=rewards.energy, weight=-0.001, params={})
    dof_acc_l2 = RewTerm(func=rewards.joint_acc_l2, weight=-2.5e-07, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=['.*_hip_.*', '.*_knee_.*', '.*_ankle_.*', 'waist_.*_joint'], preserve_order=True)})
    action_rate_l2 = RewTerm(func=rewards.action_rate_l2, weight=-0.01, params={})
    undesired_contacts = RewTerm(func=rewards.undesired_contacts, weight=-1.0, params={'sensor_cfg': SceneEntityCfg(name='contact_sensor', body_names='(?!.*ankle.*).*', preserve_order=True), 'threshold': 1.0})
    fly = RewTerm(func=rewards.fly, weight=-1.0, params={'sensor_cfg': SceneEntityCfg(name='contact_sensor', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True), 'threshold': 1.0})
    body_orientation_l2 = RewTerm(func=rewards.body_orientation_l2, weight=0.0, params={'asset_cfg': SceneEntityCfg(name='robot', body_names=['torso_link'], preserve_order=True)})
    flat_orientation_l2 = RewTerm(func=rewards.flat_orientation_l2, weight=-2.0, params={})
    termination_penalty = RewTerm(func=rewards.is_terminated, weight=-200.0, params={})
    feet_air_time = RewTerm(func=rewards.feet_air_time_positive_biped, weight=0.15, params={'sensor_cfg': SceneEntityCfg(name='contact_sensor', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True), 'threshold': 0.4})
    feet_contact_standing_stable = RewTerm(func=rewards.feet_contact_standing_stable, weight=0.5, params={'sensor_cfg': SceneEntityCfg(name='contact_sensor', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True), 'min_contact_time': 0.5})
    feet_flat_standing = RewTerm(func=rewards.feet_flat_standing, weight=0.175, params={'asset_cfg': SceneEntityCfg(name='robot', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True), 'sensor_cfg': SceneEntityCfg(name='contact_sensor', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True), 'dead_zone_deg': 3.0, 'std_deg': 10.0, 'min_contact_time': 0.5, 'contact_ramp_s': 0.25, 'roll_dead_zone': 0.5, 'pitch_dead_zone': 1.1, 'roll_scale': 0.35, 'pitch_scale': 0.45})
    feet_slide = RewTerm(func=rewards.feet_slide, weight=-0.25, params={'sensor_cfg': SceneEntityCfg(name='contact_sensor', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True), 'asset_cfg': SceneEntityCfg(name='robot', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True)})
    feet_force = RewTerm(func=rewards.body_force, weight=-0.003, params={'sensor_cfg': SceneEntityCfg(name='contact_sensor', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True), 'threshold': 500, 'max_reward': 400})
    feet_too_near = RewTerm(func=rewards.feet_too_near_humanoid, weight=-2.0, params={'asset_cfg': SceneEntityCfg(name='robot', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True), 'threshold': 0.2})
    feet_stumble = RewTerm(func=rewards.feet_stumble, weight=-2.0, params={'sensor_cfg': SceneEntityCfg(name='contact_sensor', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True)})
    dof_pos_limits = RewTerm(func=rewards.joint_pos_limits, weight=-2.0, params={})
    joint_deviation_hip = RewTerm(func=rewards.joint_deviation_l1, weight=-0.5, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=['.*_hip_z_joint', '.*_hip_x_joint'], preserve_order=True)})
    joint_deviation_waists = RewTerm(func=rewards.joint_deviation_l1, weight=-0.2, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=['waist_.*_joint'], preserve_order=True)})
    joint_deviation_legs = RewTerm(func=rewards.joint_deviation_l1, weight=-0.02, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=['.*_hip_y_joint', '.*_knee_y_joint', '.*_ankle_.*'], preserve_order=True)})
    track_body_height_exp = RewTerm(func=rewards.track_body_height_exp, weight=1.0, params={'asset_cfg': SceneEntityCfg(name='robot', body_names=['waist_z_link'], preserve_order=True), 'std': 0.1})
    track_body_roll_exp = RewTerm(func=rewards.track_body_roll_exp, weight=1.0, params={'asset_cfg': SceneEntityCfg(name='robot', body_names=['waist_z_link', 'torso_link'], preserve_order=True), 'std': 0.1})
    track_body_pitch_exp = RewTerm(func=rewards.track_body_pitch_exp, weight=1.5, params={'asset_cfg': SceneEntityCfg(name='robot', body_names=['waist_z_link', 'torso_link'], preserve_order=True), 'std': 0.1})
    track_body_yaw_exp = RewTerm(func=rewards.track_body_yaw_exp, weight=1.0, params={'asset_cfg': SceneEntityCfg(name='robot', body_names=['waist_z_link', 'torso_link'], preserve_order=True), 'std': 0.1})
    center_of_gravity_tracking = RewTerm(func=rewards.center_of_gravity_tracking, weight=0.5, params={'asset_cfg': SceneEntityCfg(name='robot', body_names=['l_ankle_x_link', 'r_ankle_x_link'], preserve_order=True), 'sigma': 0.2})
    knee_stance_width = RewTerm(func=rewards.knee_stance_width, weight=0.5, params={'asset_cfg': SceneEntityCfg(name='robot', body_names=['l_knee_y_link', 'r_knee_y_link'], preserve_order=True), 'ideal_width': 0.27, 'std': 0.1})
    joint_torque_limits_waists = RewTerm(func=rewards.joint_torque_limits, weight=-100.0, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=['waist_.*_joint'], preserve_order=True), 'soft_torque_limit': 0.9})
    joint_torque_limits_ankles = RewTerm(func=rewards.joint_torque_limits, weight=-100.0, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=['.*_ankle_.*'], preserve_order=True), 'soft_torque_limit': 0.99})
    joint_torque_limits_hip_pitch = RewTerm(func=rewards.joint_torque_limits, weight=-100.0, params={'asset_cfg': SceneEntityCfg(name='robot', joint_names=['l_hip_y_joint', 'r_hip_y_joint'], preserve_order=True), 'soft_torque_limit': 0.99})
