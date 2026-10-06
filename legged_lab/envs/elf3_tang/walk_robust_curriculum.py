"""Locomotion-only commands and fixed acceptance scenarios (no simulator import)."""
from .commands import TangCommandCfg, VelocityRanges

SCENARIOS = [
    ('stand', [0., 0., 0., .8235, 0.]),
    ('forward_030', [.3, 0., 0., .8235, 0.]),
    ('forward_060', [.6, 0., 0., .8235, 0.]),
    ('backward_030', [-.3, 0., 0., .8235, 0.]),
    ('lateral_pos', [0., .2, 0., .8235, 0.]),
    ('lateral_neg', [0., -.2, 0., .8235, 0.]),
    ('turn_pos', [0., 0., .5, .8235, 0.]),
    ('turn_neg', [0., 0., -.5, .8235, 0.]),
]


def walk_commands():
    return TangCommandCfg(
        ranges=VelocityRanges(lin_vel_x=(-.6, .6), lin_vel_y=(-.3, .3), ang_vel_z=(-.6, .6)),
        rel_standing_envs=.15, rel_turning_envs=.15, rel_lateral_envs=.15,
        rel_crouch_yaw_envs=0., initial_mode_fractions=None,
        sagittal_speed_range=(.2, .6), turning_speed_range=(.3, .6),
        focus_start_iteration=0, focus_end_iteration=10000,
        focus_lateral_initial_range=(.1, .2), focus_lateral_final_range=(.15, .3),
        rel_nominal_height=1., rel_zero_waist_yaw=1., rel_neutral_standing=1.,
        height_initial_range=(.8235, .8235), height_range=(.8235, .8235),
        sampling_height_range=(.8235, .8235), waist_yaw_initial_range=(0., 0.),
        waist_yaw_range=(0., 0.), sampling_waist_yaw_range=(0., 0.),
        velocity_initial_scale=1., resampling_time_range=(10., 14.),
        focus_hold_time_range=(10., 14.))


def configure_walk_robust(cfg):
    cfg.commands = walk_commands()
    cfg.scratch_stage = None  # Independent task, never enters the old pose stages.
    cfg.noise.add_noise = True
    cfg.arm_motion.amass_initial_scale = .05
    cfg.arm_motion.amass_scale_start_iteration = 5000
    cfg.arm_motion.amass_scale_end_iteration = 20000
    cfg.arm_motion.scale = 1.
    cfg.arm_motion.amass_neutral_fraction = .3
    # 24 control steps/update, 0.02 seconds/control step.
    # Time-based difficulty schedule; evaluation reports do not imply promotion.
    for name, end in [('push_velocity_curriculum', 20000), ('reset_velocity_curriculum', 30000)]:
        c = getattr(cfg, name)
        c.initial_scale, c.final_scale = 0., 1.
        c.curriculum_duration_s = end * 24 * .02
