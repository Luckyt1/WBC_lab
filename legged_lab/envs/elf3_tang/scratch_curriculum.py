"""Explicit scratch stages. Training age alone never promotes a stage."""
from .commands import TangCommandCfg, VelocityRanges

STAGES = ('locomotion', 'pose', 'combined', 'push_01', 'push_02', 'push_04', 'push_08')


def stage_settings(stage):
    if stage not in STAGES:
        raise ValueError(f'Unknown scratch stage: {stage}')
    index = STAGES.index(stage)
    push = (0., 0., 0., .125, .25, .5, 1.)[index]
    return dict(arm_scale=(.05, .25, .5, 1., 1., 1., 1.)[index],
                neutral_fraction=.3 if index < 3 else 0.,
                push_scale=push, reset_scale=push, noise=index > 0)


def stage_commands(stage):
    stage_settings(stage)
    common = dict(initial_mode_fractions=None, velocity_initial_scale=1.,
                  resampling_time_range=(10., 14.), focus_hold_time_range=(10., 14.))
    if stage == 'locomotion':
        return TangCommandCfg(
            **common, rel_standing_envs=.25, rel_turning_envs=.25,
            sagittal_speed_range=(.2, .4), turning_speed_range=(.3, .6),
            rel_nominal_height=1., rel_zero_waist_yaw=1., rel_neutral_standing=1.,
            sampling_height_range=(.8235, .8235), sampling_waist_yaw_range=(0., 0.))
    if stage == 'pose':
        return TangCommandCfg(
            **common, rel_standing_envs=.45, rel_turning_envs=.15, rel_lateral_envs=.2,
            sagittal_speed_range=(.2, .4), turning_speed_range=(.3, .6),
            isolate_standing_pose=True, rel_neutral_standing=.25,
            rel_nominal_height=0., rel_zero_waist_yaw=0.,
            sampling_height_range=(.65, .8235), sampling_waist_yaw_range=(-.6, .6),
            focus_lateral_initial_range=(.15, .25), focus_lateral_final_range=(.15, .25))
    return TangCommandCfg(
        **common, ranges=VelocityRanges(), rel_standing_envs=.25, rel_turning_envs=.15,
        rel_lateral_envs=.15, rel_crouch_yaw_envs=.15, rel_neutral_standing=.7,
        rel_nominal_height=.5, height_initial_range=(.5, .8235),
        waist_yaw_initial_range=(-.6, .6),
        focus_lateral_initial_range=(.15, .3), focus_lateral_final_range=(.15, .3),
        focus_vx_initial_range=(.2, .3), focus_vx_final_range=(.2, .3),
        focus_height_initial_range=(.5, .65), focus_height_final_range=(.5, .65),
        focus_yaw_initial_range=(.4, .6), focus_yaw_final_range=(.4, .6))


def configure_scratch_stage(cfg, stage):
    settings = stage_settings(stage)
    cfg.scratch_stage = stage
    cfg.commands = stage_commands(stage)
    cfg.arm_motion.scale = settings['arm_scale']
    cfg.arm_motion.amass_scale_end_iteration = None
    cfg.arm_motion.amass_neutral_fraction = settings['neutral_fraction']
    cfg.noise.add_noise = settings['noise']
    for name, value in (('push_velocity_curriculum', settings['push_scale']),
                        ('reset_velocity_curriculum', settings['reset_scale'])):
        curriculum = getattr(cfg, name)
        curriculum.initial_scale = curriculum.final_scale = value
