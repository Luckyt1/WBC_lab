"""Explicit teleoperation difficulty; elapsed updates never change a stage."""
from copy import deepcopy
from pathlib import Path

STAGES = ('quiet', 'arms_025', 'arms_050', 'arms_075', 'arms_100',
          'noise_050', 'noise_100', 'push_010', 'push_020', 'push_040', 'push_080')
NOISE_BASE = dict(lin_vel=.2, ang_vel=.2, projected_gravity=.05,
                  joint_pos=.01, joint_vel=1.5, height_scan=.1)


def settings(stage):
    i = STAGES.index(stage)
    return dict(arm_scale=(.05, .25, .5, .75, 1., 1., 1., 1., 1., 1., 1.)[i],
                noise_scale=(0., 0., 0., 0., 0., .5, 1., 1., 1., 1., 1.)[i],
                push_scale=(0., 0., 0., 0., 0., 0., 0., .125, .25, .5, 1.)[i],
                reset_scale=(0., 0., 0., 0., 0., 0., 0., .125, .25, .5, 1.)[i])


def overrides(stage):
    s = settings(stage)
    values = {'teleop_curriculum_stage': stage,
              'arm_motion.scale': s['arm_scale'],
              'arm_motion.amass_scale_end_iteration': None,
              'noise.add_noise': s['noise_scale'] > 0,
              # Remove the remaining age-based lateral-command expansion too.
              'commands.focus_lateral_initial_range': (.15, .3),
              'commands.focus_lateral_final_range': (.15, .3)}
    for key, value in NOISE_BASE.items():
        values['noise.noise_scales.'+key] = value * s['noise_scale']
    for name in ('push', 'reset'):
        for end in ('initial_scale', 'final_scale'):
            values[f'{name}_velocity_curriculum.{end}'] = s[name+'_scale']
    return values


def configure(cfg, stage):
    for path, value in overrides(stage).items():
        obj = cfg
        parts = path.split('.')
        for key in parts[:-1]:
            obj = getattr(obj, key)
        setattr(obj, parts[-1], value)


def checkpoint_stage(checkpoint):
    import yaml
    cfg = yaml.load((Path(checkpoint).parent/'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    stage = cfg.get('teleop_curriculum_stage')
    if stage not in STAGES:
        raise ValueError('Checkpoint is not part of the gated teleop curriculum')
    return stage


def expected_recipe(before, stage):
    """Allow only explicit difficulty fields; reward, observations and PPO stay exact."""
    previous = before.get('teleop_curriculum_stage')
    if previous not in STAGES or abs(STAGES.index(stage)-STAGES.index(previous)) > 1:
        raise ValueError('Curriculum changes must be adjacent, from a known curriculum')
    result = deepcopy(before)
    for path, value in overrides(stage).items():
        obj = result
        parts = path.split('.')
        for key in parts[:-1]:
            obj = obj[key]
        def serialized(v):
            if isinstance(v, (list, tuple)):
                return [serialized(x) for x in v]
            if v is None:
                return 'null'
            if isinstance(v, bool):
                return str(v).lower()
            return str(v)
        obj[parts[-1]] = serialized(value)
    return result


def evaluation_profiles(stage):
    # Retain both the immediately previous stage and the easiest learned task.
    index = STAGES.index(stage)
    stages = list(dict.fromkeys([stage, STAGES[max(0, index-1)], STAGES[0]]))
    return [dict(name='teleop_'+name, stage=name, arms=True, guard_fixed=True,
                 noise=settings(name)['noise_scale'] > 0,
                 push=settings(name)['push_scale'] > 0,
                 reset_velocity=settings(name)['reset_scale'] > 0, **settings(name))
            for name in stages]
