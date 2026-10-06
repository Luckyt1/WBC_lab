"""Ensure experimental reward overrides leave the retained recipe isolated."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('gait_ablation',
    Path(__file__).resolve().parents[1]/'legged_lab/envs/elf3_tang/gait_ablation.py')
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def recipe():
    return NS(reward=NS(feet_slide=NS(weight=-.25, params={'contact': 'unchanged'}),
                        feet_air_time=NS(weight=.15, params={'threshold': .4, 'use_standing_mask': True}),
                        standing_quiet_base=NS(weight=4.)),
              commands={'standing': .35, 'turning': .15, 'lateral': .15},
              noise={'joint_vel': 1.5}, arm_motion={'neutral_fraction': .5})


@pytest.mark.parametrize('variant,changed', [('control', None), ('slide', 'feet_slide'), ('air_time', 'feet_air_time')])
def test_one_factor_only_and_no_shared_recipe_mutation(variant, changed):
    original = recipe()
    cfg = deepcopy(original)
    info = ablation.configure_gait_ablation(cfg, variant)
    if changed:
        assert getattr(cfg.reward, changed).weight != getattr(original.reward, changed).weight
        getattr(cfg.reward, changed).weight = getattr(original.reward, changed).weight
    assert cfg == original
    assert info['before'] == {'feet_slide': -.25, 'feet_air_time': .15}


def test_changed_baseline_is_rejected_without_modification():
    cfg = recipe()
    cfg.reward.feet_slide.weight = -.7
    old = deepcopy(cfg)
    with pytest.raises(ValueError, match='Unexpected baseline'):
        ablation.configure_gait_ablation(cfg, 'slide')
    assert cfg == old
