"""Regression: a flat locomotion evaluation must not collide with XML stairs."""
import sys
from pathlib import Path
import numpy as np
import pytest
mujoco = pytest.importorskip('mujoco')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from mujoco_evaluation_terrain import configure_terrain, first_obstacle_contact

XML = '''<mujoco><worldbody>
<geom name="ground" type="plane" size="0 0 .01"/>
<geom name="stair" type="box" pos="0 0 .1" size=".4 .4 .1"/>
<body pos="0 0 .7"><freejoint/><geom name="robot" type="sphere" size=".1" mass="1"/></body>
</worldbody></mujoco>'''


def simulate(mode):
    model = mujoco.MjModel.from_xml_string(XML)
    mass = model.body_mass.copy()
    friction = model.geom_friction.copy()
    metadata = configure_terrain(model, mode)
    np.testing.assert_array_equal(model.body_mass, mass)
    np.testing.assert_array_equal(model.geom_friction, friction)
    data = mujoco.MjData(model)
    hit = None
    for _ in range(1000):
        mujoco.mj_step(model, data)
        hit = hit or first_obstacle_contact(model, data)
    return float(data.qpos[2]), hit, metadata


def test_flat_removes_stair_collision_but_keeps_ground_and_robot():
    flat_z, flat_hit, flat = simulate('flat')
    source_z, source_hit, source = simulate('source')
    assert flat_z == pytest.approx(.1, abs=.002)
    assert source_z == pytest.approx(.3, abs=.002)
    assert flat_hit is None and source_hit['obstacle'] == 'stair'
    assert flat['disabled_obstacles'] == ['stair']
    assert source['disabled_obstacles'] == []


def test_missing_plane_does_not_silently_create_groundless_evaluation():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" size="1 1 .1"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match='exactly one'):
        configure_terrain(model, 'flat')


def test_raised_plane_is_not_reported_as_zero_height_flat_ground():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" pos="0 0 .5" size="0 0 .01"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match='z=0'):
        configure_terrain(model, 'flat')
