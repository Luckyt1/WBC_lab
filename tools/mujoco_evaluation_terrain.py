"""Explicit terrain selection for ELF3 CPU evaluations.

The deployment XML can include stairs. Flat-ground policy comparisons must not
silently inherit those obstacles. Only collision masks of world obstacles change;
robot, ground material, actuator and solver parameters remain unchanged.
"""
import mujoco
import numpy as np


def configure_terrain(model, mode='flat'):
    if mode not in ('flat', 'source'):
        raise ValueError(f'Unknown evaluation terrain: {mode}')
    world = np.flatnonzero(model.geom_bodyid == 0)
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    planes = world[(model.geom_type[world] == mujoco.mjtGeom.mjGEOM_PLANE) & active[world]]
    obstacles = world[(model.geom_type[world] != mujoco.mjtGeom.mjGEOM_PLANE) & active[world]]
    if mode == 'flat':
        if len(planes) != 1:
            raise ValueError('Flat evaluation requires exactly one existing collidable world plane')
        plane = planes[0]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[plane])
        if (not np.allclose(rotation.reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-8)
                or not np.isclose(model.geom_pos[plane, 2], 0., atol=1e-8)):
            raise ValueError('Flat evaluation requires a horizontal ground plane at z=0')
        model.geom_contype[obstacles] = 0
        model.geom_conaffinity[obstacles] = 0
    describe = lambda i: dict(id=int(i), name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)),
                              type=int(model.geom_type[i]), position=model.geom_pos[i].tolist(),
                              size=model.geom_size[i].tolist(), friction=model.geom_friction[i].tolist(),
                              collision_enabled=bool(model.geom_contype[i] or model.geom_conaffinity[i]))
    return dict(mode=mode, protocol_version=2,
                disabled_obstacles=[describe(i)['name'] for i in obstacles] if mode == 'flat' else [],
                world_geometry=[describe(i) for i in world])


def first_obstacle_contact(model, data, minimum_force=0.5):
    """Return a loaded non-plane world contact at this policy sample, if present."""
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = (int(contact.geom1), int(contact.geom2))
        for geom in pair:
            if model.geom_bodyid[geom] != 0 or model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            magnitude = float(np.linalg.norm(force[:3]))
            if magnitude > minimum_force:
                return dict(time_s=float(data.time),
                            obstacle=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                            force_norm_N=magnitude, position=contact.pos.tolist())
    return None
