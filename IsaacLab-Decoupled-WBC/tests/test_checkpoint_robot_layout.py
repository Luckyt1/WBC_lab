from types import SimpleNamespace

import pytest
import yaml

from legged_lab.utils.cli_args import apply_trained_actuator_gains, apply_trained_command_config, validate_trained_robot_layout


def _config(names, base, usd="elf3.usd"):
    return SimpleNamespace(
        robot=SimpleNamespace(action_joint_names=names, base_body_name=base),
        scene=SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(usd_path=usd))),
    )


def _save(tmp_path, robot, usd="elf3.usd"):
    (tmp_path / "params").mkdir(exist_ok=True)
    data = {"robot": robot, "scene": {"robot": {"spawn": {"usd_path": usd}}}}
    (tmp_path / "params" / "env.yaml").write_text(yaml.safe_dump(data))


def test_same_action_dimension_different_robot_is_rejected(tmp_path):
    _save(tmp_path, {"action_joint_names": ["g1_hip", "g1_knee"], "base_body_name": None})
    with pytest.raises(ValueError, match="joint order"):
        validate_trained_robot_layout(_config(["elf3_hip", "elf3_knee"], "waist_z_link"), str(tmp_path))


def test_matching_robot_but_different_base_frame_is_rejected(tmp_path):
    _save(tmp_path, {"action_joint_names": ["hip", "knee"], "base_body_name": "torso_link"})
    with pytest.raises(ValueError, match="observation frame"):
        validate_trained_robot_layout(_config(["hip", "knee"], "waist_z_link"), str(tmp_path))


def test_matching_layout_loads(tmp_path):
    _save(tmp_path, {"action_joint_names": ["hip", "knee"], "base_body_name": "waist_z_link"})
    saved = validate_trained_robot_layout(_config(["hip", "knee"], "waist_z_link"), str(tmp_path))
    assert saved["robot"]["action_joint_names"] == ["hip", "knee"]


@pytest.mark.parametrize("specialists", [False, True])
def test_saved_command_sampling_preserves_specialists_and_disables_them_for_older_runs(tmp_path, specialists):
    (tmp_path / 'params').mkdir()
    saved = dict(resampling_time_range=[10., 10.], rel_standing_envs=.4, rel_heading_envs=.5,
                 heading_command=True, heading_control_stiffness=.5, rel_in_place_turn_envs=.2,
                 in_place_small_turn_fraction=.7, rel_zero_pose_envs=.2, rel_single_axis_pose_envs=.3,
                 ranges={key:[-.5,.5] for key in ('lin_vel_x','lin_vel_y','ang_vel_z','heading',
                                                  'body_height','body_roll','body_pitch','body_yaw')})
    fields = dict(rel_high_stance_turn_envs=.1, rel_high_stance_move_turn_envs=.1,
                  high_stance_height_range=(.78,.8235), high_stance_speed_range=(.3,.6),
                  high_stance_yaw_rate_range=(.2,.6), rel_low_stance_move_envs=.1,
                  low_stance_height_range=(.4735,.6), low_stance_speed_range=(.2,.4),
                  low_stance_yaw_rate_range=(.2,.6), low_stance_turn_fraction=.6)
    if specialists:
        saved.update(fields)
    (tmp_path / 'params/env.yaml').write_text(yaml.safe_dump({'commands':saved}))
    cfg = SimpleNamespace(commands=SimpleNamespace(
        **{k:None for k in saved if k != 'ranges' and k not in fields},
        **fields, ranges=SimpleNamespace(**{k:None for k in saved['ranges']})))
    apply_trained_command_config(cfg, str(tmp_path), strict=True)
    assert cfg.commands.rel_high_stance_turn_envs == (.1 if specialists else 0.)
    assert cfg.commands.rel_high_stance_move_turn_envs == (.1 if specialists else 0.)
    assert cfg.commands.in_place_small_turn_fraction == .7
    assert cfg.commands.high_stance_height_range == (.78,.8235)
    assert cfg.commands.rel_low_stance_move_envs == (.1 if specialists else 0.)
    assert cfg.commands.low_stance_turn_fraction == (.6 if specialists else .5)


@pytest.mark.parametrize("saved_body,current_body,accepted", [
    ("torso_link", ["waist_z_link"], False),
    (["waist_z_link"], "torso_link", False),
    (None, ["waist_z_link"], False),
    (["waist_z_link"], "waist_z_link", True),
    ("torso_link", ("torso_link",), True),
    (None, ["torso_link"], True),
])
def test_height_command_reference_is_part_of_checkpoint_compatibility(
    tmp_path, saved_body, current_body, accepted,
):
    _save(tmp_path, {"action_joint_names": ["hip", "knee"], "base_body_name": "waist_z_link"})
    path = tmp_path / "params" / "env.yaml"
    data = yaml.safe_load(path.read_text())
    if saved_body is not None:
        data["reward"] = {"track_body_height_exp": {"params": {
            "asset_cfg": {"body_names": saved_body},
        }}}
    path.write_text(yaml.safe_dump(data))
    cfg = _config(["hip", "knee"], "waist_z_link")
    cfg.reward = SimpleNamespace(track_body_height_exp=SimpleNamespace(
        params={"asset_cfg": SimpleNamespace(body_names=current_body)},
    ))
    if accepted:
        validate_trained_robot_layout(cfg, str(tmp_path))
    else:
        with pytest.raises(ValueError, match="Height-command reference"):
            validate_trained_robot_layout(cfg, str(tmp_path))


def test_old_checkpoint_cannot_restore_removed_joint_target_clipping(tmp_path, capsys):
    _save(tmp_path, {
        "action_joint_names": ["hip", "knee"], "base_body_name": "waist_z_link",
        "clip_joint_targets": True, "action_scale": 0.213,
    })
    cfg = _config(["hip", "knee"], "waist_z_link")
    apply_trained_actuator_gains(cfg, str(tmp_path))
    assert not hasattr(cfg.robot, "clip_joint_targets")
    assert cfg.robot.action_scale == 0.213
    assert "saved clip_joint_targets=True is ignored" in capsys.readouterr().out


def test_same_policy_layout_cannot_resume_an_older_physical_asset(tmp_path):
    robot = {"action_joint_names": ["hip", "knee"], "base_body_name": "waist_z_link"}
    cfg = _config(["hip", "knee"], "waist_z_link")
    cfg.robot.asset_revision = "elf3_dof31_urdf_6ede896d8718"
    _save(tmp_path, robot)
    with pytest.raises(ValueError, match="physical asset revision"):
        validate_trained_robot_layout(cfg, str(tmp_path))
    robot["asset_revision"] = cfg.robot.asset_revision
    _save(tmp_path, robot)
    assert validate_trained_robot_layout(cfg, str(tmp_path))["robot"] == robot


def test_legacy_g1_only_loads_for_g1(tmp_path):
    _save(tmp_path, {}, usd="/old/path/g1.usd")
    validate_trained_robot_layout(_config(["g1_hip"], None, "g1.usd"), str(tmp_path))
    with pytest.raises(ValueError, match="lacks a verifiable"):
        validate_trained_robot_layout(_config(["elf3_hip"], "waist_z_link"), str(tmp_path))


@pytest.mark.parametrize("source", ["default", "amass_elf3"])
@pytest.mark.parametrize("ramp,rate", [(2.0, 1.0), (0.0, None)])
def test_play_restores_saved_arm_motion_settings(tmp_path, source, ramp, rate):
    robot = {
        "action_joint_names": ["hip"], "base_body_name": "waist_z_link",
        "arm_motion_source": source, "arm_motion_scale": 0.4,
        "arm_motion_ramp_s": ramp, "arm_motion_max_velocity": rate,
    }
    _save(tmp_path, robot)
    cfg = _config(["hip"], "waist_z_link")
    # No actuators in this small fixture; arm fields are restored before that check.
    apply_trained_actuator_gains(cfg, str(tmp_path))
    for key, value in robot.items():
        assert getattr(cfg.robot, key) == value
    if source == "amass_elf3":
        assert cfg.robot.arm_motion_timing == "source_fps"


def test_play_restores_physics_commands_and_randomization(tmp_path):
    robot = {"action_joint_names": ["hip"], "base_body_name": "waist_z_link",
             "arm_motion_source": "amass_elf3", "arm_motion_timing": "legacy_frames"}
    _save(tmp_path, robot)
    path = tmp_path / "params" / "env.yaml"
    saved = yaml.safe_load(path.read_text())
    saved["scene"]["robot"]["spawn"]["articulation_props"] = {
        "enabled_self_collisions": True, "solver_position_iteration_count": 8,
        "solver_velocity_iteration_count": 4,
    }
    saved["commands"] = {"ranges": {"body_height": [0.95, 1.12], "body_pitch": [-0.35, 0.35]}}
    saved["domain_rand"] = {"events": {
        "reset_base": {"params": {"pose_range": {"yaw": [-0.1, 0.1]}, "asset_cfg": {"name": "saved"}}},
        "push_robot": None,
    }}
    path.write_text(yaml.safe_dump(saved))
    cfg = _config(["hip"], "waist_z_link")
    cfg.scene.robot.spawn.articulation_props = SimpleNamespace(enabled_self_collisions=False)
    cfg.commands = SimpleNamespace(ranges=SimpleNamespace(body_height=(0.6, 1.1)))
    mapped_entity = object()
    cfg.domain_rand = SimpleNamespace(events=SimpleNamespace(
        reset_base=SimpleNamespace(params={"asset_cfg": mapped_entity}), push_robot=object()
    ))
    apply_trained_actuator_gains(cfg, str(tmp_path))
    assert cfg.robot.arm_motion_timing == "legacy_frames"
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is True
    assert cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count == 8
    assert cfg.commands.ranges.body_height == [0.95, 1.12]
    assert cfg.domain_rand.events.reset_base.params["pose_range"] == {"yaw": [-0.1, 0.1]}
    assert cfg.domain_rand.events.reset_base.params["asset_cfg"] is mapped_entity
    assert cfg.domain_rand.events.push_robot is None
