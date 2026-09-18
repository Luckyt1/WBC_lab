"""Original G1 flat WBC rewards, with ELF3 anatomical selectors.

Inherit functions, weights and scalar parameters from the original reward config.
Reward terms only change entity selection. Commands, curriculum, randomization
and simulation settings follow G1, with ELF3 controls and requested heights.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.envs.elf3.elf3_config import Elf3FlatEnvCfg
from legged_lab.envs.g1.g1_config import G1FlatAgentCfg, G1FlatEnvCfg, G1RewardCfg


@configclass
class Elf3WBCRewardCfg(G1RewardCfg):
    def __post_init__(self):
        feet = ["l_ankle_x_link", "r_ankle_x_link"]

        def bodies(term_name, parameter, names):
            entity = "contact_sensor" if parameter == "sensor_cfg" else "robot"
            getattr(self, term_name).params[parameter] = SceneEntityCfg(
                entity, body_names=names, preserve_order=True
            )

        def joints(term_name, names):
            getattr(self, term_name).params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=names, preserve_order=True
            )

        # Like G1: penalize every non-ankle body, including arms and torso.
        bodies("undesired_contacts", "sensor_cfg", "(?!.*ankle.*).*")
        for name in (
            "fly", "feet_air_time", "feet_contact_standing_stable",
            "feet_flat_standing", "feet_slide", "feet_force", "feet_stumble",
        ):
            bodies(name, "sensor_cfg", feet)
        for name in ("feet_flat_standing", "feet_slide", "feet_too_near", "center_of_gravity_tracking"):
            bodies(name, "asset_cfg", feet)
        for name in ("body_orientation_l2", "track_body_height_exp"):
            bodies(name, "asset_cfg", ["torso_link"])
        for name in ("track_body_roll_exp", "track_body_pitch_exp", "track_body_yaw_exp"):
            # These functions require [pelvis, torso]. ELF3's articulation root
            # is torso_link, so relying on PhysX body order reverses the frames.
            bodies(name, "asset_cfg", ["waist_z_link", "torso_link"])
        bodies("knee_stance_width", "asset_cfg", ["l_knee_y_link", "r_knee_y_link"])

        joints("dof_acc_l2", [".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", "waist_.*_joint"])
        joints("joint_deviation_hip", [".*_hip_z_joint", ".*_hip_x_joint"])
        joints("joint_deviation_waists", ["waist_.*_joint"])
        joints("joint_deviation_legs", [".*_hip_y_joint", ".*_knee_y_joint", ".*_ankle_.*"])
        joints("joint_torque_limits_waists", ["waist_.*_joint"])
        joints("joint_torque_limits_ankles", [".*_ankle_.*"])
        joints("joint_torque_limits_hip_pitch", ["l_hip_y_joint", "r_hip_y_joint"])


@configclass
class Elf3WBCEnvCfg(Elf3FlatEnvCfg):
    reward = Elf3WBCRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        reference = G1FlatEnvCfg()
        self.commands = reference.commands.copy()
        self.domain_rand = reference.domain_rand.copy()
        self.domain_rand.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["torso_link"]
        )
        self.scene.num_envs = reference.scene.num_envs
        self.scene.max_init_terrain_level = reference.scene.max_init_terrain_level
        # Elf3FlatEnvCfg uses the shared ELF3 asset template. Copy before changing
        # physics flags so existing ELF3/forward tasks keep their trained setup.
        self.scene.robot = self.scene.robot.copy()
        self.scene.robot.spawn.articulation_props = reference.scene.robot.spawn.articulation_props.copy()
        self.robot.terminate_contacts_body_names = ["torso_link"]
        self.robot.arm_motion_source = "amass_elf3"
        self.robot.arm_motion_timing = "legacy_frames"
        self.robot.arm_motion_ramp_s = 0.0
        self.robot.arm_motion_max_velocity = None
        self.commands.ranges.body_height = (0.6, 1.1)
        self.commands.curriculum.body_height.start_range = (0.9, 1.1)
        self.commands.curriculum.body_height.end_range = (0.6, 1.1)


@configclass
class Elf3WBCAgentCfg(G1FlatAgentCfg):
    experiment_name: str = "elf3_wbc"
    wandb_project: str = "elf3_wbc"
    logger: str = "tensorboard"
