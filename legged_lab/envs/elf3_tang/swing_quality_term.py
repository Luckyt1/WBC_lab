"""Isaac RewardManager adapter for the opt-in completed-swing experiment."""
import json
from pathlib import Path

import torch
from isaaclab.managers import ManagerTermBase

from .swing_quality import SwingTracker, sole_clearance


class CompletedSwingReward(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if env.cfg.scene.terrain_generator is not None:
            raise ValueError('CompletedSwingReward clearance currently requires the flat-plane task')
        names = ['l_ankle_x_link', 'r_ankle_x_link']
        self.body_ids, _ = env.robot.find_bodies(names, preserve_order=True)
        self.sensor_ids, _ = env.contact_sensor.find_bodies(names, preserve_order=True)
        geometry = Path(__file__).resolve().parents[2]/'assets/elf3_bxi_31/foot_collision_hulls.json'
        feet = json.loads(geometry.read_text())['feet']
        self.vertices = torch.tensor([feet[n]['vertices'] for n in names], device=env.device)
        self.tracker = SwingTracker(env.num_envs, env.device, env.step_dt, **cfg.params)

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)

    def __call__(self, env, min_air=.12, max_air=.35, min_clearance=.02, target_clearance=.03):
        data = env.robot.data
        contact = env.contact_sensor.data.current_contact_time[:, self.sensor_ids] > 0
        clearance = sole_clearance(data.body_pos_w[:, self.body_ids], data.body_quat_w[:, self.body_ids],
                                   self.vertices, env.scene.env_origins[:, 2])
        result = self.tracker.step(contact, clearance, ~env.command_generator.is_standing_env)
        metrics = self.tracker.last_metrics
        env.extras['log'].update({
            'SwingQuality/qualified_landings_per_env_step': metrics['qualified'].mean(),
            'SwingQuality/raw_landings_per_env_step': metrics['touchdowns'].mean(),
            'SwingQuality/mean_positive_sole_height': clearance.clamp(min=0).mean(),
        })
        return result
