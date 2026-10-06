"""Isaac RewardManager adapter; no position channel is added to the actor."""
from isaaclab.managers import ManagerTermBase

from .standing_position import StandingPositionTracker


class StandingPositionPenalty(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # RewardManager is constructed before env.init_buffers(): resolve the
        # pelvis through env.pelvis_body_id only when computing a transition.
        self.tracker = StandingPositionTracker(env.num_envs, env.device, env.step_dt, **cfg.params)

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)

    def __call__(self, env, settling_time=0.5, deadzone=0.01, distance_scale=0.1):
        standing = env.command_generator.is_standing_env
        xy = env.robot.data.body_pos_w[:, env.pelvis_body_id, :2]
        cost = self.tracker.step(xy, standing)
        count = self.tracker.active.sum().clamp(min=1)
        # Snapshot before partial resets; standing.transition_metrics merges this
        # with the other transition diagnostics before env.reset clears logs.
        env.standing_position_metrics = {
            'Standing/position_drift_mean_m': self.tracker.distance.sum() / count,
            'Standing/position_drift_max_m': self.tracker.distance.max(),
            'Standing/position_anchor_active_fraction': self.tracker.active.sum() / standing.sum().clamp(min=1),
            'Standing/position_drift_cost': cost.sum() / count,
            # Always emit the weighted contribution, including steps on which no
            # episode ends (episodic reward logs are only available on resets).
            'Standing/position_drift_reward_rate': cost.mean() * self.cfg.weight,
        }
        return cost
