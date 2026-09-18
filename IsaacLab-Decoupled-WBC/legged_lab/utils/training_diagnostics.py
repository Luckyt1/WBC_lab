"""Read-only rollout/PPO diagnostics for the installed feed-forward RSL-RL PPO.

No extra policy sampling, reward computation, optimizer steps or simulator writes.
Hooks observe the existing mini-batches and gradients; they never change tensors.
"""

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path

import torch


@dataclass
class DiagnosticsConfig:
    pre_steps: int = 16
    post_steps: int = 16
    max_events: int = 50
    envs_per_event: int = 2
    cooldown_iterations: int = 100
    action_threshold: float = 20.0
    torque_ratio_threshold: float = 3.0
    negative_reward_threshold: float = 10.0
    value_loss_threshold: float = 10.0
    kl_threshold: float = 0.1
    rolling_retention: bool = False

    def __post_init__(self):
        if self.rolling_retention and self.max_events < 12:
            raise ValueError('Rolling diagnostics need at least 12 slots for latest/severest of six categories')
        for name in ("pre_steps", "post_steps", "max_events", "envs_per_event", "cooldown_iterations"):
            if getattr(self, name) < 1:
                raise ValueError(f"diagnostics {name} must be positive")
        for name in ("action_threshold", "torque_ratio_threshold", "negative_reward_threshold", "value_loss_threshold", "kl_threshold"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"diagnostics {name} must be finite and positive")


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(v) for v in value]
    return value


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def gaussian_kl(old_mean, old_std, mean, std):
    """Analytic KL(old || new) per observation, summed over action dimensions."""
    return (torch.log(std / old_std) +
            (old_std.square() + (old_mean - mean).square()) / (2 * std.square()) - .5).sum(-1)


class TrainingDiagnostics:
    def __init__(self, runner, config=None):
        self.runner, self.env, self.alg = runner, runner.env, runner.alg
        self.policy = self.alg.policy
        self.config = config or DiagnosticsConfig()
        if self.policy.is_recurrent or self.alg.symmetry or self.alg.rnd or runner.is_distributed:
            raise ValueError("Training diagnostics currently support single-GPU feed-forward PPO without RND/symmetry")
        if not hasattr(self.env.reward_manager, "_step_reward"):
            raise ValueError("Unsupported RewardManager: per-step reward buffer unavailable")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.directory = Path(runner.log_dir) / "diagnostics" / stamp
        self.directory.mkdir(parents=True, exist_ok=False)
        self.iteration = runner.current_learning_iteration
        self.step = 0
        self.rollout_step = 0
        self.history = deque(maxlen=self.config.pre_steps + 1)
        self.pending = []
        self.saved_events = 0
        self.last_event_iteration = -self.config.cooldown_iterations
        self.category_last = {}
        self.retained = []
        self.last_push_step = torch.full((self.env.num_envs,), -1, device=self.env.device, dtype=torch.long)
        self.last_reset_step = torch.full_like(self.last_push_step, -1)
        self.push_due = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        self.policy_ids = list(self.env.custom_joint_ids[:self.env.num_actions])
        self.joint_names = list(self.env.robot.joint_names)
        self.policy_joint_names = list(self.env.action_joint_names)
        # ContactSensor and Articulation may enumerate the same bodies differently.
        self.feet_sensor_ids = list(self.env.feet_cfg.body_ids)
        self.feet_body_names = [self.env.contact_sensor.body_names[i] for i in self.feet_sensor_ids]
        self.feet_body_ids = [self.env.robot.body_names.index(name) for name in self.feet_body_names]
        feature_names = ([f"base_angular_velocity/{axis}" for axis in "xyz"] +
                         [f"projected_gravity/{axis}" for axis in "xyz"] +
                         [f"command/{name}" for name in ("vx", "vy", "yaw_rate", "height", "roll", "pitch", "yaw")] +
                         [f"{kind}/{name}" for kind in ("joint_position_error", "joint_velocity", "previous_action")
                          for name in self.policy_joint_names] + ["left_foot_contact", "right_foot_contact"])
        obs_dim = self.alg.storage.observations.shape[-1]
        self.observation_names = feature_names if len(feature_names) == obs_dim else [f"obs_{i}" for i in range(obs_dim)]
        self.pre = None
        self.collecting_update = False
        self.batch = None
        self.update_rows = []
        self.update_samples = []
        self.rollout_rows = []
        self.joint_maxima = []
        self._closed = False
        self._metadata = {
            "schema_version": 2, "config": asdict(self.config),
            "run_directory": str(Path(runner.log_dir).resolve()),
            "policy_class": type(self.policy).__name__,
            "start_iteration": self.iteration, "step_dt_s": self.env.step_dt,
            "num_steps_per_env": runner.num_steps_per_env,
            "joint_names": self.joint_names, "policy_joint_names": self.policy_joint_names,
            "policy_joint_ids": self.policy_ids,
            "feet_body_names": self.feet_body_names,
            "feet_body_ids": self.feet_body_ids,
            "feet_sensor_ids": self.feet_sensor_ids,
            "torque_semantics": "For implicit actuators, computed/applied torque are PD estimates before/after effort clipping, not measured PhysX motor torque. They precede the final physics substep; post joint state follows it.",
            "actor_observation_names": self.observation_names,
            "reward_names": list(self.env.reward_manager.active_terms),
            "reward_terms_units": "weighted per-second terms; multiply by step_dt_s to get step reward",
            "command_names": ["vx", "vy", "yaw_rate", "height", "body_roll", "body_pitch", "body_yaw"],
            "observation_semantics": "actor/critic tensors are the exact inputs passed to PPO.act, including normalization",
            "state_semantics": "pre is before action application; post is after interval events and reward, before reset",
            "frame_step_semantics": "session-local control step; iteration identifies the policy that collected it",
            "model_semantics": "event policy_before collected the trigger rollout; policy_after follows its PPO update",
            "replay_limit": "Saved observations support network replay, not an exact replay of simulator/RNG state",
            "training_settings": _json_value(runner.cfg),
            "thresholds_change_training": False,
        }
        self._write_json(self.directory / "metadata.json", self._metadata)
        self._install_hooks()
        self.env.training_diagnostics = self
        print(f"[Diagnostics] Enabled; event records: {self.directory}", flush=True)

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def _append_json(self, name, value):
        with (self.directory / name).open("a") as stream:
            stream.write(json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False) + "\n")

    def _state(self):
        env, data = self.env, self.env.robot.data
        base = env.base_body_id
        torso_id = env.robot.body_names.index("torso_link") if "torso_link" in env.robot.body_names else None
        return {
            "joint_position": data.joint_pos.clone(), "joint_velocity": data.joint_vel.clone(),
            "base_position_w": (data.root_pos_w if base is None else data.body_pos_w[:, base]).clone(),
            "base_quaternion_w": env.base_quat_w.clone(),
            "base_linear_velocity_w": env.base_lin_vel_w.clone(),
            "base_angular_velocity_w": env.base_ang_vel_w.clone(),
            "torso_position_w": (data.root_pos_w if torso_id is None else data.body_pos_w[:, torso_id]).clone(),
            "torso_quaternion_w": (data.root_quat_w if torso_id is None else data.body_quat_w[:, torso_id]).clone(),
            "feet_position_w": data.body_pos_w[:, self.feet_body_ids].clone(),
            "feet_contact_force_w": env.contact_sensor.data.net_forces_w[:, self.feet_sensor_ids].clone(),
            "feet_contact_time_s": env.contact_sensor.data.current_contact_time[:, self.feet_sensor_ids].clone(),
            "episode_step": env.episode_length_buf.clone(),
            "amass_frame": env.amass_index.clone(),
            "steps_since_recorded_push": torch.where(self.last_push_step >= 0, self.step - self.last_push_step, -1),
            "steps_since_recorded_reset": torch.where(self.last_reset_step >= 0, self.step - self.last_reset_step, -1),
        }

    def before_step(self, obs, critic_obs, actions):
        self.pre = {
            "observation": obs.detach().clone(), "critic_observation": critic_obs.detach().clone(),
            "action": actions.detach().clone(),
            "policy_mean": self.policy.action_mean.detach().clone(),
            "policy_std": self.policy.action_std.detach().clone(),
            "value_prediction": self.alg.transition.values.detach().clone(),
            "command": self.env.command_generator.command.clone(), "state": self._state(),
        }
        self.push_due.zero_()

    def before_interval_events(self):
        # Observe exactly the timer condition used by Isaac Lab; never resample it.
        manager = self.env.event_manager
        names = manager.active_terms.get("interval", [])
        if "push_robot" in names:
            remaining = manager._interval_term_time_left[names.index("push_robot")]
            self.push_due.copy_((remaining - self.env.step_dt < 1e-6).expand_as(self.push_due))
            self.last_push_step[self.push_due] = self.step

    def after_step_before_reset(self, rewards, dones, timeouts):
        if self.pre is None:
            return
        data = self.env.robot.data
        frame = {
            "iteration": self.iteration, "step": self.step, "rollout_step": self.rollout_step,
            "pre": self.pre,
            "post": {
                "state": self._state(), "command_for_reward": self.env.command_generator.command.clone(),
                "joint_target": data.joint_pos_target.clone(),
                "computed_torque": data.computed_torque.clone(), "applied_torque": data.applied_torque.clone(),
                "effort_limits": data.joint_effort_limits.clone(),
                "reward": rewards.clone(), "reward_terms_per_second": self.env.reward_manager._step_reward.clone(),
                "done": dones.clone(), "timeout": timeouts.clone(), "push_applied": self.push_due.clone(),
            },
        }
        self.history.append(frame)
        for event in self.pending:
            if event["remaining"] > 0:
                event["frames"].append(self._select_frame(frame, event["env_ids"]))
                event["remaining"] -= 1

        raw = self.pre["action"].abs()
        ratios = data.computed_torque[:, self.policy_ids].abs() / data.joint_effort_limits[:, self.policy_ids].clamp_min(1e-6)
        action_peak, torque_peak = raw.amax(-1), ratios.amax(-1)
        finite = torch.isfinite(self.pre["observation"]).all(-1) & torch.isfinite(raw).all(-1)
        finite &= torch.isfinite(self.pre["critic_observation"]).all(-1)
        finite &= torch.isfinite(self.pre["policy_mean"]).all(-1) & torch.isfinite(self.pre["policy_std"]).all(-1)
        finite &= torch.isfinite(ratios).all(-1) & torch.isfinite(rewards)
        severity = torch.stack((action_peak / self.config.action_threshold,
                                torque_peak / self.config.torque_ratio_threshold,
                                -rewards / self.config.negative_reward_threshold)).amax(0)
        severity = torch.where(finite, severity, torch.full_like(severity, float("inf")))
        self.rollout_rows.append(torch.stack((action_peak.max(), torque_peak.max(), rewards.min(),
                                              (~finite).float().sum(), (severity > 1).float().mean())))
        self.joint_maxima.append(raw.amax(0))
        if self.config.rolling_retention:
            for category, scores in (
                ('action', action_peak / self.config.action_threshold),
                ('torque', torque_peak / self.config.torque_ratio_threshold),
                ('negative_reward', -rewards / self.config.negative_reward_threshold),
                ('nonfinite', torch.where(finite, 0., float('inf'))),
            ):
                peak = float(scores.max())
                if peak > 1 and self._can_record(category, peak):
                    self._new_event('rollout_anomaly', self._worst_envs(scores), self.policy.state_dict(),
                                    category=category, severity=peak)
        elif self._can_record() and bool((severity > 1).any()):
            ids = self._worst_envs(severity)
            self._new_event("rollout_anomaly", ids, self.policy.state_dict())
        self.last_push_step[dones.bool()] = -1
        self.last_reset_step[dones.bool()] = self.step
        self.pre = None
        self.step += 1
        self.rollout_step += 1
        self._flush_ready()

    def _can_record(self, category=None, severity=0.):
        if self.config.rolling_retention:
            last_iter, last_severity = self.category_last.get(category, (-self.config.cooldown_iterations, 0.))
            # Each category gets its own cooldown. Escalations can bypass it,
            # at most once per category per iteration to bound pending payloads.
            return len(self.pending) < self.config.max_events and self.iteration > last_iter and (
                self.iteration - last_iter >= self.config.cooldown_iterations or severity > 2 * last_severity)
        return (self.saved_events + len(self.pending) < self.config.max_events and
                self.iteration - self.last_event_iteration >= self.config.cooldown_iterations)

    def _worst_envs(self, severity):
        return severity.topk(min(self.config.envs_per_event, self.env.num_envs)).indices.cpu().tolist()

    def _select_frame(self, frame, env_ids):
        def select(value):
            if isinstance(value, torch.Tensor):
                return value[env_ids].detach().cpu().clone()
            if isinstance(value, dict):
                return {key: select(item) for key, item in value.items()}
            return value
        return select(frame)

    def _new_event(self, reason, ids, policy_before, worst_samples=None, category=None, severity=0.):
        event = {
            "id": f"event_{self.saved_events + len(self.pending):04d}_iter_{self.iteration}_step_{self.step}",
            "reason": reason, "iteration": self.iteration,
            "category": category or reason, "severity": severity,
            "state_context": ("trigger_control_step" if reason == "rollout_anomaly" else "rollout_end_context; exact outlier transitions are in worst_return_samples/worst_update_sample"),
            "trigger_step": self.history[-1]["step"], "env_ids": ids,
            "frames": [self._select_frame(frame, ids) for frame in self.history],
            "remaining": self.config.post_steps, "policy_before": _cpu(policy_before),
            "policy_after": None, "ppo_update": None, "worst_return_samples": _cpu(worst_samples),
            "observation_normalizer": _cpu(self.runner.obs_normalizer.state_dict()),
        }
        self.pending.append(event)
        self.last_event_iteration = self.iteration
        self.category_last[category or reason] = (self.iteration, severity)

    def _install_hooks(self):
        self._original_act = self.alg.act
        self._original_update = self.alg.update
        self._original_generator = self.alg.storage.mini_batch_generator
        self._original_log_prob = self.policy.get_actions_log_prob

        def act(obs, critic_obs):
            actions = self._original_act(obs, critic_obs)
            self.before_step(obs, critic_obs, actions)
            return actions

        def generator(*args, **kwargs):
            for batch in self._original_generator(*args, **kwargs):
                self.batch = batch
                yield batch
            self.batch = None

        def log_prob(actions):
            result = self._original_log_prob(actions)
            if self.collecting_update:
                old_log, old_mean, old_std = self.batch[6:9]
                with torch.no_grad():
                    kl = gaussian_kl(old_mean, old_std, self.policy.action_mean, self.policy.action_std)
                    ratio = (result.detach() - old_log.squeeze(-1)).exp()
                    self.update_rows.append({
                        "kl_mean": kl.mean(), "kl_p99": torch.quantile(kl, .99), "kl_max": kl.max(),
                        "clip_fraction": ((ratio - 1).abs() > self.alg.clip_param).float().mean(),
                        "log_ratio_abs_max": (result.detach() - old_log.squeeze(-1)).abs().max(),
                    })
                    peak = kl.argmax().reshape(1)
                    self.update_samples.append({
                        "observation": self.batch[0].index_select(0, peak).detach().clone(),
                        "critic_observation": self.batch[1].index_select(0, peak).detach().clone(),
                        "action": actions.index_select(0, peak).detach().clone(),
                        "mean_at_minibatch": self.policy.action_mean.index_select(0, peak).detach().clone(),
                        "std_at_minibatch": self.policy.action_std.index_select(0, peak).detach().clone(),
                        "mean_at_collection": old_mean.index_select(0, peak).detach().clone(),
                        "std_at_collection": old_std.index_select(0, peak).detach().clone(),
                        "kl": kl.index_select(0, peak).clone(),
                    })
            return result

        def gradients(grads):
            if self.collecting_update and self.update_rows:
                norms = [g.detach().norm() for g in grads if g is not None]
                self.update_rows[-1]["grad_norm_before_clip"] = torch.stack(norms).norm()

        def optimizer_step(optimizer, args, kwargs):
            if self.collecting_update and self.update_rows:
                self.update_rows[-1]["learning_rate"] = optimizer.param_groups[0]["lr"]

        self.alg.act, self.alg.update = act, self._update
        self.alg.storage.mini_batch_generator = generator
        self.policy.get_actions_log_prob = log_prob
        self._gradient_hook = torch.autograd.graph.register_multi_grad_hook(
            tuple(p for p in self.policy.parameters() if p.requires_grad), gradients)
        self._optimizer_hook = self.alg.optimizer.register_step_pre_hook(optimizer_step)

    def _update(self):
        self.update_rows = []
        self.update_samples = []
        before = {name: value.detach().clone() for name, value in self.policy.state_dict().items()}
        # Keep actual high-error transitions even when the short history no longer
        # contains their full physics state. Their observation/command is not replaced.
        storage = self.alg.storage
        residual = (storage.returns - storage.values).abs().squeeze(-1)
        env_scores, worst_steps = residual.max(0)
        ids = self._worst_envs(env_scores)
        times = worst_steps[ids]
        worst = {"env_ids": ids, "rollout_steps": times.clone(),
                 "observation": storage.observations[times, ids].clone(),
                 "action": storage.actions[times, ids].clone(),
                 "policy_mean": storage.mu[times, ids].clone(),
                 "policy_std": storage.sigma[times, ids].clone(),
                 "return": storage.returns[times, ids].clone(),
                 "value_prediction": storage.values[times, ids].clone()}
        return_stats = {"return_abs_max": storage.returns.abs().max(),
                        "value_abs_max": storage.values.abs().max(),
                        "return_prediction_error_abs_max": residual.max()}
        self.collecting_update = True
        try:
            result = self._original_update()
        finally:
            self.collecting_update = False
        rows = _json_value(self.update_rows)
        summary = {"iteration": self.iteration, **result, **_json_value(return_stats),
                   "learning_rate": self.alg.learning_rate}
        # A KL-stop batch has forward statistics but no gradient/optimizer step.
        for name in {key for row in rows for key in row}:
            values = [float(row[name]) for row in rows if name in row]
            summary[f"{name}_mean"] = sum(values) / len(values)
            summary[f"{name}_max"] = max(values)
        with torch.no_grad():
            delta = torch.stack([(value - before[name]).norm().square()
                                 for name, value in self.policy.state_dict().items()]).sum().sqrt()
            magnitude = torch.stack([value.norm().square() for value in before.values()]).sum().sqrt()
        summary["parameter_relative_change"] = float(delta / magnitude.clamp_min(1e-12))
        with torch.no_grad():
            for group in ("actor", "critic"):
                old = [value for name, value in before.items() if name.startswith(group + ".")]
                changes = [(value - before[name]).norm().square() for name, value in self.policy.state_dict().items()
                           if name.startswith(group + ".")]
                summary[f"{group}_parameter_relative_change"] = float(
                    torch.stack(changes).sum().sqrt() / torch.stack([value.norm().square() for value in old]).sum().sqrt().clamp_min(1e-12))
        rollout = torch.stack(self.rollout_rows)
        summary.update({"action_abs_max": float(rollout[:, 0].max()),
                        "torque_ratio_max": float(rollout[:, 1].max()),
                        "step_reward_min": float(rollout[:, 2].min()),
                        "nonfinite_env_steps": float(rollout[:, 3].sum()),
                        "anomalous_env_fraction_mean": float(rollout[:, 4].mean())})
        loss_spike = (not math.isfinite(result["value_function"]) or
                      result["value_function"] > self.config.value_loss_threshold)
        kl_spike = (not math.isfinite(summary["kl_mean_max"]) or
                    summary["kl_mean_max"] > self.config.kl_threshold)
        loss_severity = result['value_function'] / self.config.value_loss_threshold
        kl_severity = summary['kl_mean_max'] / self.config.kl_threshold
        if not math.isfinite(loss_severity):
            loss_severity = float('inf')
        if not math.isfinite(kl_severity):
            kl_severity = float('inf')
        record_loss = loss_spike and self._can_record('value_loss', loss_severity)
        record_kl = kl_spike and self._can_record('policy_kl', kl_severity)
        peak_sample = None
        if any(event["iteration"] == self.iteration for event in self.pending) or record_loss or record_kl:
            peak_index = max(range(len(rows)), key=lambda i: float(rows[i]["kl_max"])
                             if math.isfinite(float(rows[i]["kl_max"])) else float("inf"))
            peak_sample = self.update_samples[peak_index]
            matches = ((storage.observations.flatten(0, 1) == peak_sample["observation"]).all(-1) &
                       (storage.actions.flatten(0, 1) == peak_sample["action"]).all(-1)).nonzero().flatten()
            peak_sample["minibatch_index"] = peak_index
            peak_sample["matching_rollout_transition_count"] = int(matches.numel())
            peak_sample["matching_rollout_steps"] = (matches[:16] // self.env.num_envs).cpu().tolist()
            peak_sample["matching_env_ids"] = (matches[:16] % self.env.num_envs).cpu().tolist()
        if self.config.rolling_retention:
            if record_loss:
                self._new_event('value_loss_spike', ids, before, worst, category='value_loss', severity=loss_severity)
            if record_kl:
                kl_ids = list(dict.fromkeys(peak_sample['matching_env_ids']))[:self.config.envs_per_event] or ids
                self._new_event('policy_kl_spike', kl_ids, before, worst, category='policy_kl', severity=kl_severity)
        elif self._can_record() and (loss_spike or kl_spike):
            if not loss_spike and peak_sample["matching_env_ids"]:
                ids = list(dict.fromkeys(peak_sample["matching_env_ids"]))[:self.config.envs_per_event]
            self._new_event("value_loss_spike" if loss_spike else "policy_kl_spike", ids, before, worst)
        for event in self.pending:
            if event["iteration"] == self.iteration:
                event["ppo_update"] = {"summary": summary, "minibatches": rows}
                event["policy_after"] = _cpu(self.policy.state_dict())
                event["worst_update_sample"] = _cpu(peak_sample)
                if event["worst_return_samples"] is None:
                    event["worst_return_samples"] = _cpu(worst)
                trigger = next(frame for frame in event["frames"] if frame["step"] == event["trigger_step"])
                obs = trigger["pre"]["observation"].to(self.alg.device)
                with torch.no_grad():
                    after_mean = self.policy.act_inference(obs)
                    after_std = (self.policy.std if self.policy.noise_std_type == "scalar"
                                 else self.policy.log_std.exp()).expand_as(after_mean)
                    before_mean = trigger["pre"]["policy_mean"].to(self.alg.device)
                    before_std = trigger["pre"]["policy_std"].to(self.alg.device)
                    event["same_observation_update"] = _cpu({
                        "mean_before": before_mean, "mean_after": after_mean,
                        "std_before": before_std, "std_after": after_std,
                        "kl": gaussian_kl(before_mean, before_std, after_mean, after_std),
                    })
        self._append_json("updates.jsonl", summary)
        if self.runner.writer is not None:
            for name, value in summary.items():
                if name != "iteration":
                    self.runner.writer.add_scalar(f"Diagnostics/{name}", value, self.iteration)
            peaks = torch.stack(self.joint_maxima).amax(0).cpu().tolist()
            for name, value in zip(self.policy_joint_names, peaks):
                self.runner.writer.add_scalar(f"Diagnostics/JointActionMax/{name}", value, self.iteration)
            self.runner.writer.add_scalar("Diagnostics/events_recorded", self.saved_events + len(self.pending), self.iteration)
            self.runner.writer.add_scalar("Diagnostics/event_limit_reached",
                                          int(not self.config.rolling_retention and self.saved_events + len(self.pending) >= self.config.max_events), self.iteration)
            self.runner.writer.add_scalar('Diagnostics/events_retained', len(self.retained), self.iteration)
        self._flush_ready()
        self.rollout_rows.clear()
        self.joint_maxima.clear()
        self.rollout_step = 0
        self.iteration += 1
        return result

    def _flush_ready(self, force=False):
        for event in list(self.pending):
            if not force and (event["remaining"] > 0 or event["policy_after"] is None):
                continue
            event["post_window_complete"] = event["remaining"] == 0
            trigger = next(frame for frame in event["frames"] if frame["step"] == event["trigger_step"])
            env_summaries = []
            for row, env_id in enumerate(event["env_ids"]):
                raw = trigger["pre"]["action"][row]
                action_id = int(raw.abs().argmax())
                observation = trigger["pre"]["observation"][row]
                obs_id = int(observation.abs().argmax())
                post = trigger["post"]
                ratios = post["computed_torque"][row, self.policy_ids].abs() / post["effort_limits"][row, self.policy_ids].clamp_min(1e-6)
                torque_id = self.policy_ids[int(ratios.argmax())]
                joint_id = self.policy_ids[action_id]
                env_summaries.append({
                    "env_id": env_id,
                    "command_at_action": trigger["pre"]["command"][row],
                    "command_at_reward": post["command_for_reward"][row],
                    "base_height_before": trigger["pre"]["state"]["base_position_w"][row, 2],
                    "base_height_after": post["state"]["base_position_w"][row, 2],
                    "largest_action_joint": self.policy_joint_names[action_id], "raw_action": raw[action_id],
                    "policy_mean": trigger["pre"]["policy_mean"][row, action_id],
                    "policy_std": trigger["pre"]["policy_std"][row, action_id],
                    "action_noise_zscore": ((raw[action_id] - trigger["pre"]["policy_mean"][row, action_id]) /
                                            trigger["pre"]["policy_std"][row, action_id].clamp_min(1e-12)),
                    "largest_observation_feature": self.observation_names[obs_id],
                    "largest_observation_value": observation[obs_id],
                    "target_angle_rad": post["joint_target"][row, joint_id],
                    "actual_angle_rad": post["state"]["joint_position"][row, joint_id],
                    "largest_torque_ratio_joint": self.joint_names[torque_id], "torque_ratio": ratios.max(),
                    "computed_torque_nm": post["computed_torque"][row, torque_id],
                    "applied_torque_nm": post["applied_torque"][row, torque_id],
                    "step_reward": post["reward"][row], "push_this_step": post["push_applied"][row],
                    "done": post["done"][row], "timeout": post["timeout"][row],
                })
            brief = {key: event[key] for key in ("id", "reason", "category", "severity", "state_context", "iteration", "trigger_step", "env_ids", "post_window_complete")}
            brief.update({"environments": env_summaries,
                          "ppo_update": event["ppo_update"],
                          "worst_update_sample": event.get("worst_update_sample"),
                          "worst_return_samples": event.get("worst_return_samples"),
                          "same_observation_update": event.get("same_observation_update"),
                          "payload": f"{event['id']}.pt"})
            torch.save({"metadata": self._metadata, **event}, self.directory / brief["payload"])
            self._write_json(self.directory / f"{event['id']}.json", brief)
            self._append_json("events.jsonl", brief)
            self.pending.remove(event)
            self.saved_events += 1
            if self.config.rolling_retention:
                self.retained.append({key: event[key] for key in ('id', 'category', 'severity')})
                self._prune_retained()
            print(f"[Diagnostics] Saved {event['id']}: {event['reason']}; envs={event['env_ids']}", flush=True)

    def _prune_retained(self):
        """Balance category slots; protect the newest and severest within each."""
        while len(self.retained) > self.config.max_events:
            groups = {}
            for record in self.retained:
                groups.setdefault(record['category'], []).append(record)
            eligible = []
            for group in groups.values():
                protected = {group[-1]['id'], max(group, key=lambda row: row['severity'])['id']}
                eligible.extend((len(group), row) for row in group if row['id'] not in protected)
            # >=12 slots reserve two for every one of the six trigger categories.
            _, victim = max(eligible, key=lambda pair: pair[0])
            for suffix in ('.pt', '.json'):
                (self.directory / (victim['id'] + suffix)).unlink(missing_ok=True)
            self.retained.remove(victim)
            self._append_json('retention.jsonl', {'pruned': victim['id'], 'category': victim['category'],
                                                 'at_iteration': self.iteration})

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._flush_ready(force=True)
        self.alg.act, self.alg.update = self._original_act, self._original_update
        self.alg.storage.mini_batch_generator = self._original_generator
        self.policy.get_actions_log_prob = self._original_log_prob
        self._gradient_hook.remove()
        self._optimizer_hook.remove()
        self.env.training_diagnostics = None
        self.history.clear()
        if self.runner.writer is not None and hasattr(self.runner.writer, "flush"):
            self.runner.writer.flush()
