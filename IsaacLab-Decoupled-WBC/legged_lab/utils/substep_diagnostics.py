"""Read-only, per-physics-step recording for bounded fixed-policy experiments."""

from collections import deque
import csv
import json
from pathlib import Path
import torch


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    return value


class SubstepDiagnostics:
    def __init__(self, env, directory, max_events=12, pre_seconds=.5, post_seconds=.3, continuous_envs=8,
                 save_physics_series=False):
        self.env, self.directory = env, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_events = max_events
        self.history = deque(maxlen=round(pre_seconds / env.physics_dt) + 1)
        self.post_steps = round(post_seconds / env.physics_dt)
        self.pending, self.event_summaries, self.stats = [], [], []
        self.physics_series = [] if save_physics_series else None
        self.continuous_env_ids = list(range(min(continuous_envs, env.num_envs)))
        self.continuous_frames, self.continuous_chunks = [], 0
        if self.continuous_env_ids:
            (self.directory / 'continuous').mkdir()
        self.total_step, self.control_step, self.substep = 0, 0, 0
        self.enabled = False
        self.last_event_step = -100000
        self.category_counts, self.category_last = {}, {}
        self.next_id = 0
        self.counts = {key: 0 for key in ('ankle_speed_crossings_50', 'ankle_speed_crossings_100',
                                         'early_ankle_crossings_50', 'action_crossings_20', 'nonfinite_substeps')}
        self.ankle_ids = [i for i, name in enumerate(env.robot.joint_names) if '_ankle_' in name]
        self.policy_ids = env.custom_joint_ids[:env.num_actions]
        self.previous_action_big = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        data = env.robot.data
        self.limits = data.joint_pos_limits.clone()
        self.vel_limits = data.joint_vel_limits.clone()
        self.stiffness, self.damping = data.joint_stiffness.clone(), data.joint_damping.clone()
        self.metadata = {
            'schema_version': 1, 'physics_dt': env.physics_dt, 'control_dt': env.step_dt,
            'joint_names': list(env.robot.joint_names), 'body_names': list(env.contact_sensor.body_names),
            'policy_joint_names': list(env.action_joint_names), 'policy_joint_ids': self.policy_ids,
            'ankle_ids': self.ankle_ids,
            'articulation_body_names': list(env.robot.body_names),
            'torque_semantics': 'Implicit drive computed/applied torque are pre-step PD estimates, not measured PhysX drive torques. The saved q_before/qd_before/target refer to the same PD calculation.',
            'sampling': 'One record per physics update; full event windows plus all-substep aggregate CSV. No simulator state is changed by recording.',
            'continuous_env_ids': self.continuous_env_ids,
            'event_trigger': 'Separate quotas/cooldowns for early ankle |qd| crossings 50 and 100 rad/s (raw action peak <20), and raw action peak >20 crossings. Event capture does not affect aggregate counts.',
        }
        static = {'joint_limits': self.limits, 'joint_velocity_limits': self.vel_limits,
                  'stiffness': self.stiffness, 'damping': self.damping, 'armature': data.joint_armature,
                  'effort_limits': data.joint_effort_limits, 'default_joint_pos': data.default_joint_pos}
        view = env.robot.root_physx_view
        for name, method in [('masses', 'get_masses'), ('inertias', 'get_inertias'), ('materials', 'get_material_properties')]:
            static[name] = getattr(view, method)().clone()
        self.static = cpu(static)
        torch.save({'metadata': self.metadata, **self.static}, self.directory / 'physical_properties.pt')
        (self.directory / 'metadata.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        self.original_write, self.original_update = env.scene.write_data_to_sim, env.scene.update
        env.scene.write_data_to_sim, env.scene.update = self._write, self._update

    def begin_control(self, obs, actions, mean, std):
        self.substep = 0
        self.enabled = True
        self.context = {
            'observation': obs.clone(), 'raw_action': actions.clone(), 'policy_mean': mean.clone(),
            'policy_std': std.clone(), 'command': self.env.command_generator.command.clone(),
            'amass_index': self.env.amass_index.clone(), 'episode_step': self.env.episode_length_buf.clone(),
        }
        manager = self.env.event_manager
        self.context['interval_timers'] = torch.stack(manager._interval_term_time_left, -1).clone()
        large = actions.abs().amax(-1) > 20
        self.action_crossing = large & ~self.previous_action_big
        self.counts['action_crossings_20'] += int(self.action_crossing.sum())
        self.previous_action_big = large

    def end_control(self, done):
        self.enabled = False
        assert self.substep == self.env.cfg.sim.decimation, (self.substep, self.env.cfg.sim.decimation)
        self.previous_action_big[done.bool()] = False
        self.control_step += 1

    def _write(self, *args, **kwargs):
        self.original_write(*args, **kwargs)
        if not self.enabled or self.substep >= self.env.cfg.sim.decimation:
            return
        d = self.env.robot.data
        self.pre = {
            'q': d.joint_pos.clone(), 'qd': d.joint_vel.clone(), 'target': d.joint_pos_target.clone(),
            'velocity_target': d.joint_vel_target.clone(), 'feedforward_effort': d.joint_effort_target.clone(),
            'computed_torque': d.computed_torque.clone(), 'applied_torque_estimate': d.applied_torque.clone(),
            'root_state': d.root_state_w.clone(),
            'contact_force': self.env.contact_sensor.data.net_forces_w.clone(),
        }

    def _update(self, dt):
        self.original_update(dt)
        if not self.enabled or self.substep >= self.env.cfg.sim.decimation:
            return
        assert abs(dt-self.env.physics_dt) < 1e-10
        d = self.env.robot.data
        post = {'q': d.joint_pos.clone(), 'qd': d.joint_vel.clone(), 'root_state': d.root_state_w.clone(),
                'contact_force': self.env.contact_sensor.data.net_forces_w.clone()}
        position_term = self.stiffness * (self.pre['target'] - self.pre['q'])
        velocity_term = self.damping * (self.pre['velocity_target'] - self.pre['qd'])
        self.pre['pd_position_term'], self.pre['pd_velocity_term'] = position_term, velocity_term
        estimated = position_term + velocity_term + self.pre['feedforward_effort']
        pd_error = (estimated - self.pre['computed_torque']).abs().amax()
        post['joint_limit_margin'] = torch.minimum(post['q'] - self.limits[..., 0], self.limits[..., 1] - post['q'])
        self.pre['target_limit_excess'] = ((self.limits[..., 0]-self.pre['target']).clamp_min(0) +
                                         (self.pre['target']-self.limits[..., 1]).clamp_min(0))
        before_speed = self.pre['qd'][:, self.ankle_ids].abs().amax(-1)
        after_speed = post['qd'][:, self.ankle_ids].abs().amax(-1)
        if self.physics_series is not None:
            self.physics_series.append(cpu({
                'ankle_q': post['q'][:, self.ankle_ids],
                'ankle_qd': post['qd'][:, self.ankle_ids],
                'ankle_speed_before': before_speed,
                'ankle_margin': post['joint_limit_margin'][:, self.ankle_ids],
                'contact_max': post['contact_force'].norm(dim=-1).amax(-1),
            }))
        self.post = post
        cross50 = (before_speed < 50) & (after_speed >= 50)
        cross100 = (before_speed < 100) & (after_speed >= 100)
        raw_peak = self.context['raw_action'].abs().amax(-1)
        early = cross50 & (raw_peak < 20)
        finite = torch.isfinite(post['q']).all() & torch.isfinite(post['qd']).all()
        metrics = torch.stack((after_speed.max(), self.pre['qd'][:, self.ankle_ids].abs().max(),
                               post['qd'].abs().max(), raw_peak.max(), cross50.sum(), cross100.sum(), early.sum(),
                               (-post['joint_limit_margin']).clamp_min(0).max(),
                               self.pre['target_limit_excess'].max(), post['contact_force'].norm(dim=-1).max(),
                               pd_error, (~finite).float())).cpu().tolist()
        for key, val in zip(('ankle_speed_crossings_50', 'ankle_speed_crossings_100', 'early_ankle_crossings_50'), metrics[4:7]):
            self.counts[key] += int(val)
        self.counts['nonfinite_substeps'] += int(metrics[-1])
        self.stats.append([self.total_step, self.control_step, self.substep, (self.total_step+1)*dt, *metrics])
        frame = {'physics_step': self.total_step, 'control_step': self.control_step, 'substep': self.substep,
                 'time_after_s': (self.total_step+1)*dt, 'control': self.context, 'pre': self.pre, 'post': post}
        if self.continuous_env_ids:
            self.continuous_frames.append(self._select(frame, self.continuous_env_ids))
            if len(self.continuous_frames) >= 200:
                self._flush_continuous()
        self.history.append(frame)
        for event in self.pending:
            if event['remaining'] > 0:
                event['frames'].append(self._select(frame, event['env_id']))
                event['remaining'] -= 1
        for category, candidates, score in (
            ('early_ankle_velocity_100', cross100 & (raw_peak < 20), after_speed),
            ('action_escalation', self.action_crossing & (self.substep == 0), raw_peak),
            ('early_ankle_velocity_50', early & ~cross100, after_speed),
        ):
            can_record = (self.next_id < self.max_events and
                          self.category_counts.get(category, 0) < max(1, self.max_events//3) and
                          self.total_step-self.category_last.get(category, -100000) >= round(.1/dt))
            if can_record and bool(candidates.any()):
                env_id = int(torch.where(candidates, score, -torch.ones_like(score)).argmax())
                event = {'id': f'event_{self.next_id:03d}_{category}_step_{self.total_step}',
                         'category': category, 'env_id': env_id, 'trigger_step': self.total_step,
                         'frames': [self._select(f, env_id) for f in self.history], 'remaining': self.post_steps}
                self.pending.append(event)
                self.next_id += 1
                self.last_event_step = self.total_step
                self.category_counts[category] = self.category_counts.get(category, 0)+1
                self.category_last[category] = self.total_step
        self._flush()
        self.substep += 1
        self.total_step += 1

    @staticmethod
    def _select(frame, env_id):
        def select(value):
            if isinstance(value, torch.Tensor):
                return value[env_id].detach().cpu().clone()
            if isinstance(value, dict):
                return {key: select(item) for key, item in value.items()}
            return value
        return select(frame)

    def _flush_continuous(self):
        if not self.continuous_frames:
            return
        def stack(values):
            if isinstance(values[0], torch.Tensor):
                return torch.stack(values)
            if isinstance(values[0], dict):
                return {key: stack([value[key] for value in values]) for key in values[0]}
            return torch.tensor(values)
        torch.save({'env_ids': self.continuous_env_ids, 'data': stack(self.continuous_frames)},
                   self.directory / 'continuous' / f'chunk_{self.continuous_chunks:04d}.pt')
        self.continuous_frames.clear()
        self.continuous_chunks += 1

    def _flush(self, force=False):
        for event in list(self.pending):
            if not force and event['remaining'] > 0:
                continue
            trigger = next(f for f in event['frames'] if f['physics_step'] == event['trigger_step'])
            j = self.ankle_ids[int(trigger['post']['qd'][self.ankle_ids].abs().argmax())]
            brief = {key: event[key] for key in ('id', 'category', 'env_id', 'trigger_step')}
            brief.update({'joint': self.env.robot.joint_names[j], 'control_step': trigger['control_step'],
                          'substep': trigger['substep'], 'qd_before': float(trigger['pre']['qd'][j]),
                          'qd_after': float(trigger['post']['qd'][j]), 'q_before': float(trigger['pre']['q'][j]),
                          'q_after': float(trigger['post']['q'][j]), 'target': float(trigger['pre']['target'][j]),
                          'command': trigger['control']['command'].tolist(),
                          'raw_action_peak': float(trigger['control']['raw_action'].abs().max()),
                          'post_window_complete': event['remaining'] == 0})
            torch.save({'metadata': self.metadata, **event}, self.directory / (event['id']+'.pt'))
            (self.directory / (event['id']+'.json')).write_text(json.dumps(brief, indent=2)+'\n')
            self.event_summaries.append(brief)
            self.pending.remove(event)
            print('[Substep event]', json.dumps(brief), flush=True)

    def close(self):
        self.enabled = False
        self._flush(force=True)
        self._flush_continuous()
        self.env.scene.write_data_to_sim, self.env.scene.update = self.original_write, self.original_update
        columns = ['physics_step', 'control_step', 'substep', 'time_after_s', 'ankle_speed_max',
                   'ankle_speed_before_max', 'joint_speed_max', 'raw_action_max', 'crossings_50', 'crossings_100',
                   'early_crossings_50', 'joint_limit_violation_max_rad', 'target_limit_excess_max_rad',
                   'contact_force_max_N', 'pd_alignment_error_Nm', 'nonfinite']
        with (self.directory / 'substep_metrics.csv').open('w') as stream:
            writer = csv.writer(stream); writer.writerow(columns); writer.writerows(self.stats)
        if self.physics_series:
            torch.save({key: torch.stack([f[key] for f in self.physics_series])
                        for key in self.physics_series[0]}, self.directory/'physics_series.pt')
            self.physics_series.clear()
        self.history.clear()
