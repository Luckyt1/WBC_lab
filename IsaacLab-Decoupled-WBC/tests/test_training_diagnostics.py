"""Diagnostics must preserve PPO updates/RNG and capture the pre-reset incident."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from legged_lab.utils.training_diagnostics import DiagnosticsConfig, TrainingDiagnostics, gaussian_kl


class FakeEnv:
    def __init__(self):
        self.num_envs, self.num_actions, self.device, self.step_dt = 4, 3, "cpu", .02
        self.custom_joint_ids = [1, 0, 2]
        self.action_joint_names = ["left_knee", "waist", "right_knee"]
        self.base_body_id = 0
        self.feet_cfg = NS(body_ids=[2, 0])
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.amass_index = torch.zeros(4, dtype=torch.long)
        self.command_generator = NS(command=torch.arange(7.).repeat(4, 1))
        quat = torch.tensor([1., 0, 0, 0]).repeat(4, 3, 1)
        self.robot = NS(joint_names=["waist", "left_knee", "right_knee"],
                        body_names=["torso_link", "left_foot", "right_foot"], data=NS(
                            joint_pos=torch.zeros(4, 3), joint_vel=torch.zeros(4, 3),
                            joint_pos_target=torch.zeros(4, 3), computed_torque=torch.zeros(4, 3),
                            applied_torque=torch.zeros(4, 3), joint_effort_limits=torch.ones(4, 3),
                            body_pos_w=torch.arange(9.).reshape(1, 3, 3).repeat(4, 1, 1), body_quat_w=quat))
        self.base_quat_w = quat[:, 0]
        self.base_lin_vel_w = torch.zeros(4, 3)
        self.base_ang_vel_w = torch.zeros(4, 3)
        self.contact_sensor = NS(body_names=["right_foot", "torso_link", "left_foot"],
                                data=NS(net_forces_w=torch.zeros(4, 3, 3),
                                         current_contact_time=torch.ones(4, 3)))
        self.reward_manager = NS(active_terms=["torque"], _step_reward=torch.zeros(4, 1))
        self.event_manager = NS(active_terms={"interval": ["push_robot"]},
                                _interval_term_time_left=[torch.zeros(4)])
        self.training_diagnostics = None

    def observations(self):
        obs = torch.cat((self.command_generator.command, self.robot.data.joint_pos,
                         self.robot.data.joint_vel), -1)
        return obs, torch.cat((obs, self.base_lin_vel_w), -1)

    def step(self, actions):
        data = self.robot.data
        data.joint_pos_target[:, self.custom_joint_ids] = actions
        data.joint_pos.copy_(data.joint_pos_target * .5)
        data.computed_torque.copy_(data.joint_pos_target * 10)
        data.applied_torque.copy_(data.computed_torque.clamp(-1, 1))
        self.episode_length_buf += 1
        self.command_generator.command += .01  # a resampled command differs from the actor input
        if self.training_diagnostics:
            self.training_diagnostics.before_interval_events()
        reward = -actions.square().sum(-1) * self.step_dt
        self.reward_manager._step_reward[:, 0] = reward / self.step_dt
        done = torch.zeros(4, dtype=torch.long)
        done[0] = 1
        if self.training_diagnostics:
            self.training_diagnostics.after_step_before_reset(reward, done, torch.zeros_like(done))
        data.joint_pos[0] = 0  # these reset values must NOT replace the saved physical state
        self.episode_length_buf[0] = 0
        obs, critic = self.observations()
        return obs, critic, reward, done


def run_training(path, config=None, iterations=3, guarded=False):
    torch.manual_seed(83)
    env = FakeEnv()
    policy = ActorCritic(13, 16, 3, actor_hidden_dims=[8], critic_hidden_dims=[8])
    alg = PPO(policy, num_learning_epochs=2, num_mini_batches=2, learning_rate=.001,
              max_grad_norm=.1, schedule="adaptive", desired_kl=.01)
    alg.init_storage("rl", 4, 4, [13], [16], [3])
    if guarded:
        from legged_lab.utils.ppo_stability import install_ppo_stability
        install_ppo_stability(alg, dict(initial_learning_rate=1e-4, min_learning_rate=1e-5,
                                      max_learning_rate=3e-4, max_mean_kl=1e-8))
    scalars = {}
    writer = NS(add_scalar=lambda tag, value, step: scalars.update({(tag, step): value}))
    runner = NS(env=env, alg=alg, log_dir=str(path), num_steps_per_env=4,
                current_learning_iteration=1000, obs_normalizer=torch.nn.Identity(),
                writer=writer, is_distributed=False, cfg={})
    recorder = TrainingDiagnostics(runner, config) if config else None
    obs, critic = env.observations()
    outputs = []
    for _ in range(iterations):
        with torch.inference_mode():
            for _ in range(4):
                actions = alg.act(obs, critic)
                outputs.append(actions.clone())
                obs, critic, reward, done = env.step(actions)
                alg.process_env_step(reward, done, {})
            alg.compute_returns(critic)
        alg.update()
    if recorder:
        recorder.close()
    return {k: v.clone() for k, v in policy.state_dict().items()}, outputs, torch.get_rng_state(), recorder, scalars


def test_diagnostics_do_not_change_actions_weights_or_rng(tmp_path):
    baseline = run_training(tmp_path / "baseline")
    observed = run_training(tmp_path / "observed", DiagnosticsConfig(
        pre_steps=2, post_steps=2, max_events=2, cooldown_iterations=1, action_threshold=.01,
    ))
    for key in baseline[0]:
        torch.testing.assert_close(baseline[0][key], observed[0][key], rtol=0, atol=0)
    for left, right in zip(baseline[1], observed[1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(baseline[2], observed[2])
    records = list(observed[3].directory.glob("event_*.json"))
    assert len(records) == 2  # event limit bounds disk writes
    rows = [json.loads(line) for line in (observed[3].directory / "updates.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [1000, 1001, 1002]
    assert all(row["grad_norm_before_clip_max"] > .1 for row in rows)
    assert all(row["parameter_relative_change"] > 0 for row in rows)
    assert ("Diagnostics/JointActionMax/left_knee", 1000) in observed[4]


def test_event_preserves_command_timing_reset_state_and_policy_pair(tmp_path):
    _, _, _, recorder, _ = run_training(tmp_path, DiagnosticsConfig(
        pre_steps=2, post_steps=2, envs_per_event=4, max_events=1, action_threshold=.01,
    ))
    record = torch.load(next(recorder.directory.glob("event_*.pt")), weights_only=True)
    assert record["post_window_complete"]
    assert len(record["ppo_update"]["minibatches"]) == 4
    assert record["worst_update_sample"]["matching_rollout_transition_count"] >= 1
    assert len(record["worst_update_sample"]["matching_env_ids"]) >= 1
    trigger = record["frames"][0]
    assert trigger["iteration"] == 1000 and trigger["step"] == 0
    torch.testing.assert_close(trigger["post"]["command_for_reward"], trigger["pre"]["command"] + .01)
    row = record["env_ids"].index(0)
    assert trigger["post"]["done"][row] == 1
    assert trigger["post"]["state"]["joint_position"][row].abs().sum() > 0
    assert trigger["post"]["push_applied"].all()
    policy = ActorCritic(13, 16, 3, actor_hidden_dims=[8], critic_hidden_dims=[8])
    for name in ("before", "after"):
        policy.load_state_dict(record[f"policy_{name}"])
        with torch.no_grad():
            mean = policy.act_inference(trigger["pre"]["observation"])
        torch.testing.assert_close(mean, record["same_observation_update"][f"mean_{name}"])
    brief = json.loads(next(recorder.directory.glob("event_*.json")).read_text())
    assert brief["environments"][0]["largest_action_joint"] in recorder.policy_joint_names
    assert recorder.env.training_diagnostics is None


def test_loss_only_event_keeps_worst_transitions_and_flushes_partial_window(tmp_path):
    _, _, _, recorder, _ = run_training(tmp_path, DiagnosticsConfig(
        pre_steps=2, post_steps=5, max_events=1, action_threshold=1e9,
        torque_ratio_threshold=1e9, negative_reward_threshold=1e9, value_loss_threshold=1e-9,
    ), iterations=1)
    record = torch.load(next(recorder.directory.glob("event_*.pt")), weights_only=True)
    assert record["reason"] == "value_loss_spike"
    assert not record["post_window_complete"]
    assert record["policy_after"] is not None
    assert record["worst_return_samples"]["observation"].shape == (2, 13)


def test_foot_positions_resolve_sensor_names_in_articulation_order(tmp_path):
    _, _, _, recorder, _ = run_training(tmp_path, DiagnosticsConfig(
        max_events=1, action_threshold=.01,
    ), iterations=1)
    record = torch.load(next(recorder.directory.glob("event_*.pt")), weights_only=True)
    assert record["metadata"]["feet_sensor_ids"] == [2, 0]
    assert record["metadata"]["feet_body_ids"] == [1, 2]
    assert record["metadata"]["feet_body_names"] == ["left_foot", "right_foot"]
    positions = record["frames"][0]["pre"]["state"]["feet_position_w"]
    expected = torch.tensor([[3., 4., 5.], [6., 7., 8.]])
    torch.testing.assert_close(positions, expected.expand_as(positions))


def test_gaussian_kl_matches_torch_distribution():
    old_mu = torch.tensor([[0., 1.], [.3, -.2]])
    mu = old_mu + .2
    old_std, std = torch.ones_like(mu), torch.ones_like(mu) * 1.2
    reference = torch.distributions.kl_divergence(
        torch.distributions.Normal(old_mu, old_std), torch.distributions.Normal(mu, std)).sum(-1)
    torch.testing.assert_close(gaussian_kl(old_mu, old_std, mu, std), reference)


def test_rolling_retention_continues_past_cap_and_keeps_each_category(tmp_path):
    _, _, _, recorder, _ = run_training(tmp_path, DiagnosticsConfig(
        pre_steps=1, post_steps=1, max_events=12, cooldown_iterations=1, rolling_retention=True,
        action_threshold=.01, torque_ratio_threshold=.01, value_loss_threshold=1e-9,
    ), iterations=10)
    assert recorder.saved_events > 12
    assert len(list(recorder.directory.glob('event_*.pt'))) == 12
    manifest = [json.loads(row) for row in (recorder.directory / 'events.jsonl').read_text().splitlines()]
    for category in {row['category'] for row in manifest}:
        events = [row for row in manifest if row['category'] == category]
        for expected in (events[-1], max(events, key=lambda row: row['severity'])):
            assert (recorder.directory / expected['payload']).is_file()
    assert (recorder.directory / 'retention.jsonl').is_file()


def test_diagnostics_accept_kl_stop_batch_without_gradients(tmp_path):
    _, _, _, recorder, _ = run_training(tmp_path, DiagnosticsConfig(
        pre_steps=1, post_steps=1, max_events=12, cooldown_iterations=1, rolling_retention=True,
        action_threshold=.01,
    ), guarded=True)
    updates = [json.loads(row) for row in (recorder.directory / 'updates.jsonl').read_text().splitlines()]
    assert any(row['ppo_early_stop'] == 1 for row in updates)
    assert all(row['ppo_updates'] < row['ppo_batches_evaluated'] for row in updates if row['ppo_early_stop'])
