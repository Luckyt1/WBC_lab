from copy import deepcopy
import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from legged_lab.utils.ppo_stability import install_ppo_stability

CONFIG = dict(initial_learning_rate=1e-4, min_learning_rate=1e-5,
              max_learning_rate=3e-4, max_mean_kl=.02)


def make_ppo():
    torch.manual_seed(5)
    policy = ActorCritic(5, 5, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
    alg = PPO(policy, num_learning_epochs=3, num_mini_batches=4,
              learning_rate=1e-4, schedule='fixed')
    alg.init_storage('rl', 4, 8, [5], [5], [2])
    with torch.inference_mode():
        for _ in range(8):
            obs = torch.randn(4, 5)
            alg.act(obs, obs)
            alg.process_env_step(torch.randn(4), torch.zeros(4), {})
        alg.compute_returns(torch.zeros(4, 5))
    return alg


def test_uninterrupted_guard_preserves_original_ppo_objective_and_rng():
    original = make_ppo()
    original_loss = original.update()
    state, rng = deepcopy(original.policy.state_dict()), torch.get_rng_state()
    guarded = make_ppo()
    install_ppo_stability(guarded, {**CONFIG, 'max_mean_kl': 100.})
    loss = guarded.update()
    assert loss['ppo_updates'] == 12 and loss['ppo_early_stop'] == 0
    for key in original_loss:
        assert loss[key] == pytest.approx(original_loss[key], abs=1e-7)
    for key in state:
        torch.testing.assert_close(state[key], guarded.policy.state_dict()[key], atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert guarded.storage.step == 0


def test_early_stop_skips_remaining_updates_and_averages_applied_batches():
    alg = make_ppo()
    install_ppo_stability(alg, CONFIG)
    observed = []
    original = alg.storage.mini_batch_generator
    def batches(*args):
        for batch in original(*args):
            with torch.no_grad():
                value = alg.policy.evaluate(batch[1])
                clipped = batch[3] + (value - batch[3]).clamp(-alg.clip_param, alg.clip_param)
                observed.append(torch.maximum((value-batch[5]).square(), (clipped-batch[5]).square()).mean().item())
            yield batch
    alg.storage.mini_batch_generator = batches
    def force_large_update(*_):
        with torch.no_grad():
            alg.policy.actor[-1].bias.add_(2.)
    hook = alg.optimizer.register_step_post_hook(force_large_update)
    loss = alg.update()
    hook.remove()
    assert loss['ppo_updates'] == 1 and loss['ppo_batches_evaluated'] == 2
    assert loss['ppo_early_stop'] == 1 and loss['ppo_mean_kl_max'] > .02
    assert loss['value_function'] == pytest.approx(observed[0])
    assert alg.storage.step == 0


def test_loaded_adam_state_preserved_and_all_lr_groups_capped():
    alg = make_ppo()
    alg.update()
    before = deepcopy(alg.optimizer.state_dict()['state'])
    alg.optimizer.param_groups[0]['lr'] = .008
    install_ppo_stability(alg, CONFIG, reset_learning_rate=True)
    assert alg.learning_rate == 1e-4
    for key, values in before.items():
        for name, value in values.items():
            torch.testing.assert_close(value, alg.optimizer.state_dict()['state'][key][name], rtol=0, atol=0)
    alg.optimizer.param_groups[0]['lr'] = .008
    install_ppo_stability(alg, CONFIG)
    assert alg.learning_rate == 3e-4


def test_adaptive_learning_rate_never_exceeds_cap():
    alg = make_ppo()
    alg.schedule, alg.desired_kl = 'adaptive', 1.
    install_ppo_stability(alg, {**CONFIG, 'max_mean_kl': 100.})
    lrs = []
    hook = alg.optimizer.register_step_pre_hook(lambda optimizer, *args: lrs.append(optimizer.param_groups[0]['lr']))
    alg.update()
    hook.remove()
    assert len(lrs) == 12 and max(lrs) == 3e-4 and min(lrs) >= 1e-5
