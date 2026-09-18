from copy import deepcopy

import pytest
import torch

from legged_lab.utils.feedback_smoothness import FeedbackSmoothness
from legged_lab.utils.ppo_stability import install_ppo_stability
from test_ppo_stability import CONFIG, make_ppo
from test_ppo_transaction import transaction, assert_nested_equal


CFG = dict(weight=.01, sigma=.05, action_indices=[3, 4], max_samples=32, seed=713)


def test_exact_linear_sensitivity_and_gradient_without_input_or_rng_mutation():
    class LinearPolicy:
        def __init__(self):
            self.weight = torch.tensor([[1., 2., 3., 4., 5.], [-3., 1., 0., 2., -1.]], requires_grad=True)
        def act_inference(self, obs):
            return obs @ self.weight.T
    policy = LinearPolicy()
    obs = torch.zeros(32, 5)
    before, rng = obs.clone(), torch.get_rng_state()
    reg = FeedbackSmoothness(CFG, 5, 2, 'cpu')
    loss = reg.loss(policy, obs)
    expected = (reg.directions @ policy.weight[:, [3, 4]].T).square().mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert policy.weight.grad[:, :3].abs().max() < 1e-5
    assert policy.weight.grad[:, 3:].abs().max() > 0
    torch.testing.assert_close(obs, before, atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        policy.weight -= .01 * policy.weight.grad
    assert reg.loss(policy, obs) < loss


def test_penalty_does_not_backpropagate_into_critic_or_change_distribution():
    alg = make_ppo()
    reg = FeedbackSmoothness(CFG, 5, 2, 'cpu')
    mean, std = alg.policy.action_mean.clone(), alg.policy.action_std.clone()
    reg.loss(alg.policy, alg.storage.observations.flatten(0, 1)).backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in alg.policy.actor.parameters())
    assert all(p.grad is None for p in alg.policy.critic.parameters())
    torch.testing.assert_close(alg.policy.action_mean, mean, atol=0, rtol=0)
    torch.testing.assert_close(alg.policy.action_std, std, atol=0, rtol=0)


def test_integrated_update_logs_penalty_and_transaction_still_restores(tmp_path):
    alg = make_ppo()
    cfg = transaction(alg, tmp_path)
    install_ppo_stability(alg, {**CONFIG, 'feedback_smoothness': CFG, 'transaction': cfg})
    before, adam = deepcopy(alg.policy.state_dict()), deepcopy(alg.optimizer.state_dict())
    def corrupt(*_):
        with torch.no_grad():
            alg.policy.actor[-1].bias.add_(50.)
    hook = alg.optimizer.register_step_post_hook(corrupt)
    result = alg.update()
    hook.remove()
    assert result['feedback_smoothness_loss'] > 0
    assert result['feedback_smoothness_weighted_loss'] == pytest.approx(.01 * result['feedback_smoothness_loss'])
    assert result['transaction_reverted'] == 1
    assert_nested_equal(before, alg.policy.state_dict())
    assert_nested_equal(adam, alg.optimizer.state_dict())


@pytest.mark.parametrize('change', [dict(sigma=0), dict(weight=float('nan')), dict(action_indices=[3, 3])])
def test_invalid_settings_rejected(change):
    with pytest.raises(ValueError):
        FeedbackSmoothness({**CFG, **change}, 5, 2, 'cpu')
