from copy import deepcopy

import pytest
import torch

from test_ppo_stability import make_ppo, CONFIG
from legged_lab.utils.ppo_stability import install_ppo_stability
from legged_lab.utils.ppo_transaction import feedback_peaks


def transaction(alg, tmp_path, **overrides):
    obs = alg.storage.observations[0].clone()
    ids = [3, 4]
    bank = dict(observations=obs, action_indices=ids, steps=4, clip=100.,
                reference_peaks=feedback_peaks(alg.policy, obs, ids, 4, 100.))
    path = tmp_path / 'bank.pt'
    torch.save(bank, path)
    return dict(feedback_bank=str(path), post_mean_kl=100., post_max_kl=10000.,
                feedback_factor=100., feedback_floor=100., **overrides)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_accepted_transaction_preserves_update_and_rng(tmp_path):
    ordinary = make_ppo()
    install_ppo_stability(ordinary, {**CONFIG, 'max_mean_kl': 100.})
    expected_loss = ordinary.update()
    expected, rng = deepcopy(ordinary.policy.state_dict()), torch.get_rng_state()
    guarded = make_ppo()
    cfg = transaction(guarded, tmp_path)
    install_ppo_stability(guarded, {**CONFIG, 'max_mean_kl': 100., 'transaction': cfg})
    result = guarded.update()
    assert result['transaction_reverted'] == 0 and result['ppo_updates'] == 12
    for key, value in expected_loss.items():
        assert result[key] == pytest.approx(value, abs=1e-7)
    assert_nested_equal(expected, guarded.policy.state_dict())
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('reason', ['kl', 'feedback'])
def test_rejected_transaction_restores_policy_and_adam_exactly(tmp_path, reason):
    alg = make_ppo()
    # Populate real Adam moments/step counters without changing the policy.
    for p in alg.policy.parameters():
        p.grad = torch.zeros_like(p)
    alg.optimizer.step()
    cfg = transaction(alg, tmp_path)
    if reason == 'kl':
        cfg['post_mean_kl'] = .001
    else:
        cfg.update(post_mean_kl=1e9, post_max_kl=1e10, feedback_floor=.1, feedback_factor=1.)
    install_ppo_stability(alg, {**CONFIG, 'transaction': cfg})
    policy_before, adam_before = deepcopy(alg.policy.state_dict()), deepcopy(alg.optimizer.state_dict())
    def corrupt(*_):
        with torch.no_grad():
            alg.policy.actor[-1].bias.add_(50.)
    hook = alg.optimizer.register_step_post_hook(corrupt)
    result = alg.update()
    hook.remove()
    assert result['transaction_reverted'] == 1
    assert result[f'transaction_{reason}_rejected'] == 1
    assert result['ppo_updates'] == 0 and result['ppo_attempted_updates'] == 1
    assert alg.storage.step == 0
    assert_nested_equal(policy_before, alg.policy.state_dict())
    assert_nested_equal(adam_before, alg.optimizer.state_dict())


def test_feedback_probe_is_read_only(tmp_path):
    alg = make_ppo()
    obs = alg.storage.observations[0].clone()
    before, rng = obs.clone(), torch.get_rng_state()
    a = feedback_peaks(alg.policy, obs, [3, 4], 8, 100.)
    b = feedback_peaks(alg.policy, obs, [3, 4], 8, 100.)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(obs, before, atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())


def test_transaction_requires_fixed_lr(tmp_path):
    alg = make_ppo()
    cfg = transaction(alg, tmp_path)
    alg.schedule = 'adaptive'
    with pytest.raises(ValueError, match='fixed learning rate'):
        install_ppo_stability(alg, {**CONFIG, 'transaction': cfg})
