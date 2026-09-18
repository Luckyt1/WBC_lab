"""Opt-in whole-rollout PPO rollback for the controlled recovery experiment.

Validation uses the collected rollout and a separately saved feedback probe bank.
It changes update acceptance only, never actions/observations sent to the robot.
"""
from copy import deepcopy
from types import MethodType
import math

import torch


@torch.no_grad()
def feedback_peaks(policy, observations, action_indices, steps, clip):
    current = observations.clone()
    peaks = torch.zeros(current.shape[0], device=current.device)
    for _ in range(steps):
        action = policy.act_inference(current)
        peaks = torch.maximum(peaks, action.abs().amax(-1))
        current[:, action_indices] = action.clamp(-clip, clip)
    return peaks


def install_transaction(alg, config):
    if alg.schedule != 'fixed':
        raise ValueError('The transaction experiment requires a fixed learning rate')
    cfg = dict(config)
    for key in ('post_mean_kl', 'post_max_kl', 'feedback_factor', 'feedback_floor'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid transaction option {key}')
    bank = torch.load(cfg['feedback_bank'], map_location=alg.device, weights_only=True)
    obs = bank['observations']
    ids = bank['action_indices']
    if obs.ndim != 2 or obs.shape[0] == 0 or obs.shape[1] != alg.storage.observations.shape[-1]:
        raise ValueError('Feedback bank does not match policy observations')
    if len(ids) != alg.storage.actions.shape[-1] or len(set(ids)) != len(ids):
        raise ValueError('Feedback bank does not match policy actions')
    if min(ids) < 0 or max(ids) >= obs.shape[-1]:
        raise ValueError('Feedback bank has invalid action indices')
    reference = bank['reference_peaks']
    if reference.shape != (obs.shape[0],) or not torch.isfinite(obs).all() or not torch.isfinite(reference).all():
        raise ValueError('Feedback bank has invalid values')
    if bank['steps'] < 1 or bank['clip'] <= 0:
        raise ValueError('Invalid feedback probe settings')
    alg.transaction_config = cfg
    alg.feedback_bank = bank
    alg.feedback_limits = torch.maximum(reference * cfg['feedback_factor'],
                                       torch.full_like(reference, cfg['feedback_floor']))
    alg.transaction_inner_update = alg.update
    alg.update = MethodType(_transaction_update, alg)


@torch.no_grad()
def _post_update_kl(alg):
    obs = alg.storage.observations.flatten(0, 1)
    old_mean = alg.storage.mu.flatten(0, 1)
    old_std = alg.storage.sigma.flatten(0, 1)
    std = alg.policy.std if alg.policy.noise_std_type == 'scalar' else alg.policy.log_std.exp()
    total = torch.zeros((), device=obs.device)
    maximum = torch.zeros_like(total)
    for start in range(0, len(obs), 16384):
        end = start + 16384
        mean = alg.policy.act_inference(obs[start:end])
        kl = (torch.log(std / old_std[start:end]) +
              (old_std[start:end].square() + (old_mean[start:end] - mean).square()) /
              (2 * std.square()) - .5).sum(-1)
        total += kl.sum()
        maximum = torch.maximum(maximum, kl.max())
    return float(total / len(obs)), float(maximum)


def _transaction_update(self):
    before = deepcopy(self.policy.state_dict())
    optimizer_before = deepcopy(self.optimizer.state_dict())
    lr_before = self.learning_rate
    result = self.transaction_inner_update()
    cfg, bank = self.transaction_config, self.feedback_bank
    mean_kl, max_kl = _post_update_kl(self)
    peaks = feedback_peaks(self.policy, bank['observations'], bank['action_indices'],
                           bank['steps'], bank['clip'])
    kl_bad = (not math.isfinite(mean_kl) or not math.isfinite(max_kl) or
              mean_kl > cfg['post_mean_kl'] or max_kl > cfg['post_max_kl'])
    feedback_bad = not bool(torch.isfinite(peaks).all()) or bool((peaks > self.feedback_limits).any())
    reverted = kl_bad or feedback_bad
    attempted = result['ppo_updates']
    if reverted:
        self.policy.load_state_dict(before)
        self.optimizer.load_state_dict(optimizer_before)
        self.optimizer.zero_grad(set_to_none=True)
        self.learning_rate = lr_before
        result['ppo_updates'] = 0.
    result.update(
        ppo_attempted_updates=attempted, transaction_reverted=float(reverted),
        transaction_kl_rejected=float(kl_bad), transaction_feedback_rejected=float(feedback_bad),
        transaction_post_mean_kl=mean_kl, transaction_post_max_kl=max_kl,
        transaction_feedback_peak=float(peaks.max()),
        transaction_feedback_limit_ratio=float((peaks / self.feedback_limits).max()),
    )
    return result
