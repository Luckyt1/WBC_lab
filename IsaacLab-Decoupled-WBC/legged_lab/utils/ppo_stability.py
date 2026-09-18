"""Opt-in, project-local PPO update for single-GPU feed-forward recovery runs.

Keeps the existing policy, Adam state, rollout storage and runner checkpoint
format. Only the update method is replaced; unsupported PPO variants fail loudly.
The default objective follows installed RSL-RL PPO (BSD-3-Clause), with bounded
adaptive LR and a pre-minibatch KL early stop. Optional previous-action sensitivity
regularization and whole-update rollback are separately enabled by configuration.
"""

import math
from types import MethodType
import torch


def install_ppo_stability(alg, config, reset_learning_rate=False):
    if alg.policy.is_recurrent or alg.symmetry or alg.rnd or alg.is_multi_gpu:
        raise ValueError('PPO stability supports single-GPU feed-forward PPO without RND/symmetry')
    cfg = dict(config)
    if not (0 < cfg['min_learning_rate'] <= cfg['initial_learning_rate'] <= cfg['max_learning_rate']
            and math.isfinite(cfg['max_learning_rate']) and 0 < cfg['max_mean_kl'] < float('inf')):
        raise ValueError('Invalid PPO stability bounds')
    alg.stability_config = cfg
    lr = cfg['initial_learning_rate'] if reset_learning_rate else alg.optimizer.param_groups[0]['lr']
    _set_lr(alg, lr)
    if cfg.get('feedback_smoothness'):
        from legged_lab.utils.feedback_smoothness import FeedbackSmoothness
        alg.feedback_smoothness = FeedbackSmoothness(
            cfg['feedback_smoothness'], alg.storage.observations.shape[-1],
            alg.storage.actions.shape[-1], alg.device)
    alg.update = MethodType(_guarded_update, alg)
    if cfg.get('transaction'):
        from legged_lab.utils.ppo_transaction import install_transaction
        install_transaction(alg, cfg['transaction'])


def _set_lr(alg, value):
    cfg = alg.stability_config
    alg.learning_rate = max(cfg['min_learning_rate'], min(cfg['max_learning_rate'], value))
    for group in alg.optimizer.param_groups:
        group['lr'] = alg.learning_rate


def _guarded_update(self):
    totals = {'value_function': 0., 'surrogate': 0., 'entropy': 0.}
    smoothness = getattr(self, 'feedback_smoothness', None)
    if smoothness is not None:
        totals.update(feedback_smoothness_loss=0., feedback_smoothness_weighted_loss=0.)
    updates, evaluated, peak_kl, early_stop = 0, 0, 0., False
    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for (obs, critic_obs, actions, old_value, advantage, returns, old_log_prob,
         old_mean, old_std, hidden, masks, _) in generator:
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        self.policy.act(obs, masks=masks, hidden_states=hidden[0])
        log_prob = self.policy.get_actions_log_prob(actions)
        value = self.policy.evaluate(critic_obs, masks=masks, hidden_states=hidden[1])
        mean, std, entropy = self.policy.action_mean, self.policy.action_std, self.policy.entropy
        with torch.no_grad():
            kl = (torch.log(std / old_std) +
                  (old_std.square() + (old_mean - mean).square()) / (2 * std.square()) - .5).sum(-1).mean()
            mean_kl = float(kl)
        evaluated += 1
        if not math.isfinite(mean_kl):
            raise FloatingPointError('Nonfinite PPO KL; refusing another optimizer step')
        peak_kl = max(peak_kl, mean_kl)
        if self.schedule == 'adaptive' and self.desired_kl is not None:
            if mean_kl > self.desired_kl * 2:
                _set_lr(self, self.learning_rate / 1.5)
            elif 0 < mean_kl < self.desired_kl / 2:
                _set_lr(self, self.learning_rate * 1.5)
        if mean_kl > self.stability_config['max_mean_kl']:
            early_stop = True
            break
        ratio = (log_prob - old_log_prob.squeeze(-1)).exp()
        surrogate = torch.maximum(-advantage.squeeze(-1) * ratio,
                                  -advantage.squeeze(-1) * ratio.clamp(1-self.clip_param, 1+self.clip_param)).mean()
        if self.use_clipped_value_loss:
            clipped = old_value + (value - old_value).clamp(-self.clip_param, self.clip_param)
            value_loss = torch.maximum((value - returns).square(), (clipped - returns).square()).mean()
        else:
            value_loss = (value - returns).square().mean()
        loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
        if smoothness is not None:
            smoothness_loss = smoothness.loss(self.policy, obs)
            loss = loss + smoothness.weight * smoothness_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite PPO loss; refusing optimizer step')
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        updates += 1
        totals['value_function'] += value_loss.item()
        totals['surrogate'] += surrogate.item()
        totals['entropy'] += entropy.mean().item()
        if smoothness is not None:
            totals['feedback_smoothness_loss'] += smoothness_loss.item()
            totals['feedback_smoothness_weighted_loss'] += smoothness.weight * smoothness_loss.item()
    self.storage.clear()
    return {**{key: value / max(updates, 1) for key, value in totals.items()},
            'ppo_updates': float(updates), 'ppo_batches_evaluated': float(evaluated),
            'ppo_early_stop': float(early_stop), 'ppo_mean_kl_max': peak_kl}
