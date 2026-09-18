"""Optional local policy sensitivity penalty on previous-action observations.

Central finite differences use fixed, privately seeded Gaussian directions and
fresh shuffled rollout observations. No environment/action/observation is changed.
This is a local regularizer, not a stability certificate.
"""
import math

import torch


class FeedbackSmoothness:
    def __init__(self, config, observation_dim, action_dim, device):
        self.weight = float(config['weight'])
        self.sigma = float(config['sigma'])
        self.clip = float(config.get('observation_clip', 100.))
        self.indices = list(config['action_indices'])
        self.max_samples = int(config.get('max_samples', 512))
        if not all(math.isfinite(x) for x in (self.weight, self.sigma, self.clip)):
            raise ValueError('Feedback smoothness requires finite settings')
        if self.weight <= 0 or self.sigma <= 0 or self.clip <= 0 or self.max_samples < 1:
            raise ValueError('Invalid feedback smoothness settings')
        if (len(self.indices) != action_dim or len(set(self.indices)) != action_dim or
                min(self.indices) < 0 or max(self.indices) >= observation_dim):
            raise ValueError('Feedback smoothness action indices do not match observations')
        # Do not advance PPO/environment RNG. Fixed probes pair with freshly
        # shuffled rollout rows; no extra generator state is needed on resume.
        generator = torch.Generator(device=device).manual_seed(int(config.get('seed', 713)))
        self.directions = torch.randn(self.max_samples, action_dim, device=device, generator=generator)

    def loss(self, policy, observations):
        selected = observations[:self.max_samples].detach()
        positive, negative = selected.clone(), selected.clone()
        delta = self.sigma * self.directions[:len(selected)]
        positive[:, self.indices] = (positive[:, self.indices] + delta).clamp(-self.clip, self.clip)
        negative[:, self.indices] = (negative[:, self.indices] - delta).clamp(-self.clip, self.clip)
        difference = (policy.act_inference(positive) - policy.act_inference(negative)) / (2 * self.sigma)
        return difference.square().mean()
