"""Continuous GL165 IK goals connected through an offline-validated transition graph.

The policy still controls only waist/legs. There is no return-to-home phase between
arm goals; quintic transitions have zero endpoint velocity and acceleration.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class ContinuousIkArmMotion:
    def __init__(self, bank, split, names, defaults, dt):
        bank = Path(bank)
        self.manifest = json.loads((bank / 'manifest.json').read_text())
        recipe = self.manifest['recipe']
        if recipe.get('motion_mode') != 'continuous_targets_v1' or list(names) != self.manifest['arm_joint_names']:
            raise ValueError('Expected a GL165 continuous IK bank and matching joint order')
        if split not in ('train', 'test') or abs(dt - .02) > 1e-8:
            raise ValueError('Continuous IK replay requires a declared split and 50 Hz control')
        path = bank / (split + '.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest['bank_sha256'][split]:
            raise ValueError('Continuous IK bank checksum mismatch')
        self.dt, self.split, self.device = dt, split, defaults.device
        with np.load(path, allow_pickle=False) as data:
            self.ids = list(data['ids'])
            self.group_names = sorted(set(data['groups']))
            self.groups = torch.tensor([self.group_names.index(g) for g in data['groups']], device=self.device)
            self.goals = torch.tensor(data['paths'][:, 0, -1].copy(), device=self.device)
            self.neighbors = torch.tensor(data['neighbors'].copy(), dtype=torch.long, device=self.device)
            self.edge_durations = torch.tensor(data['edge_min_durations'].copy(), device=self.device)
            self.home_durations = torch.tensor(data['min_durations'][:, 0].copy(), device=self.device)
            self.overhead = torch.tensor(data['overhead'].copy(), dtype=torch.bool, device=self.device)
        if (self.neighbors.ndim != 2 or self.neighbors.shape[0] != len(self.ids)
                or self.edge_durations.shape != self.neighbors.shape
                or self.neighbors.min() < 0 or self.neighbors.max() >= len(self.ids)):
            raise ValueError('Invalid continuous IK transition graph')
        targets = {t['id']: t for t in self.manifest['targets']}
        if any(targets[i]['split'] != split for i in self.ids):
            raise ValueError('Train/test leakage in continuous IK bank')
        self.goal_positions = torch.tensor([targets[i]['positions'] for i in self.ids], device=self.device)
        self.goal_orientations = torch.tensor([targets[i]['orientations_wxyz'] for i in self.ids], device=self.device)
        self.region_names = sorted({targets[i]['region'] for i in self.ids})
        self.regions = torch.tensor([self.region_names.index(targets[i]['region']) for i in self.ids], device=self.device)
        # Balance left/right/both inside each region. Preserve the requested
        # overhead probability even when a region contains more stored targets.
        probability = float(recipe['overhead_fraction_of_active'])
        self.region_probabilities = recipe.get('region_probabilities', {'low': 1 - probability, 'overhead': probability})
        if (set(self.region_probabilities) != set(self.region_names)
                or any(not np.isfinite(p) or p <= 0 for p in self.region_probabilities.values())
                or abs(sum(self.region_probabilities.values()) - 1) > 1e-6):
            raise ValueError('Region sampling probabilities must match the bank and sum to one')
        self.weights = torch.zeros(len(self.ids), device=self.device)
        self.edge_weights = torch.zeros_like(self.edge_durations)
        for region, name in enumerate(self.region_names):
            mass = self.region_probabilities[name]
            for group in range(len(self.group_names)):
                mask = (self.regions == region) & (self.groups == group)
                edge_mask = mask[self.neighbors]
                if not bool(mask.any()) or not bool(edge_mask.any(dim=1).all()):
                    raise ValueError('Every source goal must connect to each arm group and height region')
                self.weights[mask] = mass / len(self.group_names) / mask.sum()
                self.edge_weights += edge_mask * (mass / len(self.group_names) / edge_mask.sum(dim=1, keepdim=True))
        n = len(defaults)
        self.defaults = defaults.clone()
        self.target = defaults.clone()
        self.requested_target = defaults.clone()
        self.start = defaults.clone()
        self.index = torch.zeros(n, dtype=torch.long, device=self.device)
        self.elapsed = torch.zeros(n, device=self.device)
        self.duration = torch.ones(n, device=self.device)
        self.phase = torch.ones(n, dtype=torch.long, device=self.device)
        self.neutral = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.sample_counts = torch.zeros(len(self.ids), device=self.device)
        self.completed_cycles = torch.zeros((), device=self.device)
        self.metrics = {}

    def _duration(self, minimum):
        lo, hi = self.manifest['recipe']['duration_multiplier']
        return minimum * (lo + torch.rand_like(minimum) * (hi - lo))

    def reset(self, ids, actual):
        if len(ids) == 0:
            return
        chosen = torch.multinomial(self.weights, len(ids), replacement=True)
        self.index[ids] = chosen
        self.sample_counts.scatter_add_(0, chosen, torch.ones(len(ids), device=self.device))
        self.start[ids] = self.target[ids] = self.requested_target[ids] = actual
        self.elapsed[ids] = 0.
        self.duration[ids] = self._duration(self.home_durations[chosen])
        self.phase[ids] = 1

    def step(self):
        done = self.elapsed >= self.duration
        ids = done.nonzero().flatten()
        if len(ids):
            current = self.index[ids]
            candidates = self.neighbors[current]
            # Viewer-only filtering changes which groups/regions are visible;
            # all selected edges still come from the validated graph.
            weights = self.edge_weights[current] * (self.weights[candidates] > 0)
            if not bool((weights.sum(dim=1) > 0).all()):
                raise ValueError('Selected preview region has no valid outgoing transition')
            choice = torch.multinomial(weights, 1).squeeze(1)
            chosen = candidates.gather(1, choice[:, None]).squeeze(1)
            self.start[ids] = self.target[ids]
            self.index[ids] = chosen
            self.duration[ids] = self._duration(self.edge_durations[current, choice])
            self.elapsed[ids] = 0.
            self.sample_counts.scatter_add_(0, chosen, torch.ones(len(ids), device=self.device))
            self.completed_cycles += len(ids)
        self.elapsed += self.dt
        fraction = (self.elapsed / self.duration).clamp(0., 1.)
        blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
        self.requested_target.copy_(self.start + blend[:, None] * (self.goals[self.index] - self.start))
        # A single endpoint sample enables arrival-error logging without a hold.
        self.phase.copy_(torch.where(fraction >= 1, 2, 1))
        counts = self.sample_counts.sum().clamp(min=1)
        self.metrics = {
            'IK/neutral_fraction': self.neutral.float().mean(),
            'IK/completed_cycles': self.completed_cycles.clone(),
            'IK/unique_target_coverage': (self.sample_counts > 0).float().mean(),
            'IK/transition_fraction': (self.phase == 1).float().mean(),
            'IK/endpoint_fraction': (self.phase == 2).float().mean(),
        }
        for region, name in enumerate(self.region_names):
            mask = self.regions == region
            self.metrics[f'IK/{name}_target_fraction'] = mask[self.index].float().mean()
            self.metrics[f'IK/{name}_sample_fraction'] = self.sample_counts[mask].sum() / counts
        for group, name in enumerate(self.group_names):
            self.metrics['IK/group_' + name + '_fraction'] = (self.groups[self.index] == group).float().mean()
        return self.requested_target
