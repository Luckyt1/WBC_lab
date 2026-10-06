"""Opt-in, training-only supervision of the unchanged actor trunk."""
import torch
from torch import nn
from torch.nn import functional as F


class ActorAuxiliary(nn.Module):
    def __init__(self, feature_dim=128, joint_dim=15, action_dim=29, mode='state',
                 warmup_updates=50, ramp_updates=100, seed=7341,
                 velocity_scale=(.5, .5, .3), joint_velocity_scale=3., angular_velocity_scale=1.):
        super().__init__()
        if mode not in ('state', 'dynamics') or warmup_updates < 0 or ramp_updates < 1:
            raise ValueError('Invalid auxiliary mode or warmup')
        if min(*velocity_scale, joint_velocity_scale, angular_velocity_scale) <= 0:
            raise ValueError('Auxiliary normalization scales must be positive')
        self.mode, self.joint_dim, self.action_dim = mode, joint_dim, action_dim
        self.warmup_updates, self.ramp_updates = warmup_updates, ramp_updates
        # Creating heads must not change rollout/minibatch RNG versus control.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.state_head = nn.Sequential(nn.Linear(feature_dim, 64), nn.ELU(), nn.Linear(64, 5))
            self.dynamics_head = (nn.Sequential(nn.Linear(feature_dim+action_dim,128),nn.ELU(),
                                                nn.Linear(128,joint_dim+3)) if mode == 'dynamics' else None)
        self.register_buffer('velocity_scale', torch.tensor(velocity_scale))
        self.register_buffer('joint_velocity_scale', torch.tensor(float(joint_velocity_scale)))
        self.register_buffer('angular_velocity_scale', torch.tensor(float(angular_velocity_scale)))
        self.register_buffer('updates', torch.tensor(0, dtype=torch.long))
        self.register_buffer('trunk_gain', torch.tensor(1.))

    @property
    def trunk_factor(self):
        return min(1., max(0., (self.updates.item()-self.warmup_updates)/self.ramp_updates))

    def predict(self, policy, observations, applied_action=None, training_gradient=False):
        feature = policy.actor[:-1](observations)
        if training_gradient:
            feature = feature.detach()+self.trunk_factor*self.trunk_gain.item()*(feature-feature.detach())
        state = self.state_head(feature)
        prediction = {'velocity': state[:, :3]*self.velocity_scale, 'contact_logits': state[:, 3:]}
        if self.dynamics_head is not None:
            if applied_action is None or applied_action.shape[-1] != self.action_dim:
                raise ValueError('Dynamics requires the actual normalized motor targets')
            nxt = self.dynamics_head(torch.cat((feature, applied_action.detach()), -1))
            prediction.update(joint_velocity=nxt[:, :self.joint_dim]*self.joint_velocity_scale,
                              angular_velocity=nxt[:, self.joint_dim:]*self.angular_velocity_scale)
        return prediction

    def loss(self, policy, observations, targets):
        pred = self.predict(policy, observations, targets['applied_action'], training_gradient=True)
        vel = F.smooth_l1_loss(pred['velocity']/self.velocity_scale,
                             targets['velocity'].detach()/self.velocity_scale)
        contact = F.binary_cross_entropy_with_logits(pred['contact_logits'], targets['contact'].detach())
        loss = .05*vel + .02*contact
        metrics = {'aux_velocity_huber': vel.detach(), 'aux_contact_bce': contact.detach(),
                   'aux_velocity_mae_m_s': (pred['velocity']-targets['velocity']).abs().mean().detach(),
                   'aux_trunk_factor': loss.new_tensor(self.trunk_factor)}
        if self.dynamics_head is not None:
            mask=targets['valid'].float().view(-1)
            def masked_huber(prediction, target, scale):
                per_sample=F.smooth_l1_loss(prediction/scale, target.detach()/scale, reduction='none').mean(-1)
                return (per_sample*mask).sum()/mask.sum().clamp(min=1)
            joint=masked_huber(pred['joint_velocity'],targets['next_joint_velocity'],self.joint_velocity_scale)
            angular=masked_huber(pred['angular_velocity'],targets['next_angular_velocity'],self.angular_velocity_scale)
            loss = loss+.025*joint+.025*angular
            metrics.update(aux_joint_velocity_huber=joint.detach(), aux_angular_velocity_huber=angular.detach(),
                           aux_dynamics_valid_fraction=mask.mean())
        metrics['aux_total']=loss.detach()
        return loss, metrics
