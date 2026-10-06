"""CPU-testable completed-swing scoring; no simulator or observation changes."""
import torch

from .commands import rotate_inverse


def sole_clearance(positions, quaternions, hull_vertices, ground_z):
    """Exact minimum support height of the supplied convex collision hulls."""
    up = torch.zeros_like(positions)
    up[..., 2] = 1.
    local_up = rotate_inverse(quaternions, up)
    support = torch.einsum('nfk,fvk->nfv', local_up, hull_vertices).amin(-1)
    return positions[..., 2] + support - ground_z[:, None]


class SwingTracker:
    def __init__(self, num_envs, device, dt, min_air=.12, max_air=.35,
                 min_clearance=.02, target_clearance=.03):
        if not (0 < dt < min_air < max_air and 0 < min_clearance <= target_clearance):
            raise ValueError('Invalid swing duration or clearance range')
        self.dt, self.min_air, self.max_air = dt, min_air, max_air
        self.min_clearance, self.target_clearance = min_clearance, target_clearance
        self.air = torch.zeros(num_envs, 2, device=device)
        self.peak = torch.zeros_like(self.air)
        self.supported = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.previous_contact = torch.ones_like(self.supported)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last_foot = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.last_metrics = {}

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.air[ids] = 0
        self.peak[ids] = 0
        self.supported[ids] = False
        self.previous_contact[ids] = True
        self.initialized[ids] = False
        self.last_foot[ids] = -1

    def step(self, contact, clearance, active):
        """Return a reward *rate*: manager multiplies by dt once.

        Positive event reward requires an alternating, completed, supported
        swing in the duration/clearance band. Short flights, repeated-foot
        swings, prolonged hovering and hopping do not earn an event bonus.
        """
        # A standing segment or an episode reset breaks the event history.
        self.reset((~active).nonzero(as_tuple=False).flatten())
        ready = self.initialized & active
        lifted = self.previous_contact & ~contact & ready[:, None]
        airborne = ~contact & ready[:, None]
        landed = ~self.previous_contact & contact & ready[:, None]
        other_contact = contact.flip(-1)
        self.supported = torch.where(lifted, other_contact, self.supported)
        self.supported &= ~airborne | other_contact
        self.air = torch.where(airborne, self.air+self.dt, self.air)
        self.peak = torch.where(airborne, torch.maximum(self.peak, clearance.clamp(min=0)), self.peak)

        valid = (landed & other_contact & self.supported
                 & (self.air >= self.min_air-1e-6) & (self.air <= self.max_air+1e-6)
                 & (self.peak >= self.min_clearance-1e-6))
        alternating = (self.last_foot[:, None] != torch.arange(2, device=self.air.device)[None])
        qualified = valid & alternating
        # Event impulses have fixed magnitude, independent of policy frequency.
        event = .35*qualified.float().sum(-1)
        shortfall = (1-self.air/self.min_air).clamp(0, 1)
        event -= .08*(shortfall*landed).sum(-1)
        event -= .10*(valid & ~alternating).float().sum(-1)
        self.last_foot = torch.where(qualified.any(-1), qualified.long().argmax(-1), self.last_foot)

        # Dense lifting signal before the first successful full swing; only
        # while the other foot supports the body and the duration is bounded.
        dense_mask = airborne & other_contact & self.supported & (self.air <= self.max_air)
        lift = (clearance/self.target_clearance).clamp(0, 1)
        dense = .4*(lift*dense_mask).sum(-1)
        dense -= .5*(airborne & (self.air > self.max_air)).float().sum(-1)
        dense -= ((clearance-.06).clamp(min=0)/.04).clamp(max=1).mul(airborne).sum(-1)
        value = (event/self.dt+dense)*active
        self.last_metrics = {
            'qualified': qualified.float().sum(-1),
            'touchdowns': landed.float().sum(-1),
            'air_time': self.air.clone(),
            'clearance': clearance.clone(),
        }
        self.air = torch.where(contact, torch.zeros_like(self.air), self.air)
        self.peak = torch.where(contact, torch.zeros_like(self.peak), self.peak)
        self.previous_contact.copy_(contact)
        self.initialized.copy_(active)
        return value
