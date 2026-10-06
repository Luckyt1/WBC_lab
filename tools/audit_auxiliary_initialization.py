"""Verify cold starts and matched policy, optimizer recipe and environment."""
import hashlib
import json
from pathlib import Path

import torch


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite initial tensor: {key}')
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def audit(run, reference=None):
    run = Path(run)
    fresh = json.loads((run/'fresh_start.json').read_text())
    meta = json.loads((run/'auxiliary_supervision.json').read_text())
    saved = torch.load(run/'initial_policy.pt', map_location='cpu', weights_only=False)
    assert fresh['from_scratch'] and meta['from_scratch']
    assert fresh['checkpoint_loaded'] is None and fresh['teacher_checkpoint'] is None
    assert meta['checkpoint'] is None and meta['checkpoint_sha256'] is None
    assert saved['iter'] == fresh['iteration_before_learning'] == meta['iteration_before_learning'] == 0
    assert saved['infos']['before_first_update'] and saved['infos']['from_scratch']
    assert not saved['optimizer_state_dict']['state']
    assert fresh['optimizer_state_entries'] == meta['optimizer_state_entries'] == 0
    assert saved['model_state_dict']['actor.0.weight'].shape[1] == 840
    initial_std = saved['model_state_dict']['std']
    assert torch.equal(initial_std, torch.full_like(initial_std, .5))
    assert all(group['lr'] == 3.e-4 for group in saved['optimizer_state_dict']['param_groups'])
    if meta['variant'] == 'control':
        assert not any(k in saved for k in ('auxiliary_state_dict', 'auxiliary_optimizer_state_dict', 'auxiliary_cfg'))
        assert meta.get('auxiliary_cfg') is None
    else:
        assert saved['auxiliary_cfg']['mode'] == meta['variant']
        assert int(saved['auxiliary_state_dict']['updates']) == 0
        assert not saved['auxiliary_optimizer_state_dict']['state']
        tensor_state_hash(saved['auxiliary_state_dict'])
    result = dict(run_dir=str(run), variant=meta['variant'], from_scratch=True,
                  checkpoint_loaded=None, teacher_checkpoint=None, optimizer_state_entries=0,
                  policy_state_sha256=tensor_state_hash(saved['model_state_dict']),
                  **{key:meta[key] for key in ('task', 'seed', 'num_envs', 'reward_sha256',
                                               'command_sha256', 'environment_sha256', 'algorithm_sha256')})
    if reference is not None:
        for key in ('policy_state_sha256', 'task', 'seed', 'num_envs', 'reward_sha256',
                    'command_sha256', 'environment_sha256', 'algorithm_sha256'):
            assert result[key] == reference[key], f'Mismatched fresh-start {key}'
    return result
