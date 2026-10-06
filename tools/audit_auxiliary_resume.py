"""Require exact learned-state restoration for dynamics or plain-PPO continuation."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml


STATE_KEYS = ('model_state_dict', 'optimizer_state_dict', 'auxiliary_state_dict',
              'auxiliary_optimizer_state_dict', 'auxiliary_cfg')


def completed_updates(saved, mode='dynamics'):
    if mode == 'control':
        if any(key in saved for key in STATE_KEYS[2:]):
            raise ValueError('Plain PPO continuation cannot load auxiliary state')
        if any(key not in saved for key in ('model_state_dict', 'optimizer_state_dict', 'iter')):
            raise ValueError('Plain PPO continuation requires policy, optimizer and iteration')
        if not saved['optimizer_state_dict']['state'] or int(saved['iter']) < 0:
            raise ValueError('Plain PPO continuation requires a trained optimizer and iteration')
        if (saved.get('infos') or {}).get('before_first_update'):
            raise ValueError('An initialization snapshot is not a completed training checkpoint')
        return int(saved['iter']) + 1
    if mode != 'dynamics':
        raise ValueError('Unsupported audited training mode')
    if any(key not in saved for key in STATE_KEYS):
        raise ValueError('Continuation requires policy, auxiliary heads and both optimizers')
    if saved['auxiliary_cfg']['mode'] != 'dynamics':
        raise ValueError('Continuation requires a dynamics checkpoint')
    count = int(saved['auxiliary_state_dict']['updates'])
    if count != int(saved['iter']) + 1:
        raise ValueError('Checkpoint iteration and completed auxiliary updates disagree')
    if not all(saved[key]['state'] for key in ('optimizer_state_dict', 'auxiliary_optimizer_state_dict')):
        raise ValueError('Continuation requires trained optimizer states')
    return count


def assert_equal(expected, actual, path='state'):
    if isinstance(expected, torch.Tensor):
        if (not isinstance(actual, torch.Tensor) or expected.dtype != actual.dtype
                or expected.shape != actual.shape or not torch.isfinite(actual).all()
                or not torch.equal(expected.cpu(), actual.detach().cpu())):
            raise ValueError(f'Restored tensor mismatch: {path}')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f'Restored keys mismatch: {path}')
        for key in expected:
            assert_equal(expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            raise ValueError(f'Restored length mismatch: {path}')
        for index, (a, b) in enumerate(zip(expected, actual)):
            assert_equal(a, b, f'{path}.{index}')
    elif expected != actual:
        raise ValueError(f'Restored value mismatch: {path}')


def audit_restored(source, restored, learning_rate, mode='dynamics'):
    count = completed_updates(source, mode)
    if restored['iter'] != source['iter']:
        raise ValueError('Restored iteration mismatch')
    if mode == 'control' and any(key in restored for key in STATE_KEYS[2:]):
        raise ValueError('Plain PPO restored unexpected auxiliary state')
    for key in (STATE_KEYS[:2] if mode == 'control' else STATE_KEYS):
        assert_equal(source[key], restored[key], key)
    if any(group['lr'] != learning_rate for group in source['optimizer_state_dict']['param_groups']):
        raise ValueError('PPO learning-rate scheduler was not restored')
    return count


def expected_agent_without_symmetry(agent):
    """One explicitly requested change; never relax the remaining recipe audit."""
    result = deepcopy(agent)
    symmetry = result['algorithm'].get('symmetry_cfg')
    if not isinstance(symmetry, dict) or not any(
            symmetry.get(key) in (True, 'true') for key in ('use_data_augmentation', 'use_mirror_loss')):
        raise ValueError('Disabling symmetry requires a checkpoint trained with active symmetry')
    # yaml.BaseLoader represents YAML null as this string.
    result['algorithm']['symmetry_cfg'] = 'null'
    return result


def verify_recipe(source_run, resumed_run, target_height=None, curriculum_stage=None,
                  disable_symmetry=False):
    def read(run, filename):
        return yaml.load((Path(run)/'params'/filename).read_text(), Loader=yaml.BaseLoader)
    before, after = read(source_run, 'env.yaml'), read(resumed_run, 'env.yaml')
    if target_height is not None:
        from tools.fixed_pelvis_height import expected_height_recipe
        before = expected_height_recipe(before, target_height)
    if curriculum_stage is not None:
        from tools.teleop_curriculum import expected_recipe
        before = expected_recipe(before, curriculum_stage)
    assert_equal(before, after, 'environment')
    old_agent, new_agent = read(source_run, 'agent.yaml'), read(resumed_run, 'agent.yaml')
    if disable_symmetry:
        old_agent = expected_agent_without_symmetry(old_agent)
    for key in ('policy', 'algorithm', 'seed', 'num_steps_per_env', 'empirical_normalization'):
        assert_equal(old_agent[key], new_agent[key], f'agent.{key}')
    return hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()


def record_resume(runner, checkpoint, run, target_height=None, curriculum_stage=None,
                  disable_symmetry=False, mode='dynamics'):
    checkpoint, run = Path(checkpoint), Path(run)
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    restored = dict(model_state_dict=runner.alg.policy.state_dict(),
                    optimizer_state_dict=runner.alg.optimizer.state_dict(),
                    iter=runner.current_learning_iteration)
    if mode == 'control':
        if any(getattr(runner.alg, key, None) is not None
               for key in ('auxiliary', 'auxiliary_optimizer', 'auxiliary_cfg')):
            raise ValueError('Plain PPO must have no auxiliary head, optimizer or configuration')
    else:
        restored.update(auxiliary_state_dict=runner.alg.auxiliary.state_dict(),
                        auxiliary_optimizer_state_dict=runner.alg.auxiliary_optimizer.state_dict(),
                        auxiliary_cfg=runner.alg.auxiliary_cfg)
    count = audit_restored(source, restored, runner.alg.learning_rate, mode)
    recipe_hash = verify_recipe(checkpoint.parent, run, target_height=target_height,
                                curriculum_stage=curriculum_stage, disable_symmetry=disable_symmetry)
    # Checkpoints store the last completed zero-based iteration.
    runner.current_learning_iteration = count
    if hasattr(runner.env, 'set_training_iteration'):
        runner.env.set_training_iteration(count, runner.num_steps_per_env)
    restored.update(iter=count, infos={'before_first_update': True, 'from_scratch': False})
    torch.save(restored, run/'resume_initial_policy.pt')
    evidence = dict(from_scratch=False, checkpoint_loaded=str(checkpoint.resolve()),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    source_iteration=source['iter'], iteration_before_learning=count,
                    completed_updates_before_resume=count, exact_learned_state_restored=True,
                    optimizer_state_entries=len(source['optimizer_state_dict']['state']),
                    auxiliary_supervision=mode,
                    auxiliary_optimizer_state_entries=(len(source['auxiliary_optimizer_state_dict']['state'])
                                                       if mode == 'dynamics' else 0),
                    auxiliary_updates=(int(source['auxiliary_state_dict']['updates'])
                                       if mode == 'dynamics' else None),
                    learning_rate=runner.alg.learning_rate, environment_sha256=recipe_hash,
                    simulator_state_restored=False,
                    initial_checkpoint=str((run/'resume_initial_policy.pt').resolve()))
    if disable_symmetry:
        if runner.alg.symmetry is not None:
            raise ValueError('Symmetry must be fully disabled in the live PPO instance')
        evidence['symmetry_change'] = dict(use_data_augmentation=False, use_mirror_loss=False,
                                           only_symmetry_configuration_changed=True,
                                           policy_and_both_optimizers_preserved=True)
    if target_height is not None:
        from tools.fixed_pelvis_height import checkpoint_height
        evidence['height_recipe_change'] = dict(from_m=checkpoint_height(checkpoint),
                                               to_m=float(target_height),
                                               only_command_height_changed=curriculum_stage is None)
    if curriculum_stage is not None:
        from tools.teleop_curriculum import checkpoint_stage
        evidence['teleop_curriculum'] = dict(from_stage=checkpoint_stage(checkpoint),
                                            to_stage=curriculum_stage, only_declared_difficulty_fields_changed=True)
    (run/'resume_start.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence
