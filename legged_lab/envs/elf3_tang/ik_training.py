"""Explicit procedural IK recipe, shared by training and held-out evaluation."""
import hashlib
import json
from pathlib import Path


def configure_ik_training(cfg, bank, split='train'):
    if split not in ('train', 'test'):
        raise ValueError('IK split must be train or test')
    bank = Path(bank).resolve(strict=True)
    manifest = json.loads((bank/'manifest.json').read_text())
    digest = hashlib.sha256((bank/(split+'.npz')).read_bytes()).hexdigest()
    if digest != manifest['bank_sha256'][split]:
        raise ValueError('IK trajectory bank digest mismatch')
    arm = cfg.arm_motion
    arm.source = 'ik_elf3'
    arm.ik_bank_path = str(bank)
    arm.ik_split = split
    arm.ik_bank_sha256 = digest
    arm.collision_guard = arm.guard_fixed_targets = True
    arm.guard_max_velocity = manifest['recipe']['max_velocity']
    arm.guard_margin = manifest['recipe']['guard_margin']
    arm.scale = 1.
    arm.amass_scale_end_iteration = None
    cfg.scene.max_episode_length_s = manifest['recipe'].get('max_episode_s',24.)
    cfg.ik_recipe = f"reach_hold_return_v{manifest['version']}"
    return dict(recipe=cfg.ik_recipe, path=str(bank), split=split, sha256=digest,
                counts=manifest['counts'], execution=manifest['recipe'],
                limitations=manifest['limitations'])


def checkpoint_bank(checkpoint):
    import yaml
    cfg = yaml.load((Path(checkpoint).parent/'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    return cfg.get('arm_motion', {}).get('ik_bank_path') if cfg.get('ik_recipe') else None
