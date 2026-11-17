"""Check continuous GL165 replay across transitions, partial resets and both splits."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from legged_lab.assets.gl165.model import ARM_JOINT_NAMES, DEFAULT_POSITIONS, asset_dir, ik_bank_dir
from tools.check_gl165_resources import check_resources


@torch.inference_mode()
def verify(bank, asset, steps=5000, batch=512):
    torch.set_num_threads(1)
    torch.manual_seed(20261114)
    report = check_resources(asset, bank)
    spec = importlib.util.spec_from_file_location('gl165_continuous_motion', ROOT / 'legged_lab/envs/gl165/arm_motion.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for split in ('train', 'test'):
        defaults = torch.tensor([[DEFAULT_POSITIONS[n] for n in ARM_JOINT_NAMES]]).repeat(batch, 1)
        arms = module.ContinuousIkArmMotion(bank, split, ARM_JOINT_NAMES, defaults, .02)
        source = Path(arms.manifest['source_bank'])
        with np.load(source / (split + '.npz')) as original, np.load(bank / (split + '.npz')) as extended:
            assert np.array_equal(original['paths'], extended['paths'][:len(original['paths'])])
            assert np.array_equal(original['ids'], extended['ids'][:len(original['ids'])])
        arms.reset(torch.arange(batch), defaults)
        velocity = torch.zeros_like(defaults)
        peak_velocity = peak_acceleration = peak_boundary_step = 0.
        transition_count = 0
        for step in range(steps):
            reset_ids = torch.arange(0, batch, 3) if step == steps // 2 else torch.empty(0, dtype=torch.long)
            if len(reset_ids):
                # Actual reset poses can vary slightly; unrelated environments
                # must retain their current trajectories without a new phase.
                retained = arms.target.clone()
                actual = defaults[reset_ids] + .003
                arms.reset(reset_ids, actual)
                assert torch.equal(arms.start[reset_ids], actual)
                mask = torch.ones(batch, dtype=torch.bool)
                mask[reset_ids] = False
                assert torch.equal(arms.target[mask], retained[mask])
                velocity[reset_ids] = 0.
            previous, index = arms.target.clone(), arms.index.clone()
            changing = arms.elapsed >= arms.duration
            request = arms.step()
            assert torch.isfinite(request).all()
            assert ((arms.phase == 1) | (arms.phase == 2)).all() and not arms.neutral.any()
            if changing.any():
                assert (arms.neighbors[index[changing]] == arms.index[changing, None]).any(dim=1).all()
                assert (index[changing] != arms.index[changing]).all()
                assert torch.equal(arms.start[changing], previous[changing])
                assert torch.allclose(previous[changing], arms.goals[index[changing]], atol=2e-6)
                peak_boundary_step = max(peak_boundary_step, float((request - previous)[changing].abs().max()))
                transition_count += int(changing.sum())
            new_velocity = (request - previous) / .02
            acceleration = (new_velocity - velocity) / .02
            peak_velocity = max(peak_velocity, float(new_velocity.abs().max()))
            peak_acceleration = max(peak_acceleration, float(acceleration.abs().max()))
            assert peak_velocity <= 1.21 and peak_acceleration <= 3.05
            arms.target.copy_(request)
            velocity = new_velocity
        region_samples = {}
        for region, name in enumerate(arms.region_names):
            sampled = float(arms.sample_counts[arms.regions == region].sum() / arms.sample_counts.sum())
            assert abs(sampled - arms.region_probabilities[name]) < .025
            region_samples[name] = sampled
        assert (arms.sample_counts > 0).all() and transition_count > batch
        report[split].update(replayed_transitions=transition_count,
            peak_reference_velocity_rad_s=peak_velocity, peak_reference_acceleration_rad_s2=peak_acceleration,
            max_transition_boundary_step_rad=peak_boundary_step,
            region_sample_fractions=region_samples, coverage=1., partial_reset_passed=True, original_paths_preserved=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, default=ik_bank_dir())
    parser.add_argument('--asset', type=Path, default=asset_dir())
    parser.add_argument('--output', type=Path, default=ROOT / 'output/gl165_continuous_validation/replay.json')
    args = parser.parse_args()
    result = verify(args.bank, args.asset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('GL165_CONTINUOUS_REPLAY_VERIFIED', json.dumps(result), flush=True)
