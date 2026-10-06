"""Export a frozen ELF3 840/15 ELU actor on CPU without starting Isaac Sim."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    config = yaml.load((source.parent/'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    if (config['empirical_normalization'].lower() != 'false'
            or config['policy']['activation'] != 'elu'
            or config['policy']['class_name'] != 'ActorCritic'):
        raise ValueError('Unsupported normalization or actor; export requires adaptation')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    weights = {k.removeprefix('actor.'): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('actor.')}
    widths = [840, *map(int, config['policy']['actor_hidden_dims']), 15]
    layers = []
    for i, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
        layers.append(torch.nn.Linear(inputs, outputs))
        if i < len(widths)-2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).eval()
    actor.load_state_dict(weights, strict=True)
    if not all(torch.isfinite(x).all() for x in actor.parameters()):
        raise ValueError('Non-finite actor weights')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.pending.onnx')
    with torch.inference_mode():
        torch.onnx.export(actor, torch.zeros(1, 840), str(temporary), input_names=['obs'],
                          output_names=['actions'], opset_version=11, dynamo=False)
    model = onnx.load(temporary)
    from fixed_pelvis_height import checkpoint_height
    height = checkpoint_height(source, require_fixed=False)
    for key, value in {'checkpoint': str(source), 'checkpoint_sha256': digest,
                       'history_frames': '10', 'frame_observation_size': '84',
                       'history_order': 'oldest_first', 'control_hz': '50',
                       'actor_frame': 'waist_z_link', 'height_command': str(height)}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, temporary)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(temporary), sess_options=options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(42)
    maximum = 0.
    with torch.inference_mode():
        for scale in (0., .1, 1., 10.):
            for _ in range(8):
                obs = rng.normal(size=(1, 840)).astype(np.float32) * scale
                expected = actor(torch.from_numpy(obs)).numpy()
                actual = session.run(['actions'], {'obs': obs})[0]
                np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=2e-5)
                maximum = max(maximum, float(np.abs(actual-expected).max()))
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint changed during export')
    temporary.replace(args.output)
    report = {'checkpoint': str(source), 'checkpoint_sha256': digest,
              'iteration': checkpoint['iter'], 'onnx': str(args.output.resolve()),
              'onnx_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'input_shape': [1, 840], 'output_shape': [1, 15],
              'height_command': height,
              'comparison_samples': 32, 'max_absolute_error': maximum, 'passed': True}
    args.output.with_suffix('.export.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
