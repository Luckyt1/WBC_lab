"""Preserve virtual-environment entrypoints and validate saved evaluation inputs."""
import hashlib
import json
from pathlib import Path


def deployment_python(requested, output):
    # A running supervisor keeps its original CLI arguments. This explicit,
    # run-local override repairs its evaluation runtime without interrupting PPO.
    output = Path(output)
    override = output.parent.parent/'evaluation_runtime.json'
    chosen = requested
    if output.parent.name == 'evaluations' and override.is_file():
        chosen = json.loads(override.read_text())['deployment_python']
    # Resolving a venv's python symlink would select the base interpreter instead.
    entrypoint = Path(chosen).expanduser().absolute()
    if not entrypoint.is_file():
        raise FileNotFoundError(entrypoint)
    return entrypoint


def validate_saved_isaac(checkpoint, output, task, mixed=False, gait=False, auxiliary=False):
    output = Path(output)
    report = json.loads((output/'report.json').read_text())
    if report['checkpoint_sha256'] != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
        raise ValueError('Saved Isaac report belongs to a different checkpoint')
    if report['task'] != task or not report['arm_guard_enabled']:
        raise ValueError('Saved Isaac task/guard settings do not match')
    if task in ('walk_elf3_tang_bxi_robust', 'walk_elf3_tang_bxi_still') and (Path(checkpoint).parent/'params/env.yaml').is_file():
        from fixed_pelvis_height import checkpoint_height
        if report.get('height_command', .8235) != checkpoint_height(checkpoint):
            raise ValueError('Saved Isaac height command does not match the checkpoint recipe')
    for required, key in ((mixed, 'mixed_yaw_scenarios'), (gait, 'gait_diagnostics'),
                          (auxiliary, 'auxiliary_probe')):
        if required and not report.get(key):
            raise ValueError(f'Saved Isaac report lacks {key}')
    if report['repeats'] < 32 or report['measure_s'] < 12 or report['warmup_s'] < 4:
        raise ValueError('Saved Isaac report has insufficient trials/duration')
    names = {'bootstrap', 'controlled_arms_on', 'training_disturbances'}
    if {p['settings']['name'] for p in report['profiles']} != names or len(report['profiles']) != 3:
        raise ValueError('Saved Isaac profiles are incomplete')
    scenarios = {'stand','forward_030','forward_060','backward_030','lateral_pos','lateral_neg','turn_pos','turn_neg'}
    if mixed:
        scenarios |= {'arc_forward_pos','arc_forward_neg','arc_backward_pos','arc_backward_neg'}
    for profile in report['profiles']:
        if not scenarios.issubset({s['scenario'] for s in profile['scenarios']}):
            raise ValueError('Saved Isaac scenarios are incomplete')
        name = profile['settings']['name']
        needed = [output/f'{name}.npz']
        if auxiliary:
            needed.append(output/f'{name}_auxiliary_probe.npz')
        for path in needed:
            if not path.is_file():
                raise FileNotFoundError(path)
    return report
