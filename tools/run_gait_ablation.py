"""Run bounded matched continuations or auxiliary experiments from scratch.

Each variant gets identical initial weights/optimizer, seed and environment
recipe. Evaluations use the unchanged benchmark and additional gait/heading
diagnostics. No deployment assets are replaced or model auto-promoted.
"""
import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_lab.paths import deployment_framework, deployment_mod, mujoco_xml
import time
import traceback

from run_focused_validation import ROOT, archive_sources, now, read, sha, write
from run_scratch_training import health
from walk_robust_gates import evaluate_gates
from analyze_saved_walk_gait import analyze
from summarize_gait_validation import summarize_batch
from audit_auxiliary_initialization import audit as audit_initialization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Fresh robust-task policy and optimizers; control must be first, no source checkpoint.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deployment-python', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--save-interval', type=int, default=250,
                        help='Save intermediate checkpoints at this iteration interval; evaluate them after each branch finishes.')
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variants', nargs='+', choices=('control', 'slide', 'air_time', 'swing_quality', 'mixed_yaw', 'state', 'dynamics'),
                        default=['control', 'slide', 'air_time'])
    parser.add_argument('--auxiliary-study', action='store_true', help='Unchanged rewards; control/state/dynamics actor supervision.')
    parser.add_argument('--extended-gait', action='store_true',
                        help='Re-evaluate source and references on added moving-turn commands and sole-height traces.')
    parser.add_argument('--reference-checkpoint', type=Path, action='append', default=[])
    args = parser.parse_args()
    max_updates = 5000 if args.auxiliary_study else 1000
    if not 1 <= args.updates <= max_updates or args.num_envs <= 0 or len(set(args.variants)) != len(args.variants):
        parser.error(f'Use unique variants, positive env count, and 1..{max_updates} updates per variant')
    if args.save_interval <= 0:
        parser.error('--save-interval must be positive')
    if args.reference_checkpoint and not args.extended_gait:
        parser.error('Reference evaluations require --extended-gait')
    if args.auxiliary_study:
        if not args.extended_gait or set(args.variants)-{'control','state','dynamics'} or args.reference_checkpoint:
            parser.error('Auxiliary study requires extended gait, fresh control/state/dynamics, and no old control references')
    elif set(args.variants)&{'state','dynamics'}:
        parser.error('State/dynamics variants require --auxiliary-study')
    if args.from_scratch:
        if not args.auxiliary_study or args.variants[0] != 'control' or args.checkpoint or args.expected_sha256:
            parser.error('--from-scratch requires an auxiliary study with control first and no source checkpoint')
        checkpoint, source_health = None, None
    else:
        if not args.checkpoint or not args.expected_sha256:
            parser.error('Continuation requires --checkpoint and --expected-sha256')
        checkpoint = args.checkpoint.resolve(strict=True)
        if sha(checkpoint) != args.expected_sha256:
            raise ValueError('Retained source checkpoint hash mismatch')
        source_health = health(checkpoint, std_bounds=(.05, .30))
    start_iteration = 0 if args.from_scratch else source_health['iteration']
    std_bounds = (.1, .6) if args.from_scratch else (.05, .30)
    task = 'walk_elf3_tang_bxi_robust' if args.from_scratch else 'walk_elf3_tang_bxi_still'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    archive_sources(out)
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    framework = deployment_framework()
    deploy_environment = dict(environment, PYTHONPATH=str(framework)+os.pathsep+environment.get('PYTHONPATH', ''))
    state = dict(status='starting', started_at=now(), supervisor_pid=os.getpid(),
                 source_checkpoint=str(checkpoint) if checkpoint else None, source_health=source_health,
                 from_scratch=args.from_scratch, task=task, initializations={},
                 planned_variants=args.variants, updates_per_variant=args.updates,
                 save_interval=args.save_interval,
                 num_envs=args.num_envs, seed=args.seed, child_pid=None, variants=[],
                 extended_gait=args.extended_gait, references=[],
                 study='actor_auxiliary' if args.auxiliary_study else 'gait_rewards',
                 auto_promote=False, original_training='not modified or stopped')
    child = None

    def persist():
        state['updated_at'] = now()
        write(out/'state.json', state)
        latest = ('latest_auxiliary_scratch_validation.json' if args.from_scratch else
                  'latest_auxiliary_validation.json' if args.auxiliary_study else 'latest_gait_ablation.json')
        write(ROOT/'logs/training_launches'/latest, state)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)

    def execute(argv, log, status, env=environment):
        nonlocal child
        state.update(status=status, command=argv)
        with log.open('x') as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state['child_pid'] = child.pid
            persist()
            if status == 'training' and args.extended_gait:
                summarize_batch(out, state)
            while child.poll() is None:
                if status == 'training' and args.from_scratch:
                    variant = state['active_variant']
                    if variant not in state['initializations']:
                        runs = list((ROOT/'logs'/state['experiment_name']).glob('*_'+state['active_run_name']))
                        if len(runs) == 1 and (runs[0]/'fresh_start.json').exists():
                            checked = audit_initialization(runs[0], state['initializations'].get('control'))
                            state['initializations'][variant] = checked
                            write(log.parent/'initialization_audit.json', checked)
                if status == 'training' and log.exists():
                    with log.open('rb') as f:
                        f.seek(max(0, log.stat().st_size-24000))
                        matches = re.findall(rb'Learning iteration\s+(\d+)', f.read())
                    if matches:
                        state['last_logged_iteration'] = int(matches[-1])
                persist()
                time.sleep(5)
            code = child.returncode
        state['child_pid'] = None
        child = None
        persist()
        if code != 0:
            raise RuntimeError(f'{status} exited {code}; see {log}')

    def evaluate(frozen, ev):
        checked = health(frozen, std_bounds=std_bounds)
        ev.mkdir(parents=True)
        collect_probe = args.auxiliary_study and (state.get('active_variant') == 'control'
                                                 if args.from_scratch else frozen == checkpoint)
        execute([sys.executable, '-u', 'tools/evaluate_standing_candidate.py',
                 '--checkpoint', str(frozen), '--output', str(ev),
                 '--task', task,
                 '--deployment_python', str(args.deployment_python)]
                + (['--include-mixed-yaw', '--gait-diagnostics'] if args.extended_gait else [])
                + (['--auxiliary-probe'] if collect_probe else []),
                ev/'console.log', 'evaluating')
        report = read(ev/'report.json')
        assert report['checkpoint_sha256'] == checked['sha256']
        gates = evaluate_gates(report)
        standing = read(ev/'mujoco_standing.json')
        write(ev/'isaac_gait.json', analyze(ev))
        execute([str(args.deployment_python), '-u', 'tools/diagnose_walk_mujoco.py',
                 '--mod', str(deployment_mod()),
                 '--xml', str(mujoco_xml()),
                 '--onnx', str(ev/'policy.onnx'), '--output', str(ev/'mujoco_gait')]
                + (['--include-mixed-yaw'] if args.extended_gait else []),
                ev/'mujoco_gait.log', 'diagnosing', deploy_environment)
        mj = read(ev/'mujoco_gait/report.json')
        summary = dict(checkpoint=str(frozen), health=checked,
                       original_gates_passed=gates['passed'] and standing['passed'],
                       original_gates=gates, standing=standing,
                       extended_cases_completed={k:v['completed'] for k,v in mj['results'].items()},
                       gait_report=str(ev/'isaac_gait.json'),
                       deployment_diagnostics=str(ev/'mujoco_gait/report.json'),
                       disposition='requires gait/heading comparison; not auto-promoted')
        if args.extended_gait:
            mixed = ('arc_forward_pos', 'arc_forward_neg', 'arc_backward_pos', 'arc_backward_neg')
            summary['mixed_yaw_gates'] = evaluate_gates(report, mixed, (.08, .08, .05))
            summary['mixed_yaw_tracking_limits'] = [.08, .08, .05]
        if args.auxiliary_study and frozen!=checkpoint:
            meta=read(frozen.parent/'auxiliary_supervision.json')
            if meta['variant']!='control':
                prediction_report=ev/'auxiliary_prediction.json'
                probes = (out/'control/evaluations'/frozen.stem if args.from_scratch else
                          out/'references'/f'0_{checkpoint.stem}')
                execute([str(args.deployment_python),'-u','tools/evaluate_actor_auxiliary.py',
                         '--checkpoint',str(frozen),'--probes',str(probes),
                         '--output',str(prediction_report)],ev/'auxiliary_prediction.log','checking_predictions')
                summary['auxiliary_prediction_report']=str(prediction_report)
        write(ev/'summary.json', summary)
        return summary

    try:
        persist()
        if args.extended_gait and not args.from_scratch:
            # Appending scenarios changes the randomization draw sizes. Re-run
            # both the retained source and matched old control on this benchmark.
            references = [checkpoint]+[p.resolve(strict=True) for p in args.reference_checkpoint]
            for i, ref in enumerate(references):
                if i:
                    metadata = read(ref.parent/'gait_ablation.json')
                    assert metadata['variant'] == 'control' and metadata['checkpoint_sha256'] == args.expected_sha256
                    assert metadata['seed'] == args.seed and metadata['num_envs'] == args.num_envs
                    assert health(ref, std_bounds=(.05, .30))['iteration'] == source_health['iteration']+args.updates-1
                state['active_reference'] = str(ref)
                summary = evaluate(ref, out/'references'/f'{i}_{ref.stem}')
                state['references'].append(summary)
                persist()
                if args.extended_gait:
                    summarize_batch(out, state)
            state['active_reference'] = None
        for variant in args.variants:
            state.update(active_variant=variant, last_logged_iteration=None)
            dest = out/variant
            dest.mkdir()
            run_name = f'{out.name}_{variant}_seed{args.seed}'
            experiment=('walk_tang_bxi31_auxiliary_scratch' if args.from_scratch else
                        'walk_tang_bxi31_auxiliary' if args.auxiliary_study else 'walk_tang_bxi31_gait_ablation')
            state.update(experiment_name=experiment, active_run_name=run_name)
            argv = [sys.executable, '-u', 'legged_lab/scripts/train.py',
                    '--task', task]
            argv += (['--from_scratch'] if args.from_scratch else
                     ['--resume', 'True', '--resume_checkpoint', str(checkpoint)])
            argv += [
                    '--auxiliary_supervision' if args.auxiliary_study else '--gait_ablation', variant,
                    '--num_envs', str(args.num_envs), '--max_iterations', str(args.updates),
                    '--save_interval', str(args.save_interval), '--seed', str(args.seed), '--headless',
                    '--device', 'cuda:0', '--logger', 'tensorboard',
                    '--experiment_name', experiment, '--run_name', run_name]
            write(dest/'launch.json', dict(argv=argv, source_sha256=args.expected_sha256))
            execute(argv, dest/'train.log', 'training')
            runs = list((ROOT/'logs'/experiment).glob('*_'+run_name))
            if len(runs) != 1:
                raise RuntimeError('Missing or ambiguous experiment run')
            run = runs[0]
            metadata = read(run/('auxiliary_supervision.json' if args.auxiliary_study else 'gait_ablation.json'))
            assert metadata['checkpoint_sha256'] == args.expected_sha256
            assert metadata['actor_observation_dim'] == 840
            if args.from_scratch:
                checked = audit_initialization(run, state['initializations'].get('control'))
                state['initializations'][variant] = checked
                write(dest/'initialization_audit.json', checked)
            else:
                assert metadata['optimizer_state_entries'] > 0
            assert metadata['num_envs'] == args.num_envs and metadata['seed'] == args.seed
            assert metadata['iteration_before_learning'] == start_iteration
            if args.auxiliary_study and state['variants']:
                assert metadata['reward_sha256']==state['variants'][0]['metadata']['reward_sha256']
                assert metadata['command_sha256']==state['variants'][0]['metadata']['command_sha256']
            checkpoints = sorted(run.glob('model_*.pt'), key=lambda p:int(p.stem.split('_')[-1]))
            assert checkpoints and int(checkpoints[-1].stem.split('_')[-1]) == start_iteration+args.updates-1
            row = dict(variant=variant, run_dir=str(run), metadata=metadata, evaluations=[])
            state['variants'].append(row)
            # Training and evaluation are serialized to keep GPU memory bounded.
            # The first iteration's save is excluded; keep all subsequent
            # periodic saves and the final checkpoint (start+updates-1).
            selected = [p for p in checkpoints if int(p.stem.split('_')[-1]) > start_iteration]
            if not selected:
                selected = [checkpoints[-1]]
            for frozen in selected:
                ev = dest/'evaluations'/frozen.stem
                summary = evaluate(frozen, ev)
                row['evaluations'].append(summary)
                persist()
                if args.extended_gait:
                    summarize_batch(out, state)
            row['completed_at'] = now()
        state.update(status='completed', completed_at=now(), active_variant=None,
                     completed_updates=args.updates*len(args.variants))
        persist()
        if args.extended_gait:
            summarize_batch(out, state)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     child_pid=None, stop_reason=str(exc), traceback=traceback.format_exc())
        persist()
        raise


if __name__ == '__main__':
    main()
