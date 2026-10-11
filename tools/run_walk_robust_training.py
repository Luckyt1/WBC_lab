"""Supervise fresh locomotion or checkpoint-based quiet-standing refinement."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

from run_focused_validation import ROOT, archive_sources, now, read, write
from run_scratch_training import health
from walk_robust_gates import evaluate_gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_iterations', type=int, default=50000)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--evaluation_interval', type=int, default=5000)
    parser.add_argument('--task', choices=('walk_elf3_tang_bxi_robust', 'walk_elf3_tang_bxi_still'), default='walk_elf3_tang_bxi_robust')
    parser.add_argument('--resume_checkpoint', type=Path)
    parser.add_argument('--deployment_python', type=Path)
    parser.add_argument('--auxiliary_supervision', choices=('dynamics', 'control'))
    parser.add_argument('--standing_position_penalty', action='store_true',
                        help='Enable the additional stop-position reward recipe.')
    parser.add_argument('--teleop_standing', action='store_true',
                        help='Standing-first arm-motion recipe, including the position penalty.')
    parser.add_argument('--teleop_standing_fraction', type=float, default=.8)
    parser.add_argument('--pelvis_height', type=float, default=None)
    parser.add_argument('--elf3_symmetry', action='store_true')
    parser.add_argument('--liftoff_rewards', action='store_true')
    parser.add_argument('--ik_arm_bank', type=Path)
    parser.add_argument('--ik_python', type=Path, help='Isolated Pinocchio + MuJoCo evaluation interpreter.')
    parser.add_argument('--disable_symmetry_on_resume', action='store_true')
    parser.add_argument('--teleop_curriculum_stage', default=None)
    parser.add_argument('--teleop_curriculum_decision', type=Path)
    parser.add_argument('--external_evaluation', action='store_true',
                        help='Gated curriculum controller evaluates after training releases the GPU.')
    parser.add_argument('--initialization_reference', type=Path,
                        help='Read-only fresh-run audit reference; never loaded into the policy.')
    parser.add_argument('--save_interval', type=int, default=500)
    args = parser.parse_args()
    if args.ik_arm_bank and (args.teleop_curriculum_stage or not args.ik_python):
        parser.error('IK bank requires no gated curriculum and --ik_python')
    if not 0. <= args.teleop_standing_fraction <= 1.:
        parser.error('--teleop_standing_fraction must be between 0 and 1')
    if args.teleop_standing_fraction != .8 and not args.teleop_standing:
        parser.error('--teleop_standing_fraction requires --teleop_standing')
    assert min(args.max_iterations, args.num_envs, args.evaluation_interval, args.save_interval) > 0
    auxiliary = args.auxiliary_supervision is not None
    # Both variants use audited checkpoints; control has no auxiliary learning.
    if args.auxiliary_supervision == 'control' and args.teleop_curriculum_stage is None:
        parser.error('Plain PPO control currently requires the externally evaluated teleop curriculum')
    if args.disable_symmetry_on_resume and (args.elf3_symmetry or not auxiliary or not args.resume_checkpoint):
        parser.error('Disabling symmetry requires an auxiliary checkpoint continuation without symmetry')
    if args.elf3_symmetry and args.task != 'walk_elf3_tang_bxi_robust':
        parser.error('--elf3_symmetry requires the robust task')
    if args.liftoff_rewards and args.task != 'walk_elf3_tang_bxi_robust':
        parser.error('--liftoff_rewards requires the robust task')
    if args.teleop_curriculum_stage is not None:
        from teleop_curriculum import STAGES
        if (args.teleop_curriculum_stage not in STAGES or not args.teleop_standing or not auxiliary
                or not args.external_evaluation):
            parser.error('Curriculum chunks require dynamics/control, teleop standing and external evaluation')
    elif args.external_evaluation or args.teleop_curriculum_decision:
        parser.error('External evaluation/decision is only for a teleop curriculum chunk')
    if args.pelvis_height is not None:
        from fixed_pelvis_height import validate_height
        validate_height(args.pelvis_height)
        if args.task != 'walk_elf3_tang_bxi_robust':
            parser.error('--pelvis_height requires the robust task')
    if args.teleop_standing and args.task != 'walk_elf3_tang_bxi_robust':
        parser.error('--teleop_standing requires walk_elf3_tang_bxi_robust')
    if auxiliary and (args.task != 'walk_elf3_tang_bxi_robust' or not args.deployment_python):
        parser.error('Dynamics long training requires the robust task and deployment Python')
    if args.initialization_reference and not auxiliary:
        parser.error('--initialization_reference requires dynamics training')
    if args.initialization_reference and args.resume_checkpoint:
        parser.error('Fresh initialization reference cannot be used for continuation')
    checkpoint_source = args.resume_checkpoint.resolve() if args.resume_checkpoint else None
    if args.task.endswith('_still') and (checkpoint_source is None or args.deployment_python is None):
        parser.error('standing refinement needs --resume_checkpoint and --deployment_python')
    # A resume source may come from either the original (.1, .6) policy or
    # quiet-standing refinement (.05, .30). The loaded policy projects to the
    # destination bounds; subsequent checkpoint checks use those tighter bounds.
    initial_health = health(checkpoint_source, std_bounds=(.05, .6)) if checkpoint_source else None
    start_iteration = initial_health['iteration'] if initial_health else 0
    if auxiliary and checkpoint_source:
        import torch
        from audit_auxiliary_resume import completed_updates
        start_iteration = completed_updates(torch.load(checkpoint_source, map_location='cpu', weights_only=False),
                                            args.auxiliary_supervision)
    std_bounds = (.05, .30) if args.task.endswith('_still') else (.1, .6)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'launch.lock').open('x') as f:
        f.write(str(os.getpid()))
    archive_sources(out)
    if args.ik_arm_bank:
        # Include generated inputs and the upstream YAML needed by the numerical
        # source loader; source hashes alone cannot reconstruct a trajectory bank.
        import hashlib
        import zipfile
        bank = args.ik_arm_bank.resolve(strict=True)
        files = [bank/name for name in ('manifest.json','train.npz','test.npz')]
        from elf3_ik_core import SOURCE, PROVENANCE
        files.extend(SOURCE/name for name in read(PROVENANCE)['sha256'])
        if read(bank/'manifest.json')['version']>=3:
            files.extend(p for p in (ROOT/'third_party/ruckig').rglob('*') if p.is_file())
            files.extend((ROOT/'tools/native').glob('*.cpp'))
            files.extend((ROOT/'legged_lab/envs/elf3_tang/wbc').glob('_arm_ruckig*.so'))
        with zipfile.ZipFile(out/'source.zip','a',zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                try:
                    relative = str(path.relative_to(ROOT))
                except ValueError:
                    relative = ('ik_bank/' if path.parent == bank else 'ik_upstream/')+path.name
                archive.write(path,relative)
        write(out/'ik_input_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    suffix = f'from{start_iteration}' if checkpoint_source else 'fresh'
    name = f'{out.name}_{args.num_envs}_{suffix}'
    if args.teleop_curriculum_stage is not None:
        name = f'{out.parent.parent.name}_{name}'
    state = dict(status='starting', started_at=now(), supervisor_pid=os.getpid(),
                 task=args.task, from_scratch=checkpoint_source is None,
                 resumed_checkpoint=str(checkpoint_source) if checkpoint_source else None,
                 initial_iteration=start_iteration, completed_updates=0,
                 target_total_iterations=start_iteration + args.max_iterations,
                 initial_health=initial_health, teacher_checkpoint=None, num_envs=args.num_envs,
                 seed=42, planned_updates=args.max_iterations, run_name=name,
                 commands='15% stand, 15% turn, 15% lateral, 55% forward/back; height .8235m and waist yaw 0 throughout',
                 disturbances='push 0->full over 20k updates; reset velocity 0->full over 30k; observation noise on; AMASS .05->1 from 5k to 20k',
                 last_logged_iteration=None, latest_checkpoint=None, best_checkpoint=None,
                 best_score=None, child_pid=None, eval_pid=None, evaluations=[],
                 save_interval=500, evaluation_interval=args.evaluation_interval,
                 console=str(out/'train.log'), source_archive=str(out/'source.zip'))
    state['elf3_symmetry'] = args.elf3_symmetry
    state['liftoff_reward_recipe'] = 'bcd_v1' if args.liftoff_rewards else None
    state['disable_symmetry_on_resume'] = args.disable_symmetry_on_resume
    state['save_interval'] = args.save_interval
    if auxiliary:
        state.update(variant=args.auxiliary_supervision,auxiliary_supervision=args.auxiliary_supervision,auto_promote=False,
                     total_completed_updates=start_iteration,
                     initialization_reference=str(args.initialization_reference) if args.initialization_reference else None,
                     evaluation_scope='Frozen 12 scenarios x 32 trials x 3 profiles; MuJoCo gait/standing; own held-out auxiliary probes')
    if args.task.endswith('_still'):
        state['commands'] = '35% stand, 15% turn, 15% lateral, 35% forward/back; fixed height .8235m and waist yaw 0'
        state['disturbances'] = 'inherited curriculum clock; full push/arms, reset velocity reaches full at 30k; 50% neutral arms'
        state['deployment_validation'] = '60s MuJoCo zero command and walk-to-stop; max drift .15m, mean speed .02m/s'
    if args.teleop_standing:
        state['teleop_standing'] = True
        state['teleop_standing_fraction'] = args.teleop_standing_fraction
        state['standing_position_penalty'] = True
        moving = round(1. - args.teleop_standing_fraction, 15)
        state['commands'] = (f'{100*args.teleop_standing_fraction:g}% stand, {25*moving:g}% turn, '
                             f'{25*moving:g}% lateral, {50*moving:g}% forward/back; fixed height .8235m and waist yaw 0')
        state['disturbances'] += '; 70% AMASS / 30% neutral arm references (independent of command mode)'
    lock = threading.RLock()
    if args.ik_arm_bank:
        bank = args.ik_arm_bank.resolve(strict=True)
        manifest = read(bank/'manifest.json')
        state['ik_arm_bank'] = str(bank)
        state['ik_counts'] = manifest['counts']
        state['ik_bank_sha256'] = manifest['bank_sha256']
        state['ik_execution_version'] = manifest['version']
        state['ik_python'] = str(args.ik_python)
        state['disturbances'] = ('push 0->full over 20k updates; reset velocity 0->full over 30k; noise on; '
            'procedural IK full range from start; 85% reach/hold/return, 15% neutral; 30% active targets from prior drift/instability cases; no gated arm curriculum')
        state['evaluation_scope'] += '; frozen unseen IK targets with online original IK, live shared guard/rate limiter and gravity in MuJoCo'
        if manifest['version']>=2:
            state['evaluation_scope'] = ('Frozen AMASS retention and auxiliary probes; unchanged IK goal pool including rejected goals; '
                'validated branch guides, live waist correction, arrival gating, shared OBB/rate guard and gravity in MuJoCo')
            state['disturbances'] += f"; one complete arrival-gated arm attempt per {manifest['recipe']['max_episode_s']}s episode, neutral after return; OBB margin 3mm"
        if manifest['version']>=3:
            state['evaluation_scope'] += '; local Ruckig v/a/j limits, predicted moving obstacles, bounded RRT-Connect; planning failures explicitly fail without blind IK fallback'
    if args.pelvis_height is not None:
        state['pelvis_height'] = args.pelvis_height
        state['commands'] = state['commands'].replace('.8235m', f'{args.pelvis_height}m')
    if args.teleop_curriculum_stage is not None:
        from teleop_curriculum import settings
        state['teleop_curriculum_stage'] = args.teleop_curriculum_stage
        state['difficulty'] = settings(args.teleop_curriculum_stage)
        state['disturbances'] = 'Fixed at this stage; only frozen evaluation can change difficulty'
        state['evaluation_scope'] = 'External gated controller: 60s Isaac current/easier stages plus 60s MuJoCo standing'
    def persist():
        with lock:
            state['updated_at'] = now()
            write(out/'state.json', state)
            latest = 'latest_dynamics_scratch_training.json' if auxiliary else 'latest_walk_robust_bxi31.json'
            if args.auxiliary_supervision == 'control':
                latest = 'latest_control_scratch_training.json'
            write(ROOT/'logs/training_launches'/latest, state)
            if args.auxiliary_supervision == 'dynamics':
                from evaluate_dynamics_checkpoint import write_comparison
                write_comparison(out,state)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    argv = [sys.executable, '-u', 'legged_lab/scripts/train.py', '--task', state['task'],
            '--num_envs', str(args.num_envs), '--max_iterations', str(args.max_iterations),
            '--seed', '42', '--headless', '--device', 'cuda:0', '--logger', 'tensorboard', '--run_name', name]
    if args.standing_position_penalty:
        argv += ['--standing_position_penalty']
        state['standing_position_penalty'] = True
    if args.teleop_standing:
        argv += ['--teleop_standing', '--teleop_standing_fraction', str(args.teleop_standing_fraction)]
    if args.pelvis_height is not None:
        argv += ['--pelvis_height', str(args.pelvis_height)]
    if args.elf3_symmetry:
        argv += ['--elf3_symmetry']
    if args.liftoff_rewards:
        argv += ['--liftoff_rewards']
    if args.ik_arm_bank:
        argv += ['--ik_arm_bank', str(args.ik_arm_bank.resolve())]
    if args.disable_symmetry_on_resume:
        argv += ['--disable_symmetry_on_resume']
    if args.teleop_curriculum_stage is not None:
        argv += ['--teleop_curriculum_stage', args.teleop_curriculum_stage]
    if args.teleop_curriculum_decision:
        argv += ['--teleop_curriculum_decision', str(args.teleop_curriculum_decision)]
    if checkpoint_source:
        if auxiliary:
            argv += ['--resume', 'True', '--resume_checkpoint', str(checkpoint_source), '--resume_auxiliary_training']
        else:
            argv += ['--resume', 'True', '--load_run', checkpoint_source.parent.name, '--checkpoint', checkpoint_source.name]
    else:
        argv += ['--from_scratch']
    experiment = 'walk_tang_bxi31_auxiliary_scratch' if auxiliary else 'walk_tang_bxi31_robust'
    if args.auxiliary_supervision == 'control':
        experiment = 'walk_tang_bxi31_control_scratch'
    argv += ['--save_interval',str(args.save_interval)]
    if auxiliary:
        argv += ['--auxiliary_supervision',args.auxiliary_supervision,'--experiment_name',experiment]
    state['argv'] = argv
    child = evaluator = eval_stream = None
    seen_initial = checkpoint_source is not None and not auxiliary
    checked = set()
    completed = set()
    # Keep evaluations on the original global schedule after a continuation.
    next_eval = (start_iteration // args.evaluation_interval + 1) * args.evaluation_interval
    eval_checkpoint = None
    try:
        persist()
        child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=auxiliary)
        state.update(status='training', child_pid=child.pid)
        def stream_training():
            with (out/'train.log').open('w') as f:
                for line in child.stdout:
                    f.write(line)
                    f.flush()
                    match = re.search(r'Learning iteration\s+(\d+)', line)
                    if match:
                        with lock:
                            state['last_logged_iteration'] = int(match[1])
                            state['completed_updates'] = int(match[1]) - start_iteration + 1
                            if auxiliary:
                                state['total_completed_updates'] = int(match[1]) + 1
        reader = threading.Thread(target=stream_training, daemon=True)
        reader.start()
        while True:
            code = child.poll()
            if code is not None and code != 0:
                raise RuntimeError(f'Training exited {code}; see train.log')
            runs = list((ROOT/'logs'/experiment).glob('*_' + name))
            if len(runs) > 1:
                raise RuntimeError('Ambiguous run directory')
            checkpoints = []
            if runs:
                run = runs[0]
                state['run_dir'] = str(run)
                if auxiliary and checkpoint_source and not seen_initial and (run/'resume_start.json').is_file():
                    info = read(run/'resume_start.json')
                    assert info['exact_learned_state_restored']
                    assert info['checkpoint_sha256'] == initial_health['sha256']
                    assert info['iteration_before_learning'] == start_iteration
                    state['resume_audit'] = info
                    write(out/'resume_audit.json', info)
                    seen_initial = True
                if not seen_initial and (run/'fresh_start.json').is_file():
                    info = read(run/'fresh_start.json')
                    assert info['from_scratch'] and info['checkpoint_loaded'] is None
                    assert info['num_envs'] == args.num_envs and info['actor_observation_dim'] == 840
                    state['fresh_start'] = info
                    state['initial_health'] = health(run/'initial_policy.pt', initial=True)
                    if auxiliary:
                        from audit_auxiliary_initialization import audit
                        reference=audit(args.initialization_reference) if args.initialization_reference else None
                        state['initialization_audit']=audit(run,reference)
                        write(out/'initialization_audit.json',state['initialization_audit'])
                    seen_initial = True
                checkpoints = sorted((p for p in run.glob('model_*.pt')
                                      if re.fullmatch(r'model_\d+\.pt', p.name)
                                      and (code == 0 or time.time()-p.stat().st_mtime > 3)),
                                     key=lambda p: int(p.stem.split('_')[-1]))
                for checkpoint in checkpoints:
                    if str(checkpoint) not in checked:
                        state['latest_health'] = health(checkpoint, std_bounds=std_bounds)
                        state['latest_checkpoint'] = str(checkpoint)
                        checked.add(str(checkpoint))
            if evaluator is not None and evaluator.poll() is not None:
                eval_stream.close()
                eval_stream = None
                state['eval_pid'] = None
                if evaluator.returncode:
                    state.setdefault('evaluation_errors', []).append(dict(
                        checkpoint=str(eval_checkpoint), returncode=evaluator.returncode, at=now()))
                else:
                    dest = out/'evaluations'/eval_checkpoint.stem
                    report = read(dest/'report.json')
                    assert report['checkpoint_sha256'] == health(eval_checkpoint, std_bounds=std_bounds)['sha256']
                    gates = evaluate_gates(report)
                    if args.task.endswith('_still'):
                        deployment = read(dest/'mujoco_standing.json')
                        gates['mujoco_standing'] = deployment
                        gates['passed'] = gates['passed'] and deployment['passed']
                    summary = (read(dest/'summary.json') if auxiliary else
                               dict(checkpoint=str(eval_checkpoint), completed_at=now(),gates=gates,report_dir=str(dest)))
                    write(dest/'summary.json', summary)
                    state['evaluations'].append(summary)
                    score = gates['profiles']['training_disturbances']['score']
                    if not auxiliary and gates['passed'] and (state['best_score'] is None or score < state['best_score']):
                        state.update(best_checkpoint=str(eval_checkpoint), best_score=score)
                completed.add(str(eval_checkpoint))
                evaluator = None
            if evaluator is None and checkpoints and not args.external_evaluation:
                candidate = checkpoints[-1]
                age = int(candidate.stem.split('_')[-1])
                if str(candidate) not in completed and (age >= next_eval or code == 0):
                    eval_checkpoint = candidate
                    next_eval = (age//args.evaluation_interval + 1)*args.evaluation_interval
                    dest = out/'evaluations'/candidate.stem
                    dest.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, '-u', 'tools/evaluate_walk_tang.py', '--task', state['task'],
                               '--checkpoint', str(candidate), '--output', str(dest), '--arm-guard', '--headless']
                    if args.task.endswith('_still'):
                        command = [sys.executable, '-u', 'tools/evaluate_standing_candidate.py',
                                   '--checkpoint', str(candidate), '--output', str(dest),
                                   '--deployment_python', str(args.deployment_python)]
                    if auxiliary:
                        command=[sys.executable,'-u','tools/evaluate_dynamics_checkpoint.py',
                                 '--checkpoint',str(candidate),'--output',str(dest),
                                 '--deployment_python',str(args.deployment_python)]
                        if args.ik_arm_bank:
                            command += ['--ik-bank', str(args.ik_arm_bank.resolve()), '--ik-python', str(args.ik_python)]
                            if code == 0:
                                command.append('--ik-full')
                    eval_stream = (dest/'console.log').open('w')
                    evaluator = subprocess.Popen(command, cwd=ROOT, env=environment,
                                                 stdout=eval_stream, stderr=subprocess.STDOUT, start_new_session=auxiliary)
                    state.update(eval_pid=evaluator.pid, evaluating_checkpoint=str(candidate))
            if code == 0:
                state.update(child_pid=None, status='final_evaluation')
                if checkpoints and (args.external_evaluation or str(checkpoints[-1]) in completed) and evaluator is None:
                    assert seen_initial
                    reader.join(timeout=5)
                    state.update(status='completed', completed_at=now(), completed_updates=args.max_iterations)
                    persist()
                    return
            persist()
            time.sleep(5)
    except BaseException as exc:
        for process in (child, evaluator):
            if process is not None and process.poll() is None:
                if auxiliary:
                    os.killpg(process.pid,signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if auxiliary:
                        os.killpg(process.pid,signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=10)
        if eval_stream is not None:
            eval_stream.close()
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     stop_reason=str(exc), traceback=traceback.format_exc(), child_pid=None, eval_pid=None)
        persist()
        raise


if __name__ == '__main__':
    main()
