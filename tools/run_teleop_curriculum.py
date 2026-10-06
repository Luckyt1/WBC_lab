"""Train dynamics or plain PPO with sequential, performance-gated teleop difficulty."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_focused_validation import archive_sources, now, read, write, sha
from tools.teleop_curriculum import STAGES, settings
from tools.teleop_curriculum_gates import decide


def training_command(python, deployment_python, output, stage, updates, num_envs, height,
                     checkpoint=None, decision=None, symmetry=False, liftoff_rewards=False,
                     disable_symmetry_on_resume=False, standing_fraction=.8, auxiliary_supervision='dynamics'):
    if auxiliary_supervision not in ('dynamics', 'control'):
        raise ValueError('Curriculum requires dynamics or plain-PPO control')
    if not 0. <= standing_fraction <= 1.:
        raise ValueError('Standing command fraction must be between 0 and 1')
    if disable_symmetry_on_resume and (symmetry or checkpoint is None):
        raise ValueError('Symmetry removal requires a checkpoint and disabled symmetry')
    argv = [str(python), '-u', 'tools/run_walk_robust_training.py', '--output', str(output),
            '--task', 'walk_elf3_tang_bxi_robust', '--auxiliary_supervision', auxiliary_supervision,
            '--teleop_standing', '--teleop_standing_fraction', str(standing_fraction),
            '--pelvis_height', str(height), '--teleop_curriculum_stage', stage,
            '--external_evaluation', '--max_iterations', str(updates), '--num_envs', str(num_envs),
            '--save_interval', '500', '--evaluation_interval', '1000',
            '--deployment_python', str(deployment_python)]
    if symmetry:
        argv.append('--elf3_symmetry')
    if liftoff_rewards:
        argv.append('--liftoff_rewards')
    if disable_symmetry_on_resume:
        argv.append('--disable_symmetry_on_resume')
    if checkpoint:
        argv += ['--resume_checkpoint', str(checkpoint)]
    if decision:
        argv += ['--teleop_curriculum_decision', str(decision)]
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deployment-python', type=Path, required=True)
    parser.add_argument('--max-total-updates', type=int, default=100000)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--pelvis-height', type=float, default=.78)
    parser.add_argument('--elf3-symmetry', action='store_true')
    parser.add_argument('--liftoff-rewards', action='store_true')
    parser.add_argument('--auxiliary-supervision', choices=('dynamics', 'control'), default='dynamics',
                        help='control: plain PPO without prediction heads, auxiliary losses or label collection.')
    parser.add_argument('--teleop-standing-fraction', type=float, default=.8,
                        help='Standing probability; remaining commands use a 2:1:1 forward/back:lateral:turn split.')
    parser.add_argument('--resume-checkpoint', type=Path,
                        help='Continue a completed checkpoint in a new controller directory.')
    parser.add_argument('--disable-symmetry-on-resume', action='store_true')
    parser.add_argument('--adopt-active-training', action='store_true',
                        help='Replace an exited controller while leaving its active training segment running.')
    args = parser.parse_args()
    if not 0. <= args.teleop_standing_fraction <= 1.:
        parser.error('--teleop-standing-fraction must be between 0 and 1')
    if args.adopt_active_training and args.resume_checkpoint:
        parser.error('Adopting an active trainer and resuming a checkpoint are mutually exclusive')
    if args.disable_symmetry_on_resume and (args.elf3_symmetry or not args.resume_checkpoint):
        parser.error('Disabling symmetry requires --resume-checkpoint without --elf3-symmetry')
    from tools.fixed_pelvis_height import validate_height
    validate_height(args.pelvis_height)
    if args.max_total_updates < 5000 or args.num_envs <= 0:
        parser.error('Use at least the first 5000 updates and positive environment count')
    resumed_checkpoint, initial_updates, initial_stage = None, 0, STAGES[0]
    if args.resume_checkpoint:
        import torch
        from tools.audit_auxiliary_resume import completed_updates
        from tools.teleop_curriculum import checkpoint_stage
        resumed_checkpoint = args.resume_checkpoint.resolve(strict=True)
        initial_updates = completed_updates(torch.load(resumed_checkpoint, map_location='cpu', weights_only=False),
                                            args.auxiliary_supervision)
        initial_stage = checkpoint_stage(resumed_checkpoint)
        if initial_updates >= args.max_total_updates:
            parser.error('Continuation needs a target greater than completed checkpoint updates')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lease = (out/'controller.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    adopted = None
    if args.adopt_active_training:
        from tools.adopt_training_process import AdoptedTrainingProcess, process_identity
        state = read(out/'state.json')
        if state['status'] != 'training' or process_identity(state['supervisor_pid']) is not None:
            raise ValueError('Previous controller must have exited while its training remains active')
        if (state['target_total_iterations'] != args.max_total_updates or state['num_envs'] != args.num_envs
                or state['pelvis_height'] != args.pelvis_height
                or state.get('elf3_symmetry', False) != args.elf3_symmetry
                or state.get('teleop_standing_fraction', .8) != args.teleop_standing_fraction
                or state.get('auxiliary_supervision', 'dynamics') != args.auxiliary_supervision
                or bool(state.get('liftoff_reward_recipe')) != args.liftoff_rewards):
            raise ValueError('Controller adoption cannot change the training recipe or total target')
        adopted = AdoptedTrainingProcess(state['active_segment'])
        prior_controller = state['supervisor_pid']
        revision = out/'controller_revisions'/str(os.getpid())
        revision.mkdir(parents=True, exist_ok=False)
        write(revision/'prior_state.json', state)
        archive_sources(revision)
        state.update(supervisor_pid=os.getpid(), adopted_at=now(), previous_controller_pid=prior_controller,
                     controller_source_archive=str(revision/'source.zip'))
    else:
        with (out/'launch.lock').open('x') as stream:
            stream.write(str(os.getpid()))
        archive_sources(out)
    initial_state = dict(status='starting', started_at=now(), supervisor_pid=os.getpid(),
                 from_scratch=resumed_checkpoint is None,
                 resumed_checkpoint=str(resumed_checkpoint) if resumed_checkpoint else None,
                 auxiliary_supervision=args.auxiliary_supervision,
                 teleop_standing=True, standing_position_penalty=True, pelvis_height=args.pelvis_height,
                 teleop_standing_fraction=args.teleop_standing_fraction,
                 elf3_symmetry=args.elf3_symmetry,
                 liftoff_reward_recipe='bcd_v1' if args.liftoff_rewards else None,
                 num_envs=args.num_envs, seed=42, target_total_iterations=args.max_total_updates,
                 first_segment_updates=5000, subsequent_segment_updates=1000,
                 total_completed_updates=initial_updates, curriculum_stage=initial_stage, stage_start_updates=initial_updates,
                 difficulty=settings(initial_stage), latest_checkpoint=str(resumed_checkpoint) if resumed_checkpoint else None,
                 disable_symmetry_on_resume=args.disable_symmetry_on_resume, training_supervisor_pid=None,
                 child_pid=None, eval_pid=None, evaluations=[], auto_promote=True,
                 method='Only two consecutive frozen 60s gate passes promote one stage; current/previous/quiet retention; independent MuJoCo; no simultaneous training/evaluation',
                 output=str(out))
    if not args.adopt_active_training:
        state = initial_state
    state['limits'] = 'No curriculum-triggered early stop; plateau/long stage holds, retention failures reduce difficulty; continue through full-stage validation to the total update target'
    state['stop_on_stagnation'] = False
    state['stop_on_full_stage_validated'] = False
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    child = None
    def persist():
        state['updated_at'] = now()
        write(out/'state.json', state)
        write(ROOT/'logs/training_launches/latest_teleop_curriculum.json', state)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    def execute(argv, log, phase, chunk=None, existing=None):
        nonlocal child
        if existing is not None:
            child = existing
        else:
            with log.open('x') as stream:
                child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        state['status'] = phase
        state['active_argv'] = argv
        state['active_log'] = str(log)
        state['training_supervisor_pid' if phase == 'training' else 'eval_pid'] = child.pid
        persist()
        while child.poll() is None:
            if chunk is not None and (chunk/'state.json').exists():
                progress = read(chunk/'state.json')
                for key in ('total_completed_updates', 'latest_checkpoint', 'child_pid', 'run_dir',
                            'latest_health', 'fresh_start', 'initialization_audit', 'resume_audit'):
                    if progress.get(key) is not None:
                        state[key] = progress[key]
            persist()
            time.sleep(5)
        if child.returncode:
            raise RuntimeError(f'{phase} exited {child.returncode}: {log}')
        child = None
        state.update(training_supervisor_pid=None, child_pid=None, eval_pid=None)
        persist()
    try:
        persist()
        checkpoint, transition = resumed_checkpoint, None
        completed, stage_start, stage = initial_updates, initial_updates, initial_stage
        history = []
        if adopted is not None:
            progress = read(Path(state['active_segment'])/'state.json')
            completed = progress['initial_iteration']
            stage, stage_start = state['curriculum_stage'], state['stage_start_updates']
            history = state['evaluations']
            checkpoint = progress.get('resumed_checkpoint')
            transition = history[-1]['decision_path'] if history else None
        while completed < args.max_total_updates:
            count = min(5000 if completed == 0 else 1000, args.max_total_updates-completed)
            chunk = out/'segments'/f'{completed:06d}_{stage}'
            if adopted is None:
                chunk.mkdir(parents=True, exist_ok=False)
            elif (chunk.resolve() != Path(state['active_segment']).resolve()
                    or read(chunk/'state.json')['target_total_iterations'] != completed+count):
                raise ValueError('Adopted segment does not match the expected next boundary')
            state.update(curriculum_stage=stage, stage_start_updates=stage_start,
                         difficulty=settings(stage), active_segment=str(chunk))
            command = training_command(sys.executable, args.deployment_python.absolute(), chunk, stage,
                count, args.num_envs, args.pelvis_height, checkpoint, transition,
                symmetry=args.elf3_symmetry, liftoff_rewards=args.liftoff_rewards,
                standing_fraction=args.teleop_standing_fraction,
                auxiliary_supervision=args.auxiliary_supervision,
                disable_symmetry_on_resume=(args.disable_symmetry_on_resume and checkpoint == resumed_checkpoint))
            execute(command, chunk/'supervisor.log', 'training', chunk, existing=adopted)
            adopted = None
            result = read(chunk/'state.json')
            if result['status'] != 'completed' or result['total_completed_updates'] != completed+count:
                raise RuntimeError('Training chunk did not complete its exact planned updates')
            completed += count
            checkpoint = Path(result['latest_checkpoint'])
            state.update(total_completed_updates=completed, latest_checkpoint=str(checkpoint))
            evaluation = out/'evaluations'/f'{completed:06d}_{stage}'
            evaluation.mkdir(parents=True, exist_ok=False)
            execute([sys.executable, '-u', 'tools/evaluate_teleop_curriculum.py', '--checkpoint', str(checkpoint),
                     '--output', str(evaluation), '--deployment-python', str(args.deployment_python.absolute())],
                    evaluation/'console.log', 'evaluating')
            evidence = read(evaluation/'curriculum_gate.json')
            if evidence['checkpoint_sha256'] != sha(checkpoint) or evidence['stage'] != stage:
                raise RuntimeError('Gate/checkpoint mismatch')
            decision = decide(history, stage, stage_start, completed, evidence['gate'])
            artifact = dict(**evidence, completed_updates=completed, decision=decision,
                            stage_start_updates=stage_start, prior_history=history)
            transition = evaluation/'decision.json'
            write(transition, artifact)
            history.append(dict(stage=stage, completed_updates=completed, gate=evidence['gate'],
                                decision=decision, checkpoint_sha256=evidence['checkpoint_sha256'],
                                decision_path=str(transition), decision_sha256=sha(transition)))
            state['evaluations'] = history
            state['last_decision'] = decision
            if decision['action'] == 'full_stage_validated':
                state.setdefault('first_full_stage_validated_at', now())
            if decision['next_stage'] != stage:
                stage, stage_start = decision['next_stage'], completed
            persist()
        state.update(status='budget_completed', finished_at=now())
        persist()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     error=str(exc), traceback=traceback.format_exc(), training_supervisor_pid=None,
                     child_pid=None, eval_pid=None)
        persist()
        raise


if __name__ == '__main__':
    main()
