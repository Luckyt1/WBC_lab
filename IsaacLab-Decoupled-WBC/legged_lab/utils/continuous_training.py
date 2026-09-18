"""Unbounded RSL-RL continuation with stop requests handled after a full update."""

from pathlib import Path
import signal


class _StopAfterUpdate(Exception):
    pass


def learn_continuously(runner, checkpoint_dir, on_progress=None, batch_size=None):
    """Keep the same simulator and optimizer alive until SIGINT/SIGTERM.

    RSL-RL records the last completed iteration, so advance its index between
    learn() calls. A stop request during rollout/update is deferred until log(),
    which the runner calls after completing the update and setting its index.
    """
    if runner.log_dir is None or runner.disable_logs:
        raise ValueError('Continuous training requires the per-iteration logging hook')
    batch_size = runner.save_interval if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    checkpoint_dir = Path(checkpoint_dir)
    stop_requested = False
    last_completed = runner.current_learning_iteration - 1
    original_log = runner.log
    previous_handlers = {}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    def log_after_update(*args, **kwargs):
        nonlocal last_completed
        original_log(*args, **kwargs)
        last_completed = runner.current_learning_iteration
        if stop_requested:
            raise _StopAfterUpdate

    runner.log = log_after_update
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        first_batch = True
        try:
            while not stop_requested:
                runner.learn(num_learning_iterations=batch_size, init_at_random_ep_len=first_batch)
                first_batch = False
                last_completed = runner.current_learning_iteration
                if on_progress is not None:
                    on_progress(last_completed)
                if not stop_requested:
                    runner.current_learning_iteration = last_completed + 1
        except _StopAfterUpdate:
            pass
        # Only completed updates reach this point; errors propagate without
        # writing a potentially half-updated optimizer as a resumable checkpoint.
        runner.current_learning_iteration = last_completed
        destination = checkpoint_dir / f'model_{last_completed}.pt'
        temporary = checkpoint_dir / f'.model_{last_completed}.stopping.pt'
        runner.save(str(temporary))
        temporary.replace(destination)
        if on_progress is not None:
            on_progress(last_completed)
        print(f'TRAINING_STOPPED_AND_SAVED {destination}', flush=True)
        return destination
    finally:
        runner.log = original_log
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
