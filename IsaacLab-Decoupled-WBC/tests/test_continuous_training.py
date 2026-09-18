import json
from pathlib import Path
import signal

import pytest

from legged_lab.utils.continuous_training import learn_continuously


class FakeRunner:
    """Mirror installed runner order: update, set iteration, log, checkpoint."""
    def __init__(self, folder, stop_at=None, fail_at=None, signum=signal.SIGINT):
        self.log_dir = str(folder)
        self.disable_logs = False
        self.save_interval = 2
        self.current_learning_iteration = 11
        self.stop_at, self.fail_at, self.signum = stop_at, fail_at, signum
        self.completed = []
        self.init_flags = []
        self.optimizer_steps = 10

    def log(self, values):
        pass

    def save(self, filename):
        Path(filename).write_text(json.dumps(dict(iteration=self.current_learning_iteration,
                                                  optimizer_steps=self.optimizer_steps)))

    def learn(self, num_learning_iterations, init_at_random_ep_len):
        self.init_flags.append(init_at_random_ep_len)
        start = self.current_learning_iteration
        for it in range(start, start + num_learning_iterations):
            if it == self.fail_at:
                raise RuntimeError('update failed')
            if it == self.stop_at:
                signal.raise_signal(self.signum)
            self.optimizer_steps += 1
            self.current_learning_iteration = it
            self.completed.append(it)
            self.log({'it': it})
        self.save(str(Path(self.log_dir)/f'model_{self.current_learning_iteration}.pt'))


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_stop_waits_for_completed_update_and_does_not_repeat_iterations(tmp_path, signum):
    before = signal.getsignal(signum)
    runner = FakeRunner(tmp_path, stop_at=16, signum=signum)
    original_log = runner.log
    progress = []
    saved = learn_continuously(runner, tmp_path, progress.append)
    assert runner.completed == [11, 12, 13, 14, 15, 16]
    assert runner.init_flags == [True, False, False]
    assert progress == [12, 14, 16]
    assert json.loads(saved.read_text()) == dict(iteration=16, optimizer_steps=16)
    assert runner.log == original_log
    assert signal.getsignal(signum) == before


def test_stop_between_batches_saves_last_completed_index(tmp_path):
    runner = FakeRunner(tmp_path)
    def stop_after_batch(iteration):
        if iteration == 12:
            signal.raise_signal(signal.SIGINT)
    saved = learn_continuously(runner, tmp_path, stop_after_batch)
    assert runner.completed == [11, 12]
    assert saved.name == 'model_12.pt'
    assert json.loads(saved.read_text())['optimizer_steps'] == 12


def test_failed_update_is_not_saved_as_successful_checkpoint(tmp_path):
    runner = FakeRunner(tmp_path, fail_at=14)
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(RuntimeError, match='update failed'):
        learn_continuously(runner, tmp_path)
    assert sorted(p.name for p in tmp_path.glob('model_*.pt')) == ['model_12.pt']
    assert signal.getsignal(signal.SIGINT) == before
