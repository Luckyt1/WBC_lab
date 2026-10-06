"""Observe an existing supervised training segment without restarting its trainer."""
import json
from pathlib import Path
import subprocess
import time


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if fields[0] == 'Z':
        return None
    return fields[19]  # Linux process start time; reject PID reuse.


class AdoptedTrainingProcess:
    def __init__(self, output):
        self.output = Path(output).resolve()
        state = json.loads((self.output/'state.json').read_text())
        if state['status'] != 'training' or state.get('eval_pid') is not None:
            raise ValueError('Only an active, non-evaluating training segment can be adopted')
        self.pid = state['supervisor_pid']
        self.identity = process_identity(self.pid)
        if self.identity is None or process_identity(state['child_pid']) is None:
            raise ValueError('Training supervisor and trainer must both be alive')
        args = Path(f'/proc/{self.pid}/cmdline').read_bytes().decode().split('\0')
        if ('tools/run_walk_robust_training.py' not in args or '--output' not in args
                or Path(args[args.index('--output')+1]).resolve() != self.output):
            raise ValueError('Supervisor command does not match the training segment')
        self.returncode = None

    def poll(self):
        identity = process_identity(self.pid)
        if identity is not None:
            if identity != self.identity:
                raise RuntimeError('Adopted supervisor PID was reused')
            return None
        state = json.loads((self.output/'state.json').read_text())
        complete = (state['status'] == 'completed' and state.get('child_pid') is None
                    and state.get('eval_pid') is None)
        self.returncode = 0 if complete else 1
        return self.returncode

    def wait(self, timeout):
        end = time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic() >= end:
                raise subprocess.TimeoutExpired(str(self.output), timeout)
            time.sleep(.1)
        return self.returncode
