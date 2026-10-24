"""Resolve this checkout and its local configuration without importing Isaac."""
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]


def load_env(path=None):
    path = Path(path) if path else ROOT / '.env'
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:]
        key, sep, raw = line.partition('=')
        key = key.strip()
        if not sep or not key.startswith('ELF3_'):
            raise ValueError(f'{path}:{number}: expected ELF3_NAME=value')
        values = shlex.split(raw, comments=True)
        if len(values) > 1:
            raise ValueError(f'{path}:{number}: quote paths containing spaces')
        os.environ.setdefault(key, values[0] if values else '')


def project_path(value):
    path = Path(value).expanduser()
    return path.absolute() if path.is_absolute() else (ROOT / path).absolute()


def python_path(role):
    variables = {'train': 'ELF3_TRAIN_PYTHON', 'eval': 'ELF3_DEPLOYMENT_PYTHON', 'ik': 'ELF3_IK_PYTHON'}
    return project_path(os.environ.get(variables[role], f'.venv-{role}/bin/python'))


def environment():
    load_env()
    env = os.environ.copy()
    paths = [str(ROOT), str(ROOT / 'rsl_rl'),
             str(project_path(env.get('ELF3_BXI_FRAMEWORK_DIR', 'resources/bxi_example_py_elf3')))]
    paths += [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p]
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(paths))
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('MKL_NUM_THREADS', '2')
    return env
