"""Deployment contracts: paths survive moves, dry-run is inert, archives are scoped."""
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from scripts import runtime_env
from scripts.pack_release import include
from scripts.train_current import build_command, validate_config
from legged_lab import paths

ROOT = Path(__file__).resolve().parents[1]


def test_live_recipe_and_argument_boundaries():
    recipe = json.loads((ROOT/'configs/current.json').read_text())
    command = build_command(recipe, '/output/a b', '/python/eval', '/python/ik', '/bank/a b', '/train/python')
    assert command[0] == '/train/python'
    assert command[command.index('--ik_arm_bank')+1] == '/bank/a b'
    start = command.index('--pelvis_height_range')
    assert command[start+1:start+3] == ['0.4', '0.8']
    assert command[command.index('--num_envs')+1] == '12556'
    assert command[command.index('--height_curriculum_updates')+1] == '20000'
    assert not any('waist_yaw_range' in value for value in command)


@pytest.mark.parametrize('key,value', [('num_envs', 0), ('max_iterations', -1),
    ('evaluation_interval', 0), ('pelvis_height_range', [.8, .4]), ('waist_yaw_range', [-.2, .2])])
def test_invalid_or_unrequested_recipe_is_rejected(key, value):
    recipe = json.loads((ROOT/'configs/current.json').read_text())
    recipe[key] = value
    with pytest.raises(ValueError):
        validate_config(recipe)


def test_dry_run_from_other_directory_does_not_create_run(tmp_path):
    output = tmp_path/'a run with spaces'
    python = '/opt/isaac with spaces/bin/python'
    env = dict(os.environ, ELF3_TRAIN_PYTHON=python, PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([str(ROOT/'deploy.sh'), 'train', '--dry-run', '--num-envs', '128',
                             '--output', str(output)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True)
    command = shlex.split(result.stdout)
    assert command[0] == python
    assert command[command.index('--output')+1] == str(output)
    assert command[command.index('--num_envs')+1] == '128'
    assert not output.exists()


def test_env_is_data_and_keeps_explicit_environment(tmp_path, monkeypatch):
    p = tmp_path/'.env'
    p.write_text("ELF3_IK_BANK='resources/a bank'\nELF3_TRAIN_PYTHON='/a/python'\n")
    monkeypatch.delenv('ELF3_IK_BANK', raising=False)
    monkeypatch.setenv('ELF3_TRAIN_PYTHON', '/explicit/python')
    runtime_env.load_env(p)
    assert os.environ['ELF3_IK_BANK'] == 'resources/a bank'
    assert os.environ['ELF3_TRAIN_PYTHON'] == '/explicit/python'


def test_resource_paths_are_relative_to_checkout_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ELF3_IK_BANK', 'resources/ik_bank_v3')
    monkeypatch.setenv('ELF3_ASSET_DIR', 'resources/robot with spaces')
    assert paths.ik_bank_dir() == ROOT/'resources/ik_bank_v3'
    assert paths.robot_asset_dir() == ROOT/'resources/robot with spaces'


@pytest.mark.parametrize('name', ['.env', '.env.backup', '.git/config', 'logs/state.json',
    '.venv-ik/pyvenv.cfg', 'resources/private.key', 'resources/elf3_bxi_31/key.pem',
    'resources/bxi_example_py_elf3/model.onnx', 'legged_lab/__pycache__/x.pyc'])
def test_transfer_bundle_excludes_machine_state_and_weights(name):
    assert not include(Path(name), with_resources=True)


def test_source_archive_excludes_training_inputs_but_keeps_fixtures():
    assert not include(Path('resources/ik_bank_v3/train.npz'), with_resources=False)
    assert include(Path('resources/ik_bank_v3/train.npz'), with_resources=True)
    assert include(Path('tests/fixtures/ik_banks/v3/train.npz'), with_resources=False)
