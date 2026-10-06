"""Compare the port against original WBC formulas on the same simulator state."""
import ast
from types import SimpleNamespace

import torch
import yaml


def check_reward_parity(env, reference_root):
    from legged_lab.envs.elf3_tang.wbc import rewards

    reference_cfg = reference_root / 'logs/elf3_wbc/wbc_g1_pelvis_min050_from2500_20260920/params/env.yaml'
    cfg = yaml.load(reference_cfg.read_text(), Loader=yaml.BaseLoader)['reward']
    expected_names = {name for name, term in cfg.items() if float(term['weight']) != 0}
    assert set(env.reward_manager.active_terms) == set(cfg)
    source = reference_root / 'legged_lab/mdp/rewards.py'
    names = {term['func'].split(':')[-1] for term in cfg.values()}
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             *definitions], type_ignores=[])
    namespace = dict(vars(rewards))
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)

    class WBCView:
        def __init__(self):
            self.command_generator = SimpleNamespace(
                command=env.command_generator.wbc_command,
                is_standing_env=env.command_generator.is_standing_env,
            )

        def __getattr__(self, key):
            return getattr(env, key)

    view = WBCView()
    errors = {}
    for name in sorted(expected_names):
        term = env.reward_manager.get_term_cfg(name)
        original = cfg[name]
        assert term.weight == float(original['weight']), name
        assert set(term.params) == set(original['params']), name
        for key, value in original['params'].items():
            actual = term.params[key]
            if isinstance(value, dict):
                for field in ('name', 'joint_names', 'body_names'):
                    if value.get(field, 'null') != 'null':
                        expected_field = value[field]
                        if field.endswith('_names') and isinstance(expected_field, str):
                            expected_field = [expected_field]
                        assert getattr(actual, field) == expected_field, (name, field)
                assert actual.preserve_order == (value['preserve_order'] == 'true'), name
            else:
                assert actual == yaml.safe_load(value), (name, key)
        expected = namespace[original['func'].split(':')[-1]](view, **term.params)
        actual = term.func(env, **term.params)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6, msg=name)
        errors[name] = float((actual.to(torch.float32) - expected.to(torch.float32)).abs().max())
    print(f'WBC_REWARD_PARITY_PASS: {len(errors)} active terms; max error={max(errors.values())}', flush=True)
    return {'active_terms': len(errors), 'max_absolute_error': max(errors.values()), 'per_term': errors}
