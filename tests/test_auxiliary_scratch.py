"""Cold-start evidence must reject inherited state and unequal comparisons."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'rsl_rl'))
from audit_auxiliary_initialization import audit
from evaluate_actor_auxiliary import evaluate
from rsl_rl.modules.actor_auxiliary import ActorAuxiliary


def make_run(run):
    run.mkdir()
    fresh = dict(from_scratch=True, checkpoint_loaded=None, teacher_checkpoint=None,
                 iteration_before_learning=0, optimizer_state_entries=0)
    meta = dict(**fresh, variant='control', checkpoint=None, checkpoint_sha256=None,
                task='walk_elf3_tang_bxi_robust', seed=42, num_envs=64,
                reward_sha256='reward', command_sha256='command',
                environment_sha256='environment', algorithm_sha256='algorithm')
    saved = dict(iter=0, infos=dict(from_scratch=True, before_first_update=True),
                 model_state_dict={'actor.0.weight':torch.zeros(2,840), 'std':torch.full((15,),.5)},
                 optimizer_state_dict=dict(state={},param_groups=[dict(lr=3e-4)]))
    (run/'fresh_start.json').write_text(json.dumps(fresh))
    (run/'auxiliary_supervision.json').write_text(json.dumps(meta))
    torch.save(saved, run/'initial_policy.pt')
    return saved


@pytest.mark.parametrize('change', ['optimizer', 'iteration', 'weights', 'environment', 'teacher'])
def test_initialization_audit_rejects_inheritance_or_unmatched_recipes(tmp_path, change):
    make_run(tmp_path/'control')
    baseline = audit(tmp_path/'control')
    saved = make_run(tmp_path/'candidate')
    assert audit(tmp_path/'candidate',baseline)['policy_state_sha256'] == baseline['policy_state_sha256']
    if change == 'optimizer':
        saved['optimizer_state_dict']['state']={0:{'step':1}}
    elif change == 'iteration':
        saved['iter']=26000
    elif change == 'weights':
        saved['model_state_dict']['actor.0.weight'][0,0]=1.
    elif change == 'environment':
        path=tmp_path/'candidate/auxiliary_supervision.json'
        meta=json.loads(path.read_text());meta['environment_sha256']='different';path.write_text(json.dumps(meta))
    else:
        path=tmp_path/'candidate/fresh_start.json'
        meta=json.loads(path.read_text());meta['teacher_checkpoint']='old.pt';path.write_text(json.dumps(meta))
    torch.save(saved,tmp_path/'candidate/initial_policy.pt')
    with pytest.raises(AssertionError):
        audit(tmp_path/'candidate',baseline)


@pytest.mark.parametrize('one_survivor', [False, True])
def test_probe_report_handles_early_falls_without_nan_or_fabricated_accuracy(tmp_path, one_survivor):
    actor=torch.nn.Sequential(torch.nn.Linear(840,512),torch.nn.ELU(),torch.nn.Linear(512,256),
                              torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,15))
    aux=ActorAuxiliary(mode='dynamics')
    checkpoint=tmp_path/'model.pt'
    torch.save(dict(auxiliary_cfg=dict(mode='dynamics'),auxiliary_state_dict=aux.state_dict(),
                    model_state_dict={'actor.'+k:v for k,v in actor.state_dict().items()}),checkpoint)
    (tmp_path/'base_env.yaml').write_text('normalization:\n  obs_scales:\n    commands: 1\n    joint_vel: 1\n    ang_vel: 1\n')
    data={name:np.zeros((2,width),dtype=np.float32) for name,width in
          [('observations',840),('applied_action',29),('contact',2),('velocity',3),
           ('next_joint_velocity',15),('next_angular_velocity',3),
           ('current_joint_velocity',15),('current_angular_velocity',3)]}
    data.update(alive=np.array([one_survivor,False]),valid=np.zeros(2,dtype=bool),scenario=np.array([0,1]))
    np.savez(tmp_path/'bootstrap_auxiliary_probe.npz',**data)
    report=evaluate(checkpoint,tmp_path)
    json.dumps(report,allow_nan=False)
    metrics=report['profiles']['bootstrap_auxiliary_probe']
    assert metrics['samples']==int(one_survivor)
    assert metrics['velocity_by_scenario']['1']['estimated_mae']==[None]*3
    assert metrics['joint_velocity_mae']==[None]*15
    if not one_survivor:
        assert metrics['contact_accuracy']==[None]*2
        assert metrics['contact_sides'][0]['balanced_accuracy'] is None
