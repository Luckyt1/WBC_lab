from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from tools.teleop_curriculum import STAGES, settings, configure, expected_recipe, evaluation_profiles
from tools.teleop_curriculum_gates import TASKS, position_metrics, gate, decide, validate_transition
from tools.audit_auxiliary_resume import verify_recipe
from tools.run_teleop_curriculum import training_command


def protocol(stage='quiet'):
    rows = []
    for name in TASKS:
        rows.append(dict(scenario=name, trials=32, survived=32, valid_samples=100,
            mae=[.001]*5, knee_clipped_fraction={'left':0., 'right':0.},
            position=dict(valid=True, survivor_trials=32, measurement_seconds=55.98,
                          p95_max_drift_m=.02, mean_horizontal_speed_m_s=.002, p95_max_heading_deg=1.)))
    isaac = dict(checkpoint_sha256='weights', teleop_curriculum_stage=stage, repeats=32,
                 warmup_s=4., measure_s=56., arm_guard_enabled=True, height_command=.78,
                 neutral_reference_fraction_in_amass_profiles=0.,
                 profiles=[dict(settings=p,scenarios=deepcopy(rows)) for p in evaluation_profiles(stage)])
    mj = dict(onnx_sha256='onnx',height_command=.78,arms='fixed_training_default',noise=False,push=False,
              results=[dict(case=c,completed=True,valid=True,seconds=60.,measurement_seconds=55.98,
                            max_drift_m=.02,mean_horizontal_speed_m_s=.002)
                       for c in ('zero_command','walk_then_stop')])
    return isaac, mj


@pytest.mark.parametrize('stage', STAGES)
@pytest.mark.parametrize('fraction', [.8, .5])
def test_all_stages_have_fixed_scales_and_preserve_standing_recipe(stage, fraction):
    cfg=NS(arm_motion=NS(amass_scale_end_iteration=20000, amass_neutral_fraction=.3),
           noise=NS(noise_scales=NS()), commands=NS(rel_standing_envs=fraction),
           push_velocity_curriculum=NS(), reset_velocity_curriculum=NS())
    configure(cfg,stage)
    assert cfg.arm_motion.amass_scale_end_iteration is None
    assert cfg.commands.rel_standing_envs==fraction and cfg.arm_motion.amass_neutral_fraction==.3
    assert cfg.push_velocity_curriculum.initial_scale==cfg.push_velocity_curriculum.final_scale
    assert cfg.reset_velocity_curriculum.initial_scale==cfg.reset_velocity_curriculum.final_scale
    assert cfg.commands.focus_lateral_initial_range==cfg.commands.focus_lateral_final_range
    before=deepcopy(vars(cfg.noise.noise_scales))
    configure(cfg,stage)
    assert vars(cfg.noise.noise_scales)==before  # No repeated scaling when restored.


def test_actual_environment_clock_cannot_increase_difficulty():
    path=ROOT/'legged_lab/envs/elf3_tang/elf3_env.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Elf3Env')
    names=('_get_push_velocity_curriculum_progress','_get_reset_velocity_curriculum_progress','_get_arm_curriculum_progress')
    ns={'torch':torch}
    exec(compile(ast.Module(body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(path),'exec'),ns)
    for stage in ('quiet','arms_075','push_020','push_080'):
        cfg=NS(arm_motion=NS(),noise=NS(noise_scales=NS()),commands=NS(),
               push_velocity_curriculum=NS(curriculum_duration_s=9600.),
               reset_velocity_curriculum=NS(curriculum_duration_s=14400.))
        configure(cfg,stage)
        env=NS(cfg=cfg,learning_iteration_dt=.48,physics_dt=.005,use_amass=True,device='cpu')
        for age in (0,5000,30000,1000000):
            env.learning_iteration=age
            assert ns[names[0]](env)==settings(stage)['push_scale']
            assert ns[names[1]](env)==settings(stage)['reset_scale']
            assert float(ns[names[2]](env))==pytest.approx(settings(stage)['arm_scale'])


@pytest.mark.parametrize('defect', ['drift','height','speed','yaw','knees','survival','mujoco','previous_stage'])
def test_gate_cannot_pass_by_survival_or_other_scenarios_alone(defect):
    isaac,mj=protocol('arms_050')
    assert gate(isaac,mj,'arms_050','weights','onnx')['passed']
    row=isaac['profiles'][0]['scenarios'][0]
    if defect=='drift':row['position']['p95_max_drift_m']=.16
    elif defect=='height':row['mae'][3]=.011
    elif defect=='speed':row['mae'][0]=.021
    elif defect=='yaw':row['position']['p95_max_heading_deg']=16.
    elif defect=='knees':row['knee_clipped_fraction']['left']=.011
    elif defect=='survival':row['survived']=30
    elif defect=='mujoco':mj['results'][0]['max_drift_m']=1.24
    else:isaac['profiles'][1]['scenarios'][0]['mae'][0]=.3
    result=gate(isaac,mj,'arms_050','weights','onnx')
    assert not result['passed']
    assert result['regression']==(defect in ('mujoco','previous_stage'))


@pytest.mark.parametrize('defect',['wrong_weights','wrong_profile','short_test','missing_scenario','wrong_onnx'])
def test_incomplete_or_mismatched_evaluation_is_pipeline_error(defect):
    a,m=protocol()
    if defect=='wrong_weights':a['checkpoint_sha256']='other'
    elif defect=='wrong_profile':a['profiles'][0]['settings']['arm_scale']=1.
    elif defect=='short_test':a['measure_s']=12.
    elif defect=='missing_scenario':a['profiles'][0]['scenarios'].pop()
    else:m['onnx_sha256']='other'
    with pytest.raises(ValueError):gate(a,m,'quiet','weights','onnx')


def test_going_out_and_returning_does_not_hide_drift_and_failed_trials_are_counted():
    xy=np.zeros((3000,32,3)); yaw=np.zeros((3000,32)); valid=np.ones((3000,32),dtype=bool)
    xy[1000:2000,:,0]=.4
    failure=np.full(32,-1.); failure[0]=2.; valid[100:,0]=False
    r=position_metrics(xy,yaw,valid,failure,200,.02)
    assert r['p95_max_drift_m']==.4 and r['survivor_trials']==31
    assert r['measurement_seconds']==pytest.approx(55.98)
    xy[300,1,0]=np.nan
    with pytest.raises(ValueError):position_metrics(xy,yaw,valid,failure,200,.02)


def entry(stage, completed, result, decision):
    return dict(stage=stage,completed_updates=completed,gate=result,decision=decision)


def test_promotion_hold_rollback_and_no_clock_only_promotion():
    good=dict(passed=True,regression=False,score=.2)
    bad=dict(passed=False,regression=False,score=2.)
    d=decide([],'quiet',0,5000,good)
    assert d['action']=='hold'
    history=[entry('quiet',5000,good,d)]
    assert decide(history,'quiet',0,6000,good)['next_stage']=='arms_025'
    with pytest.raises(ValueError):decide(history,'quiet',0,5000,good)
    assert decide(history,'quiet',0,6000,bad)['next_stage']=='quiet'
    assert decide([],'quiet',0,999999,bad)['next_stage']=='quiet'
    regression=dict(passed=False,regression=True,score=4.)
    d=decide([],'arms_025',6000,7000,regression)
    assert d['action']=='hold'
    history=[entry('arms_025',7000,regression,d)]
    assert decide(history,'arms_025',6000,8000,regression)['action']=='rollback_difficulty'


def test_stagnation_and_long_stage_age_never_end_training():
    history=[]
    for age in (5000,6000,7000,8000):
        r=dict(passed=False,regression=False,score=10.)
        d=decide(history,'quiet',0,age,r)
        history.append(entry('quiet',age,r,d))
    for completed in (9000,15000,30000,90000):
        assert decide(history,'quiet',0,completed,r)['action']=='hold'
    assert decide(history,'quiet',0,9000,dict(r,score=8.))['action']=='hold'


def test_difficulty_resume_cannot_change_other_rewards_or_inputs(tmp_path):
    base=dict(teleop_curriculum_stage='quiet',arm_motion={},noise=dict(noise_scales={}),commands={},
              push_velocity_curriculum={},reset_velocity_curriculum={},reward={'weight':'-1.'})
    base=expected_recipe(base,'quiet')
    for name,cfg in [('source',base),('target',expected_recipe(base,'arms_025'))]:
        folder=tmp_path/name/'params';folder.mkdir(parents=True)
        (folder/'env.yaml').write_text(yaml.safe_dump(cfg))
        (folder/'agent.yaml').write_text('policy: {}\nalgorithm: {}\nseed: 42\nnum_steps_per_env: 24\nempirical_normalization: false\n')
    assert verify_recipe(tmp_path/'source',tmp_path/'target',curriculum_stage='arms_025')
    p=tmp_path/'target/params/env.yaml';cfg=yaml.safe_load(p.read_text());cfg['reward']['weight']='-2.'
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):verify_recipe(tmp_path/'source',tmp_path/'target',curriculum_stage='arms_025')
    with pytest.raises(ValueError):expected_recipe(base,'arms_100')


def test_fresh_chunk_has_no_model_and_later_chunk_carries_exact_checkpoint_and_decision():
    argv=training_command('/gpu/python','/deploy/python',Path('/out'),'quiet',5000,4096,.78)
    assert '--resume_checkpoint' not in argv and '--external_evaluation' in argv
    argv=training_command('/gpu/python','/deploy/python',Path('/out'),'arms_025',1000,4096,.78,
                          Path('/prior/model_5999.pt'),Path('/decision.json'))
    assert argv[argv.index('--resume_checkpoint')+1]=='/prior/model_5999.pt'
    assert argv[argv.index('--teleop_curriculum_decision')+1]=='/decision.json'


@pytest.mark.parametrize('mode', ['dynamics', 'control'])
@pytest.mark.parametrize('symmetry', [False, True])
def test_balanced_mixture_and_training_mode_are_retained_across_segments(mode, symmetry):
    for checkpoint in [None, Path('/prior/model_4999.pt')]:
        argv = training_command('/gpu/python', '/deploy/python', Path('/out'), 'quiet', 1000, 4096, .8,
                                checkpoint=checkpoint, standing_fraction=.5, symmetry=symmetry,
                                liftoff_rewards=True, auxiliary_supervision=mode)
        assert argv[argv.index('--teleop_standing_fraction')+1] == '0.5'
        assert ('--elf3_symmetry' in argv) == symmetry
        assert argv[argv.index('--auxiliary_supervision')+1] == mode
        assert '--liftoff_rewards' in argv
        assert '--disable_symmetry_on_resume' not in argv


def test_symmetry_removal_is_explicit_and_only_needed_for_first_continuation():
    kwargs = dict(checkpoint=Path('/prior/model_15999.pt'), symmetry=False, liftoff_rewards=True)
    argv = training_command('/gpu/python', '/deploy/python', Path('/out'), 'quiet', 1000, 4096, .8,
                            **kwargs, disable_symmetry_on_resume=True)
    assert '--disable_symmetry_on_resume' in argv
    assert '--elf3_symmetry' not in argv and '--liftoff_rewards' in argv
    subsequent = training_command('/gpu/python', '/deploy/python', Path('/out2'), 'quiet', 1000, 4096, .8,
                                  checkpoint=Path('/new/model_16999.pt'), liftoff_rewards=True)
    assert '--disable_symmetry_on_resume' not in subsequent and '--elf3_symmetry' not in subsequent
    with pytest.raises(ValueError):
        training_command('python', '/deploy', '/out', 'quiet', 1000, 4096, .8,
                         disable_symmetry_on_resume=True)
    with pytest.raises(ValueError):
        training_command('python', '/deploy', '/out', 'quiet', 1000, 4096, .8,
                         checkpoint=Path('/prior/model.pt'), symmetry=True, disable_symmetry_on_resume=True)


def transition_fixture(tmp_path):
    run=tmp_path/'run';(run/'params').mkdir(parents=True)
    (run/'params/env.yaml').write_text('teleop_curriculum_stage: quiet\n')
    history=[]
    for count in (5000,6000):
        cp=run/f'model_{count-1}.pt';cp.write_bytes(str(count).encode())
        out=tmp_path/str(count);out.mkdir()
        a,m=protocol();a['checkpoint_sha256']=hashlib.sha256(cp.read_bytes()).hexdigest()
        model=out/'policy.onnx';model.write_bytes(b'test fixture')
        m['onnx_sha256']=hashlib.sha256(model.read_bytes()).hexdigest()
        (out/'report.json').write_text(json.dumps(a));(out/'mujoco.json').write_text(json.dumps(m))
        g=gate(a,m,'quiet',a['checkpoint_sha256'],m['onnx_sha256'])
        d=decide(history,'quiet',0,count,g)
        evidence={k:dict(path=str(out/f),sha256=hashlib.sha256((out/f).read_bytes()).hexdigest())
                  for k,f in [('isaac','report.json'),('mujoco','mujoco.json'),('onnx','policy.onnx')]}
        path=out/'decision.json'
        path.write_text(json.dumps(dict(stage='quiet',completed_updates=count,checkpoint_sha256=a['checkpoint_sha256'],
            gate=g,decision=d,evidence=evidence,stage_start_updates=0,prior_history=history)))
        history.append(dict(stage='quiet',completed_updates=count,gate=g,decision=d,
                            checkpoint_sha256=a['checkpoint_sha256'],decision_path=str(path),
                            decision_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return cp,path


def test_adjacent_promotion_requires_intact_checkpoint_reports_and_two_decisions(tmp_path):
    cp,path=transition_fixture(tmp_path)
    assert validate_transition(path,cp,'arms_025')['action']=='promote'
    with pytest.raises(ValueError):validate_transition(path,cp,'arms_050')
    artifact=json.loads(path.read_text());artifact['prior_history'][0]['gate']['passed']=False
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):validate_transition(path,cp,'arms_025')


@pytest.mark.parametrize('changed',['checkpoint','report','previous_decision'])
def test_evidence_mutation_cannot_authorize_a_transition(tmp_path,changed):
    cp,path=transition_fixture(tmp_path)
    a=json.loads(path.read_text())
    target=cp if changed=='checkpoint' else Path(a['evidence']['isaac']['path']) if changed=='report' else Path(a['prior_history'][0]['decision_path'])
    target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ValueError):validate_transition(path,cp,'arms_025')
