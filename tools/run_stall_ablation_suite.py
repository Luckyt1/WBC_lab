"""Sequential five-arm screen, with predeclared outcomes and optional second-seed replication."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_focused_validation import ROOT, archive_sources, sha, write, now

VARIANTS=('baseline','knee','guidance','exploration','combined')


def summarize(report):
    rows={r['scenario']:r for r in report['scenarios']}
    names=('forward','backward','turn_pos','turn_neg')
    axes=(0,0,2,2)
    valid=all('mae' in rows[n] for n in names)
    score=(sum(rows[n]['mae'][a]/(.08 if a==0 else .15) for n,a in zip(names,axes))/4) if valid else None
    stand=rows['stand']
    stand_ok=(stand['survived']>=31 and stand.get('drift_survivors_mean_m') is not None
              and stand['drift_survivors_mean_m']<=.05)
    full=stand_ok
    for row in rows.values():
        full &= (row['survived']>=31 and len(row.get('mae',[]))==5
                 and all(e<=t for e,t in zip(row.get('mae',[]),(.08,.08,.15,.03,.1)))
                 and len(row.get('knee_clipped_fraction',[]))==2
                 and max(row.get('knee_clipped_fraction',[1]))<=.5)
    return dict(score=score,standing_passed=stand_ok,all_basic_tasks_passed=bool(full),
                scenarios={n:{k:r.get(k) for k in ('survived','actual_mean','mae','single_support_fraction',
                            'knee_clipped_fraction','drift_survivors_mean_m')} for n,r in rows.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--updates',type=int,default=300)
    p.add_argument('--num-envs',type=int,default=1024)
    args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'launch.lock').open('x') as f:f.write(str(os.getpid()))
    archive_sources(out)
    checkpoint=Path(args.checkpoint).resolve()
    state=dict(status='running',started_at=now(),supervisor_pid=os.getpid(),checkpoint=str(checkpoint),
               checkpoint_sha256=sha(checkpoint),updates_per_case=args.updates,num_envs=args.num_envs,
               variants=list(VARIANTS),training_seed=42,replication_seed=137,results=[],child_pid=None,
               protocol='Same checkpoint including optimizer, equal updates/envs/rollout length. Independent output directories. Paired seed 42 screen. Best candidate repeats against baseline at seed 137 only if normalized tracking score improves >=20%, standing passes, and no task loses >2 survivors. Never replace production checkpoint automatically.')
    def persist():state['updated_at']=now();write(out/'state.json',state)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def run(variant,seed):
        dest=out/f'{variant}_seed{seed}';dest.mkdir()
        argv=[sys.executable,'-u','tools/run_stall_ablation_case.py','--variant',variant,
              '--checkpoint',str(checkpoint),'--output',str(dest),'--updates',str(args.updates),
              '--num-envs',str(args.num_envs),'--seed',str(seed),'--headless']
        with (dest/'console.log').open('x') as stream:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            state.update(current_variant=variant,current_seed=seed,child_pid=child.pid);persist()
            code=child.wait()
        state['child_pid']=None
        if code:raise RuntimeError(f'{variant}/{seed} exited {code}; see {dest}/console.log')
        report=json.loads((dest/'report.json').read_text())
        assert report['checkpoint_sha256']==state['checkpoint_sha256']
        summary=dict(variant=variant,seed=seed,report_dir=str(dest),**summarize(report))
        state['results'].append(summary);persist()
        print('CASE_COMPLETE',json.dumps(summary),flush=True)
        return summary
    try:
        persist()
        primary=[run(v,42) for v in VARIANTS]
        baseline=primary[0]
        candidates=[]
        for trial in primary[1:]:
            retention=all(trial['scenarios'][n]['survived']>=r['survived']-2 for n,r in baseline['scenarios'].items())
            if (trial['score'] is not None and baseline['score'] is not None
                and trial['score']<=.8*baseline['score'] and trial['standing_passed'] and retention):
                candidates.append(trial)
        if candidates:
            best=min(candidates,key=lambda x:x['score'])
            state['replication_candidate']=best['variant'];persist()
            run('baseline',137);run(best['variant'],137)
        else:
            state['replication_candidate']=None
            state['replication_reason']='No candidate met the predeclared improvement+retention screen.'
        state.update(status='completed',completed_at=now(),child_pid=None);persist()
    except BaseException as exc:
        state.update(status='failed',error=str(exc));persist();raise


if __name__=='__main__':main()
