"""Deterministic bounded RRT-Connect repair, preserving every target and split.

This is our own joint-space RRT-Connect implementation, not an OMPL binding.
Collision checking uses both arms together, dense edges, and final MuJoCo checks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from repair_ik_arm_bank import PathBuilder
from elf3_ik_core import forward_targets


def rrt_connect(start,goal,low,high,edge_valid,rng,max_nodes=500,step=.3,deadline=None):
    """Return a start->goal polyline, or None after the explicit finite budget."""
    if edge_valid(start,goal): return np.array([start,goal])
    trees=[([np.array(start)],[-1]),([np.array(goal)],[-1])]
    def extend(tree,target):
        nodes,parents=tree
        nearest=int(np.argmin(np.linalg.norm(np.asarray(nodes)-target,axis=1)))
        delta=target-nodes[nearest];length=np.abs(delta).max()
        q=nodes[nearest]+delta*min(1.,step/max(length,1e-12))
        if not edge_valid(nodes[nearest],q): return None,False
        nodes.append(q);parents.append(nearest)
        return len(nodes)-1,length<=step+1e-10
    def branch(tree,index):
        values=[]
        while index>=0: values.append(tree[0][index]);index=tree[1][index]
        return values[::-1]
    for k in range(max_nodes):
        if deadline is not None and time.monotonic()>=deadline: break
        side=k%2;a,b=trees[side],trees[1-side]
        target=b[0][0] if rng.random()<.2 else rng.uniform(low,high)
        idx,_=extend(a,target)
        if idx is None: continue
        for _ in range(80):
            other,done=extend(b,a[0][idx])
            if other is None: break
            if done:
                pa,pb=branch(a,idx),branch(b,other)
                return np.array(pa+pb[-2::-1] if side==0 else pb+pa[-2::-1])
    return None


class SearchBuilder(PathBuilder):
    def edge(self,a,b):
        n=max(2,int(np.ceil(np.abs(b-a).max()/.035))+1)
        return bool(self.clearances(np.linspace(a,b,n)).min()>=.0035)

    def search(self,target,old,budget_s=120.):
        deadline=time.monotonic()+budget_s
        seed=int.from_bytes(hashlib.sha256(target['id'].encode()).digest()[:4],'little')
        rng=np.random.default_rng(seed)
        limits=np.asarray(self.m['joint_limits'])
        gp,gq=np.asarray(target['positions']),np.asarray(target['orientations_wxyz'])
        homep,homeq=forward_targets(self.solver,self.default)
        home=dict(positions=homep,orientations_wxyz=homeq)
        attempts=0
        for k in range(512):
            if time.monotonic()>=deadline: break
            if k==0: endpoint=np.asarray(target['witness_arm_q'])
            elif k==1: endpoint=old[0,-1]
            else:
                start=(np.clip(np.asarray(target['witness_arm_q'])+rng.normal(0,1.1 if k<64 else 1.5,14),limits[:,0],limits[:,1])
                       if k%2==0 else rng.uniform(limits[:,0],limits[:,1]))
                self.solver.update_measured(self.module.ARM_JOINTS,start)
                self.solver.iterations=96
                result=self.solver.solve(gp,gq)
                if result.positions is None: continue
                endpoint=result.positions.astype(float)
            err=self.error(endpoint,target)
            if err[0]>.005 or err[1]>.03 or self.clearances([endpoint]).min()<.0035: continue
            attempts+=1
            # Reserve time for alternate IK branches if this tree cannot connect.
            route=rrt_connect(self.default,endpoint,limits[:,0],limits[:,1],self.edge,rng,
                              max_nodes=400,deadline=min(deadline,time.monotonic()+15.))
            if route is None: continue
            # Deterministic collision-checked shortcuts reduce unnecessary turns.
            for _ in range(60):
                if len(route)<=2: break
                i,j=sorted(rng.choice(len(route),2,replace=False))
                if j>i+1 and self.edge(route[i],route[j]): route=np.concatenate((route[:i+1],route[j:]))
            path=self.solve_guide(self.guide(route))
            if path is None: continue
            outbound=self.valid(path,target)
            if not outbound: continue
            back=path[::-1].copy()
            inbound=self.valid(back,home)
            if inbound:
                return np.stack((path,back)),dict(method='rrt_connect_upstream_ik',outbound=outbound,inbound=inbound,
                    endpoint_branches=attempts,seed=seed,route_waypoints=route.tolist())
        return None,dict(method='rejected',reason=('no_collision_clear_endpoint_within_ik_budget' if attempts==0
                                                 else 'no_verified_path_within_rrt_and_ik_budget'),
                         endpoint_branches=attempts,search_budget_s=budget_s,seed=seed)


BUILDER=None
def initialize(manifest,xml):
    global BUILDER
    BUILDER=SearchBuilder(manifest,xml)

def repair(arguments):
    target,old=arguments
    path,report=BUILDER.search(target,old)
    return target['id'],path,report

def minimum_durations(paths,recipe):
    # Conservative amplitude/total-variation timing heuristic. Ruckig separately
    # enforces actual command derivative limits, including live waist correction.
    travel=np.abs(np.diff(paths,axis=2)).sum(2).max(-1)
    return np.maximum.reduce([1.875*travel/recipe.get('planned_max_velocity',recipe['max_velocity']),
        np.sqrt(6*travel/recipe.get('planned_max_acceleration',recipe['max_acceleration'])),
        np.cbrt(60*travel/recipe.get('planned_max_jerk',recipe['max_jerk'])),np.full_like(travel,.5)]).astype(np.float32)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--xml',type=Path,required=True)
    p.add_argument('--original-bank',type=Path,help='Original v1 bank, if the v2 parent bank has moved')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    m=json.loads((args.source/'manifest.json').read_text())
    original=args.original_bank or Path(m['parent_bank'])
    if not original.is_absolute():
        original=(Path.cwd() if args.original_bank else args.source)/original
    if hashlib.sha256((original/'manifest.json').read_bytes()).hexdigest()!=m['parent_manifest_sha256']:
        raise ValueError('Original bank manifest does not match the v2 parent hash')
    paths={};old={};reports={}
    for split in ('train','test'):
        data=np.load(args.source/(split+'.npz'));paths.update(zip(data['ids'],data['paths']))
        data=np.load(original/(split+'.npz'));old.update(zip(data['ids'],data['paths']))
    targets=m['targets']+m.get('rejected_targets',[])
    for t in m['targets']: reports[t['id']]=t['execution_validation']
    rejected=m.get('rejected_targets',[])
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(m,args.xml)) as pool:
        for future in as_completed([pool.submit(repair,(t,old[t['id']])) for t in rejected]):
            ident,path,report=future.result();reports[ident]=report
            if path is not None: paths[ident]=path
            print(json.dumps(dict(id=ident,repaired=path is not None,**report)),flush=True)
            (args.output/'search_progress.json').write_text(json.dumps(reports,indent=2)+'\n')
    accepted=[];failed=[]
    for t in sorted(targets,key=lambda t:t['id']):
        t['execution_validation']=reports[t['id']]
        (accepted if t['id'] in paths else failed).append(t)
    m.update(version=3,targets=accepted,rejected_targets=failed,parent_bank=str(args.source.resolve()),
        parent_manifest_sha256=hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
        counts=dict(train=sum(t['split']=='train' for t in accepted),test=sum(t['split']=='test' for t in accepted),
                    hard=sum(t['hard'] for t in accepted),rejected=len(failed)))
    m['recipe'].update(max_velocity=6.,max_acceleration=80.,max_jerk=1000.,
        planned_max_velocity=2.,planned_max_acceleration=4.,planned_max_jerk=20.,
        executor='predictive_ruckig_v3',prediction_times_s=[.02,.08,.2],dynamic_extra_margin_m=.005,
        unsafe_recovery='bounded_outward_improving_clearance',
        timing='total_variation_minimum_duration',max_episode_s=40.,arrival_timeout_s=5.,
        arrival_dwell_s=.1,
        planning_failure='hold_neutral_and_report_failure')
    for split in ('train','test'):
        subset=[t for t in accepted if t['split']==split];array=np.stack([paths[t['id']] for t in subset]).astype(np.float32)
        minimum=minimum_durations(array,m['recipe'])
        np.savez_compressed(args.output/(split+'.npz'),paths=array,min_durations=minimum,
            ids=np.array([t['id'] for t in subset]),hard=np.array([t['hard'] for t in subset]),groups=np.array([t['group'] for t in subset]))
    m['bank_sha256']={s:hashlib.sha256((args.output/(s+'.npz')).read_bytes()).hexdigest() for s in ('train','test')}
    m['generation']['v3']='Reuse validated v2 paths; rejected goals get bounded seeded bidirectional RRT-Connect, alternate upstream IK seeds, dense OBB edge checks, upstream waypoint IK and final MuJoCo validation. Keep rejected goals and split unchanged.'
    m['limitations']='Sampled finite-horizon collision prediction, finite planning budget, reference derivative limits only; no continuous-time or hardware safety guarantee. A rejected target is an explicit evaluation failure.'
    (args.output/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
    print(json.dumps(m['counts']),flush=True)

if __name__=='__main__':main()
