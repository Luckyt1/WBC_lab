"""Evaluate training-only heads on one frozen source-policy probe dataset (CPU)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rsl_rl'))
from rsl_rl.modules.actor_auxiliary import ActorAuxiliary


def mean_or_none(values):
    """Early scratch policies can fall before any measurement sample survives."""
    return values.mean(0).tolist() if len(values) else [None]*values.shape[1]


@torch.inference_mode()
def evaluate(checkpoint, probes):
    torch.set_num_threads(2)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    aux=ActorAuxiliary(**saved['auxiliary_cfg']);aux.load_state_dict(saved['auxiliary_state_dict']);aux.eval()
    actor=torch.nn.Sequential(torch.nn.Linear(840,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),
                              torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,15)).eval()
    actor.load_state_dict({k.removeprefix('actor.'):v for k,v in saved['model_state_dict'].items() if k.startswith('actor.')})
    policy=SimpleNamespace(actor=actor)
    config=yaml.load((probes/'base_env.yaml').read_text(),Loader=yaml.BaseLoader)
    scales=config['normalization']['obs_scales']
    profiles={}
    for path in sorted(probes.glob('*_auxiliary_probe.npz')):
        a=np.load(path);hist={}
        for start in range(0,len(a['observations']),1024):
            chunk=slice(start,start+1024)
            p=aux.predict(policy,torch.from_numpy(a['observations'][chunk]),torch.from_numpy(a['applied_action'][chunk]))
            for k,v in p.items():hist.setdefault(k,[]).append(v.numpy())
        pred={k:np.concatenate(v) for k,v in hist.items()}
        live=a['alive'];valid=a['valid'];truth=a['contact'][live].astype(bool)
        contact=pred['contact_logits'][live]>0
        metrics=dict(samples=int(live.sum()),velocity_mae=mean_or_none(np.abs(pred['velocity'][live]-a['velocity'][live])),
                     zero_velocity_baseline_mae=mean_or_none(np.abs(a['velocity'][live])),
                     contact_accuracy=mean_or_none(contact==truth),contact_sides=[])
        # Command is not a measured velocity. Keep this approximate baseline
        # explicit: its XY frame differs slightly when the pelvis is tilted.
        command=a['observations'][:, -84+6:-84+9]/float(scales['commands'])
        proxy=np.column_stack((command[:,:2],np.zeros(len(command))))
        metrics['command_velocity_proxy_mae_approx']=mean_or_none(np.abs(proxy[live]-a['velocity'][live]))
        metrics['velocity_by_scenario']={}
        for scenario in np.unique(a['scenario']):
            mask=live & (a['scenario']==scenario)
            metrics['velocity_by_scenario'][str(int(scenario))]=dict(samples=int(mask.sum()),
                estimated_mae=mean_or_none(np.abs(pred['velocity'][mask]-a['velocity'][mask])),
                zero_baseline_mae=mean_or_none(np.abs(a['velocity'][mask])))
        for side in range(2):
            y,z=truth[:,side],contact[:,side]
            stance=float(z[y].mean()) if y.any() else None
            swing=float((~z[~y]).mean()) if (~y).any() else None
            metrics['contact_sides'].append(dict(stance_recall=stance,swing_recall=swing,
                balanced_accuracy=(stance+swing)/2 if stance is not None and swing is not None else None,
                true_swing_samples=int((~y).sum()),always_contact_accuracy=float(y.mean()) if len(y) else None))
        if aux.mode=='dynamics':
            metrics['dynamics_samples']=int(valid.sum())
            for k in ('joint_velocity','angular_velocity'):
                truth_next=a['next_'+k][valid]
                metrics[k+'_mae']=mean_or_none(np.abs(pred[k][valid]-truth_next))
                metrics[k+'_persistence_mae']=mean_or_none(np.abs(a['current_'+k][valid]-truth_next))
                columns=slice(-84+26,-84+41) if k=='joint_velocity' else slice(-84,-81)
                scale=float(scales['joint_vel' if k=='joint_velocity' else 'ang_vel'])
                metrics[k+'_observed_persistence_mae']=mean_or_none(np.abs(a['observations'][valid,columns]/scale-truth_next))
        profiles[path.stem]=metrics
    if not profiles:raise ValueError('No held-out auxiliary probe datasets found')
    return dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                mode=aux.mode,updates=int(aux.updates),profiles=profiles,
                method='Identical held-out source-policy trajectories for all heads; actor histories only, '
                       'actual applied targets supplied only to the forward decoder; terminal transitions masked for dynamics.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--probes',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.write_text(json.dumps(evaluate(args.checkpoint,args.probes),indent=2,allow_nan=False)+'\n')
