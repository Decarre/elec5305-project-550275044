"""Frozen protocol v2: absence-aware metrics, timestamp bins and weak-label retraining."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from instrument_localization.alignment import aggregate_time_grid
from instrument_localization.config import ExperimentConfig,load_config
from instrument_localization.metrics import multilabel_metrics,select_balanced_thresholds
from instrument_localization.events import event_metrics
from instrument_localization.training import train_arrays,load_checkpoint,predict_probabilities,load_npz_dataset


def frame_predictions(model,features,batch=32):
    scores=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(features),batch):
            p=model(torch.as_tensor(features[start:start+batch],dtype=torch.float32))['frame_probabilities'].numpy()
            scores.append(aggregate_time_grid(p))
    return np.concatenate(scores)


def evaluate(model,checkpoint,data,original_data):
    output={}
    for name,root in [('natural_subset',original_data),('controlled',data)]:
        vx,vy=load_npz_dataset(root/'validation.npz')
        tx,ty=load_npz_dataset(root/'test.npz')
        with np.load(str(root/'validation_temporal.npz')) as f: vr=f['references']
        with np.load(str(root/'test_temporal.npz')) as f:
            tr=f['references'];variants=f['variants'] if 'variants' in f else np.asarray(['original']*len(tx))
        vs=frame_predictions(model,vx);ts=frame_predictions(model,tx)
        thresholds=select_balanced_thresholds(vr,vs)
        clip_scores=predict_probabilities(model,tx,32,'cpu')
        output[name]={'frame_thresholds':thresholds.tolist(),'frame':multilabel_metrics(tr,ts,thresholds),'clip':multilabel_metrics(ty,clip_scores,checkpoint['validation_thresholds']), 'variants':{}}
        output[name]['events']=event_metrics(tr,ts,thresholds)
        for variant in np.unique(variants):
            mask=variants==variant
            output[name]['variants'][str(variant)]=multilabel_metrics(tr[mask],ts[mask],thresholds)
            output[name]['variants'][str(variant)]['events']=event_metrics(tr[mask],ts[mask],thresholds)
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--old-runs',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    a.results.mkdir(parents=True,exist_ok=True)
    protocol={'seeds':[5305,5306,5307],'models':['baseline','attention','max'],'selection':'validation clip macro balanced accuracy; frame thresholds validation balanced accuracy','primary':'controlled frame macro balanced accuracy','model_choice':'mean validation clip macro balanced accuracy across seeds; test never used for choice','alignment':'centre timestamps t*320/16000, [left,right), discard t=5.0','training':'weak clip labels only; original+absence+transition, source split unchanged'}
    (a.results/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for root_name,root in [('natural_subset',a.original),('controlled',a.data)]:
        _,ty=load_npz_dataset(root/'test.npz')
        with np.load(str(root/'test_temporal.npz')) as f:tr=f['references']
        bx,by=load_npz_dataset(root/'train.npz')
        baselines={}
        for name,scores in [('always_positive',np.ones(ty.shape)),('always_negative',np.zeros(ty.shape)),('training_prevalence',np.broadcast_to(by.mean(axis=0),ty.shape))]:
            fs=np.broadcast_to(scores[:,None,:],tr.shape)
            baselines[name]={'clip':multilabel_metrics(ty,scores),'frame':multilabel_metrics(tr,fs)}
        (a.results/(root_name+'_baselines.json')).write_text(json.dumps(baselines,indent=2))
    for model in ['mean','attention']:
        for seed in protocol['seeds']:
            dest=a.results/('legacy_'+model+'_'+str(seed)+'.json')
            if dest.exists():continue
            m,_,checkpoint=load_checkpoint(a.old_runs/(model+'_seed'+str(seed))/'best_checkpoint.pt')
            result=evaluate(m,checkpoint,a.data,a.original)
            dest.write_text(json.dumps({'kind':'legacy checkpoint, new timestamp and validation-balanced frame thresholds','model':model,'seed':seed,'evaluation':result},indent=2))
            print(dest.name,flush=True)
    tx,ty=load_npz_dataset(a.data/'train.npz');vx,vy=load_npz_dataset(a.data/'validation.npz')
    for model in protocol['models']:
        for seed in protocol['seeds']:
            dest=a.results/(model+'_'+str(seed)+'.json')
            if dest.exists():continue
            base=load_config(ROOT/'configs'/'babyslakh_attention.yaml')
            config=ExperimentConfig(**{**base.to_dict(),'model_type':model,'seed':seed}).validate()
            run_dir=a.runs/(model+'_seed'+str(seed))
            print('TRAIN',model,seed,flush=True)
            summary=train_arrays(tx,ty,vx,vy,config,run_dir,selection_objective='balanced_accuracy',device='cpu')
            m,_,checkpoint=load_checkpoint(run_dir/'best_checkpoint.pt')
            dest.write_text(json.dumps({'kind':'controlled retraining','model':model,'seed':seed,'validation':summary,'evaluation':evaluate(m,checkpoint,a.data,a.original)},indent=2))
            print('DONE',dest.name,flush=True)


if __name__=='__main__':main()
