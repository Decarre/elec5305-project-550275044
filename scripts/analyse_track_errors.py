"""Track-level robustness and conditional FPR diagnostics with support counts."""
import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from instrument_localization.training import load_npz_dataset,load_checkpoint
from instrument_localization.metrics import multilabel_metrics
from run_controlled_experiment import frame_predictions


def conditional_rates(truth,predicted,names):
    truth=np.asarray(truth,dtype=bool).reshape(-1,len(names))
    predicted=np.asarray(predicted,dtype=bool).reshape(truth.shape)
    rows=[]
    for c,name in enumerate(names):
        for d,other in enumerate(names):
            if c==d:continue
            positive=(~truth[:,c])&truth[:,d]
            negative=(~truth[:,c])&(~truth[:,d])
            rows.append({'target':name,'other':other,'other_active_support':int(positive.sum()),'other_absent_support':int(negative.sum()),'false_positives_other_active':int((predicted[:,c]&positive).sum()),'false_positives_other_absent':int((predicted[:,c]&negative).sum()),'fpr_other_active':float(predicted[positive,c].mean()) if positive.any() else None,'fpr_other_absent':float(predicted[negative,c].mean()) if negative.any() else None})
    return rows


def write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--controlled',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    out=ROOT/'results'/'ablation_v4';out.mkdir(parents=True,exist_ok=True)
    per_track=[];conditional=[]
    names=['drums','bass','guitar','piano']
    for dataset,root in [('natural_subset',a.original),('controlled',a.controlled)]:
        x,_=load_npz_dataset(root/'test.npz')
        with np.load(str(root/'test_temporal.npz')) as f:truth=f['references'];track=f['track_ids']
        for model_name in ['baseline','attention']:
            for seed in [5305,5306,5307]:
                result=json.loads((ROOT/'results'/'babyslakh_v2'/(model_name+'_'+str(seed)+'.json')).read_text())
                thresholds=np.asarray(result['evaluation'][dataset]['frame_thresholds'])
                model,_,_=load_checkpoint(a.runs/(model_name+'_seed'+str(seed))/'best_checkpoint.pt')
                score=frame_predictions(model,x)
                for name in np.unique(track):
                    mask=track==name;r=multilabel_metrics(truth[mask],score[mask],thresholds,names)
                    per_track.append({'dataset':dataset,'model':model_name,'seed':seed,'track':str(name),'scenes':int(mask.sum()),'micro_f1':r['micro']['f1'],'macro_balanced_accuracy':r['macro']['balanced_accuracy']})
                for row in conditional_rates(truth,score>=thresholds[None,None,:],names):
                    conditional.append({'dataset':dataset,'model':model_name,'seed':seed,**row})
    write(out/'per_track.csv',per_track);write(out/'conditional_false_positives.csv',conditional)
    differences={}
    for dataset in ['natural_subset','controlled']:
        tracks=sorted({r['track'] for r in per_track if r['dataset']==dataset})
        delta=[]
        for track in tracks:
            values={model:mean(r['macro_balanced_accuracy'] for r in per_track if r['dataset']==dataset and r['model']==model and r['track']==track) for model in ['baseline','attention']}
            delta.append(values['attention']-values['baseline'])
        loo=[mean(delta[:i]+delta[i+1:]) for i in range(len(delta))]
        differences[dataset]={'tracks':tracks,'attention_minus_mean_track_BA':delta,'unweighted_mean_delta':mean(delta),'positive_tracks':sum(d>0 for d in delta),'leave_one_track_out_mean_delta_range':[min(loo),max(loo)],'scope':'four-track descriptive robustness, not a confidence interval or independent test'}
    (out/'track_robustness.json').write_text(json.dumps(differences,indent=2))
    matrix=np.full((4,4),np.nan)
    candidates=[]
    for c,name in enumerate(names):
        for d,other in enumerate(names):
            records=[r for r in conditional if r['dataset']=='controlled' and r['model']=='attention' and r['target']==name and r['other']==other]
            if not records or min(records[0]['other_active_support'],records[0]['other_absent_support'])<20:continue
            matrix[c,d]=mean(r['fpr_other_active']-r['fpr_other_absent'] for r in records)
            candidates.append({'target':name,'other':other,'support_other_active':records[0]['other_active_support'],'support_other_absent':records[0]['other_absent_support'],'fpr_active':mean(r['fpr_other_active'] for r in records),'fpr_absent':mean(r['fpr_other_absent'] for r in records),'difference':float(matrix[c,d])})
    (out/'supported_conditional_contrasts.json').write_text(json.dumps(candidates,indent=2))
    fig,ax=plt.subplots(figsize=(5.5,4));im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(4));ax.set_xticklabels(names);ax.set_yticks(range(4));ax.set_yticklabels(names)
    ax.set_xlabel('Other instrument');ax.set_ylabel('Absent target instrument')
    for c in range(4):
        for d in range(4):
            if np.isfinite(matrix[c,d]):ax.text(d,c,f'{matrix[c,d]:+.2f}',ha='center',va='center')
    ax.set_title('Conditional false-positive rate difference\nother active minus absent (descriptive)');fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(out/'conditional_fpr.png',dpi=180);plt.close(fig)
    print(json.dumps(differences,indent=2))


if __name__=='__main__':main()
