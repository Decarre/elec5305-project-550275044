"""Clip-local event scoring on an explicitly quantised reference grid."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def intervals(activity, step=.5):
    changes=np.diff(np.r_[False,np.asarray(activity,dtype=bool),False].astype(int))
    return np.stack([np.flatnonzero(changes==1),np.flatnonzero(changes==-1)],axis=1)*step


def event_metrics(reference,scores,thresholds,step=.5,tolerance=.5):
    truth=np.asarray(reference)>=.5
    predicted=np.asarray(scores)>=np.asarray(thresholds)[None,None,:]
    tp=fp=fn=0;onset=[];offset=[]
    for n in range(len(truth)):
        for c in range(truth.shape[2]):
            actual=intervals(truth[n,:,c],step);guess=intervals(predicted[n,:,c],step)
            matched=0
            if len(actual) and len(guess):
                error=np.abs(actual[:,None,:]-guess[None,:,:])
                valid=(error<=tolerance+1e-9).all(axis=2)
                # Invalid matches cost more than every possible valid match combined.
                cost=np.where(valid,error.sum(axis=2)/(2*tolerance+1e-9),len(actual)+len(guess)+1)
                a,b=linear_sum_assignment(cost)
                for left,right in zip(a,b):
                    if valid[left,right]:
                        matched+=1;onset.append(float(error[left,right,0]));offset.append(float(error[left,right,1]))
            tp+=matched;fp+=len(guess)-matched;fn+=len(actual)-matched
    return {'tp':tp,'fp':fp,'fn':fn,'event_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'matched_onset_mae_seconds':float(np.mean(onset)) if onset else None,'matched_offset_mae_seconds':float(np.mean(offset)) if offset else None,'grid_seconds':step,'endpoint_tolerance_seconds':tolerance,'scope':'clip-local, boundaries censored at clip edges; not continuous-track event scoring'}
