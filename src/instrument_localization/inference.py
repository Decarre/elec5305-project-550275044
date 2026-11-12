"""Audio-to-timeline inference with a checkpoint-bound validation calibration."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .alignment import aggregate_time_grid
from .features import checkpoint_features, load_audio, PREPROCESSING_VERSION
from .training import load_checkpoint


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):digest.update(part)
    return digest.hexdigest()


def predict_audio(audio_path,checkpoint_path,calibration_path,output_dir):
    model,config,_=load_checkpoint(checkpoint_path)
    calibration=json.loads(Path(calibration_path).read_text())
    if calibration['checkpoint_sha256']!=sha256(checkpoint_path):
        raise ValueError('calibration belongs to a different checkpoint')
    if calibration['preprocessing_version']!=PREPROCESSING_VERSION:
        raise ValueError('unsupported preprocessing version')
    audio,sr=load_audio(Path(audio_path),config.sample_rate)
    if not len(audio):raise ValueError('audio is empty')
    chunk_size=int(sr*config.clip_duration)
    thresholds=np.asarray(calibration['frame_thresholds'])
    rows=[];probabilities=[]
    for start in range(0,len(audio),chunk_size):
        waveform=audio[start:start+chunk_size]
        valid_duration=len(waveform)/sr
        padded=np.pad(waveform,(0,chunk_size-len(waveform)))
        with torch.no_grad():
            prediction=model(torch.from_numpy(checkpoint_features(padded,config)[None]))
        scores=aggregate_time_grid(prediction['frame_probabilities'].numpy(),sr,config.hop_length,config.clip_duration,config.reporting_grid_seconds)[0]
        clip_scores=prediction['clip_probabilities'][0].numpy()
        probabilities.append({'start_seconds':start/sr,'valid_duration':valid_duration,'probabilities':dict(zip(config.target_instruments,map(float,clip_scores)))})
        for b,score in enumerate(scores):
            local=b*config.reporting_grid_seconds
            if local>=valid_duration:break
            row={'start_seconds':start/sr+local,'end_seconds':start/sr+min(local+config.reporting_grid_seconds,valid_duration)}
            row.update({name:float(value) for name,value in zip(config.target_instruments,score)})
            rows.append(row)
    intervals=[]
    for c,name in enumerate(config.target_instruments):
        onset=None
        for i in range(len(rows)+1):
            active=i<len(rows) and rows[i][name]>=thresholds[c]
            if active and onset is None:onset=i
            if not active and onset is not None:
                intervals.append({'instrument':name,'start_seconds':rows[onset]['start_seconds'],'end_seconds':rows[i-1]['end_seconds']})
                onset=None
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'timeline.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'audio':str(audio_path),'checkpoint_sha256':calibration['checkpoint_sha256'],'duration_seconds':len(audio)/sr,'classes':config.target_instruments,'frame_thresholds':thresholds.tolist(),'preprocessing_version':PREPROCESSING_VERSION,'postprocessing':'none; contiguous grid bins merged across clip edges','tail_handling':'last clip zero-padded, output clipped to audio duration','clip_predictions':probabilities,'intervals':intervals,'caution':'Research prototype, synthetic training; probability calibration and real-audio reliability are not established.'}
    (out/'prediction.json').write_text(json.dumps(result,indent=2))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(config.target_instruments),1,figsize=(10,7),sharex=True)
    times=[(r['start_seconds']+r['end_seconds'])/2 for r in rows]
    for c,(ax,name) in enumerate(zip(np.atleast_1d(axes),config.target_instruments)):
        ax.plot(times,[r[name] for r in rows],label='frame score')
        ax.axhline(thresholds[c],color='gray',ls='--',label='validation threshold')
        ax.set_ylabel(name);ax.set_ylim(-.05,1.05)
    np.atleast_1d(axes)[0].legend();np.atleast_1d(axes)[-1].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(out/'timeline.png',dpi=160);plt.close(fig)
    return result
