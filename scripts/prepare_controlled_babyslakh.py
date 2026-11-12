"""Build a separate, deterministic stem-remixing benchmark on the original split."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from instrument_localization.slakh import discover_tracks, normalise_inst_class, file_sha256
from prepare_babyslakh import note_intervals, reference_grid
from instrument_localization.features import checkpoint_features
from instrument_localization.config import ExperimentConfig


def control_gates(variant, rng, bins=10):
    gates = np.ones((bins, 4), dtype=np.float32)
    if variant == 'absence':
        # Every clip removes two families, independently of its labels/model output.
        gates[:, rng.choice(4, 2, replace=False)] = 0
    elif variant == 'transition':
        for c in range(4):
            boundary = int(rng.randint(3, 8))
            if rng.randint(2):
                gates[:boundary, c] = 0
            else:
                gates[boundary:, c] = 0
    return gates


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--original-prepared',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    original=json.loads((args.original_prepared/'summary.json').read_text())
    partitions={t:p for p,tracks in original['split_tracks'].items() for t in tracks}
    buffers={p:{'features':[],'labels':[],'references':[],'track_ids':[],'start_seconds':[],'variants':[]} for p in original['split_tracks']}
    manifest=[]
    original_groups={}
    with (args.original_prepared/'clip_manifest.csv').open(newline='') as f:
        for row in csv.DictReader(f): original_groups[row['track_id']]=row['duplicate_group_sha256']
    for track in discover_tracks(args.dataset_root):
        assert file_sha256(track/'all_src.mid')==original_groups[track.name]
        partition=partitions[track.name]
        metadata=yaml.safe_load((track/'metadata.yaml').read_text())
        mix,sr=sf.read(str(track/'mix.wav'),dtype='float32')
        assert sr==16000 and mix.ndim==1
        sources=np.zeros((len(mix),5),dtype=np.float32)
        for stem_id,stem in metadata['stems'].items():
            path=track/'stems'/(stem_id+'.wav')
            if not path.exists(): continue
            waveform,stem_sr=sf.read(str(path),dtype='float32')
            assert stem_sr==sr and waveform.ndim==1
            family=normalise_inst_class(stem.get('inst_class',''))
            index=['drums','bass','guitar','piano'].index(family) if family else 4
            n=min(len(sources),len(waveform))
            sources[:n,index]+=waveform[:n]
        intervals=note_intervals(track,metadata)
        for clip_index in range(len(mix)//80000):
            start=clip_index*5.0
            reference=reference_grid(intervals,start,5,.5)
            chunk=sources[clip_index*80000:(clip_index+1)*80000]
            for variant in ('original','absence','transition'):
                seed=int(hashlib.sha256((track.name+str(clip_index)+variant+'protocol-v2').encode()).hexdigest()[:8],16)
                gates=control_gates(variant,np.random.RandomState(seed))
                if variant=='original':
                    audio=mix[clip_index*80000:(clip_index+1)*80000]
                else:
                    envelope=np.repeat(gates,8000,axis=0)
                    # Five-ms linear ramps reduce discontinuities, without using labels.
                    for boundary in range(1,10):
                        at=boundary*8000
                        for c in range(4):
                            if gates[boundary,c]!=gates[boundary-1,c]:
                                envelope[at-40:at+40,c]=np.linspace(gates[boundary-1,c],gates[boundary,c],80)
                    audio=(chunk[:,:4]*envelope).sum(axis=1)+chunk[:,4]
                truth=reference*gates
                features=checkpoint_features(audio,ExperimentConfig(sample_rate=sr,n_fft=1024,hop_length=320))
                b=buffers[partition]
                for key,value in [('features',features),('labels',truth.any(axis=0).astype(np.float32)),('references',truth),('track_ids',track.name),('start_seconds',start),('variants',variant)]: b[key].append(value)
                manifest.append({'track_id':track.name,'split':partition,'clip_index':clip_index,'variant':variant,'seed':seed,'midi_sha256':original_groups[track.name]})
        print(track.name,partition,flush=True)
    for partition,b in buffers.items():
        np.savez_compressed(str(args.output_dir/(partition+'.npz')),features=np.stack(b['features']),labels=np.stack(b['labels']))
        np.savez_compressed(str(args.output_dir/(partition+'_temporal.npz')),**{k:np.asarray(v) for k,v in b.items() if k not in ('features','labels')})
    with (args.output_dir/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    summary={'protocol':'v2: original, two-family absence, gated transitions; fixed SHA256-seeded interventions', 'split_tracks':original['split_tracks'], 'counts':{p:len(b['labels']) for p,b in buffers.items()},'note':'Controlled artificial interventions; not natural-recording generalisation. Gates are independent of MIDI labels and model scores.'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__': main()
