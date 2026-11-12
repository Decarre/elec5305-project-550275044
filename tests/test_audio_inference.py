import json
import numpy as np
import pytest
import soundfile as sf
import torch
from instrument_localization.config import ExperimentConfig
from instrument_localization.features import checkpoint_features,log_mel_spectrogram,PREPROCESSING_VERSION
from instrument_localization.inference import predict_audio,sha256
from instrument_localization.training import build_model


def test_preprocessing_contract_matches_training_scale():
    config=ExperimentConfig(sample_rate=8000,n_fft=256,hop_length=80,n_mels=8)
    audio=np.sin(2*np.pi*220*np.arange(8000)/8000).astype(np.float32)
    db=log_mel_spectrogram(audio,8000,8,256,80)
    result=checkpoint_features(audio,config)
    assert np.array_equal(result,np.clip(db,-80,0).astype(np.float32)/80)
    assert result.min()>=-1 and result.max()<=0


def test_audio_cli_tail_and_checkpoint_bound_calibration(tmp_path):
    from instrument_localization.cli import main
    config=ExperimentConfig(sample_rate=8000,n_fft=256,hop_length=80,n_mels=8,
        hidden_size=8,clip_duration=1.,reporting_grid_seconds=.5,target_instruments=['drums','bass'])
    path=tmp_path/'model.pt'
    torch.save({'config':config.to_dict(),'model_state_dict':build_model(config).state_dict(),'validation_thresholds':[.5,.5]},path)
    calibration=tmp_path/'calibration.json'
    calibration.write_text(json.dumps({'checkpoint_sha256':sha256(path),'preprocessing_version':PREPROCESSING_VERSION,'frame_thresholds':[.5,.5]}))
    audio=tmp_path/'audio.wav';sf.write(str(audio),np.zeros(9600),8000)
    out=tmp_path/'output'
    assert main(['predict','--audio',str(audio),'--checkpoint',str(path),'--calibration',str(calibration),'--output-dir',str(out)])==0
    result=json.loads((out/'prediction.json').read_text())
    assert result['duration_seconds']==1.2
    assert len(result['clip_predictions'])==2
    assert all(v['end_seconds']<=1.2 for v in result['intervals'])
    assert (out/'timeline.png').exists()
    calibration.write_text(json.dumps({'checkpoint_sha256':'wrong','preprocessing_version':PREPROCESSING_VERSION}))
    with pytest.raises(ValueError,match='different checkpoint'):
        predict_audio(audio,path,calibration,out)
