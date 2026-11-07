"""Synthetic fixtures test safeguards only; never counted as user voice samples."""
import numpy as np
import pytest
from .session import create_session, read_session, save_clip, evaluate_session, EMOTIONS


def fixture_audio():
    return (.1*np.sin(2*np.pi*200*np.arange(16000)/16000)).astype(np.float32)


def test_balanced_session_and_no_early_predictions(tmp_path):
    folder=create_session(tmp_path)
    data=read_session(folder)
    assert all(data['plan'].count(label)==5 for label in EMOTIONS)
    with pytest.raises(AssertionError,match='twenty'):
        evaluate_session(folder,tmp_path)
    assert not (folder/'evaluation.json').exists()


def test_silence_cannot_be_saved_as_a_valid_recording(tmp_path):
    folder=create_session(tmp_path)
    with pytest.raises(ValueError,match='silent'):
        save_clip(folder,np.zeros(16000),16000,'SYNTHETIC TEST FIXTURE')
    assert len(read_session(folder)['clips'])==0


def test_twenty_labelled_recordings_limit_and_data_tampering(tmp_path):
    folder=create_session(tmp_path)
    for _ in range(20):
        save_clip(folder,fixture_audio(),16000,'SYNTHETIC TEST FIXTURE')
    data=read_session(folder)
    assert [c['intended_emotion'] for c in data['clips']]==data['plan']
    with pytest.raises(ValueError,match='twenty'):
        save_clip(folder,fixture_audio(),16000,'SYNTHETIC TEST FIXTURE')
    path=folder/data['clips'][0]['filename']
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(AssertionError,match='audio changed'):
        evaluate_session(folder,tmp_path)


def test_frozen_dataset_disallows_replacement_after_predictions(tmp_path):
    folder=create_session(tmp_path)
    (folder/'frozen_recordings.json').write_text('{}')
    with pytest.raises(ValueError,match='frozen'):
        save_clip(folder,fixture_audio(),16000,'SYNTHETIC TEST FIXTURE')
