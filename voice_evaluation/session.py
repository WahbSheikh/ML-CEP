"""Twenty prelabelled recordings, frozen before any predictions are exposed."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = ['happy', 'disgust', 'sad', 'anger']
PROMPT = 'The airplane is almost full.'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_session(root=ROOT):
    folder = root/'voice_evaluation/sessions'/f'{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
    folder.mkdir(parents=True)
    # Five blocks, each with every requested emotion, limits order/fatigue imbalance.
    rng = np.random.default_rng(42)
    plan = [str(label) for _ in range(5) for label in rng.permutation(EMOTIONS)]
    data = {'schema': 1, 'created_local': datetime.now().isoformat(),
            'speaker': 'user_personal_voice', 'prompt': PROMPT,
            'labels': 'Self-reported intended emotions, not independent listener ratings',
            'plan': plan, 'clips': [], 'models_frozen_before_recording': {},
            'role': 'Personal external evaluation only; no training, calibration or tuning'}
    for name in ['artifacts/frozen_selection.json','experiments/representation_v2/selection.json']:
        if (root/name).exists():
            data['models_frozen_before_recording'][name] = digest(root/name)
    (folder/'session.json').write_text(json.dumps(data,indent=2))
    return folder


def read_session(folder):
    folder = Path(folder)
    data = json.loads((folder/'session.json').read_text())
    assert len(data['plan']) == 20
    assert all(data['plan'].count(label) == 5 for label in EMOTIONS)
    return data


def save_clip(folder, samples, rate, device):
    import soundfile as sf
    folder = Path(folder)
    data = read_session(folder)
    if (folder/'evaluation.json').exists() or (folder/'frozen_recordings.json').exists():
        raise ValueError('Recordings already frozen for evaluation')
    i = len(data['clips'])
    if i >= 20:
        raise ValueError('All twenty clips are already saved')
    samples = np.asarray(samples,dtype=np.float32).reshape(-1)
    if not np.isfinite(samples).all() or len(samples)/rate < 1 or np.max(abs(samples)) < 1e-4:
        raise ValueError('Audio silent, too short or invalid; microphone check karke dobara record karein')
    path = folder/f'{i+1:02d}_{data["plan"][i]}.wav'
    sf.write(path,samples,rate,subtype='PCM_16')
    data['clips'].append({'filename':path.name, 'intended_emotion':data['plan'][i],
                          'duration':len(samples)/rate,'sample_rate':int(rate),
                          'sha256':digest(path),'device':device,
                          'peak':float(abs(samples).max()),
                          'clipped_sample_fraction':float(np.mean(abs(samples)>=0.999)),
                          'saved_local':datetime.now().isoformat()})
    temp = folder/'session.partial'
    temp.write_text(json.dumps(data,indent=2))
    temp.replace(folder/'session.json')
    return data


def evaluate_session(folder, root=ROOT):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    from threadpoolctl import threadpool_limits
    from src.audio_features import extract_file
    from src.experiments import LABELS
    from experiments.representation_v2.features import egemaps, extractor
    from experiments.representation_v2.run import identity, guard
    folder = Path(folder)
    data = read_session(folder)
    assert len(data['clips']) == 20, 'Save all twenty real recordings first'
    for clip in data['clips']:
        assert digest(folder/clip['filename']) == clip['sha256'], 'Saved audio changed'
    if (folder/'evaluation.json').exists():
        return json.loads((folder/'evaluation.json').read_text())
    out = root/'experiments/representation_v2'
    frozen = json.loads((out/'selection.json').read_text())
    assert frozen['identity'] == identity(root), 'Model inputs changed'
    for name, expected in data['models_frozen_before_recording'].items():
        assert digest(root/name) == expected, 'Model selection changed during recording'
    guard(root)
    snapshot = {'session_sha256':digest(folder/'session.json'),
                'v2_identity':frozen['identity'], 'v2_winner':frozen['winner'],
                'baseline_pipeline_sha256':digest(root/'artifacts/winning_pipeline.joblib'),
                'v2_pipeline_sha256':digest(out/'winner.joblib')}
    if (folder/'frozen_recordings.json').exists():
        assert json.loads((folder/'frozen_recordings.json').read_text()) == snapshot
    else:
        (folder/'frozen_recordings.json').write_text(json.dumps(snapshot,indent=2))
    baseline = joblib.load(root/'artifacts/winning_pipeline.joblib')
    improved = joblib.load(out/'winner.joblib')
    schema = json.loads((root/'artifacts/feature_schema.json').read_text())
    smile = extractor()
    rows = []
    with threadpool_limits(1):
        for clip in data['clips']:
            path = folder/clip['filename']
            features, names = extract_file(path,filtered=True)
            assert names == schema
            extra = egemaps(path,smile)
            vector = np.r_[features,extra] if improved['representation']=='fusion' else extra
            rows.append({**clip,'baseline_prediction':str(baseline.predict(features[None,:])[0]),
                         'improved_prediction':str(improved['pipeline'].predict(vector[None,:])[0])})
    table = pd.DataFrame(rows)
    result = {'recordings':20,'speaker_count':1,'requested_emotions':EMOTIONS,
              'support_per_emotion':table.intended_emotion.value_counts().to_dict(),
              'v2_winner':frozen['winner'],'models':{},
              'limits':'One speaker, one short session, five clips per intended emotion; descriptive personal performance only. Four supported emotions; classifier still predicts six. Fear/neutral predictions count as errors. No tuning on these recordings.'}
    for model, column in [('baseline','baseline_prediction'),('improved','improved_prediction')]:
        result['models'][model] = {
            'correct':int((table[column]==table.intended_emotion).sum()),
            'accuracy':float(accuracy_score(table.intended_emotion,table[column])),
            'macro_f1_four_supported':float(f1_score(table.intended_emotion,table[column],labels=EMOTIONS,average='macro',zero_division=0)),
            'macro_f1_six_fixed':float(f1_score(table.intended_emotion,table[column],labels=LABELS,average='macro',zero_division=0)),
            'classification_report':classification_report(table.intended_emotion,table[column],labels=LABELS,output_dict=True,zero_division=0),
            'confusion_matrix_six':confusion_matrix(table.intended_emotion,table[column],labels=LABELS).tolist(),
            'confusion_label_order':LABELS}
    table.to_csv(folder/'predictions.csv',index=False)
    (folder/'evaluation.json').write_text(json.dumps(result,indent=2))
    report = '# Personal voice evaluation\n\nReal user microphone recordings: 20, five each happy/disgust/sad/anger. Both models were frozen before prediction; no training uses these clips.\n\n'
    for model, values in result['models'].items():
        report += f"- {model}: {values['correct']}/20 correct, accuracy {values['accuracy']:.1%}, four-supported-class macro-F1 {values['macro_f1_four_supported']:.4f}; fixed-six-class macro-F1 {values['macro_f1_six_fixed']:.4f}.\n"
    report += '\n'+result['limits']+'\n\nIntended labels are self-reported; they are not independently rated perceived emotion. Repeated takes are correlated. This personal test is separate from the reused CREMA-D test and cannot establish generalization across speakers. Saved WAVs and results are local and ignored by Git.\n'
    (folder/'report.md').write_text(report,encoding='utf-8')
    return result
