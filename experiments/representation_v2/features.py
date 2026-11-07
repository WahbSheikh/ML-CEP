"""Fixed eGeMAPSv02 features; no pretrained emotion model or corpus fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import opensmile
from src.audio_features import load_audio, extract_file


def extractor():
    return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                          feature_level=opensmile.FeatureLevel.Functionals,
                          num_workers=1)


def egemaps(path, smile=None):
    smile = smile or extractor()
    values = smile.process_signal(load_audio(path).astype(np.float32), 16000)
    array = values.to_numpy(dtype=np.float64).ravel()
    if array.shape != (88,) or not np.isfinite(array).all():
        raise ValueError(f"Invalid eGeMAPS features: {path}")
    return array


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_all(root, manifest):
    out = root / 'experiments/representation_v2'
    identity = {'manifest': sha(root / 'data/split_manifest.csv'),
                'source': sha(Path(__file__)), 'opensmile': opensmile.__version__,
                'audio_source': sha(root / 'src/audio_features.py')}
    cache = out / 'features.npz'
    metadata = out / 'features_metadata.json'
    if cache.exists():
        assert json.loads(metadata.read_text())['identity'] == identity, 'Stale feature cache'
        with np.load(cache) as data:
            assert np.array_equal(data['filenames'], manifest.filename.to_numpy())
            return data['egemaps']
    smile = extractor()
    rows = []
    partial = out / 'feature_progress.npz'
    if partial.exists():
        with np.load(partial) as data:
            assert json.loads(str(data['identity'])) == identity
            rows = list(data['egemaps'])
    for i in range(len(rows), len(manifest)):
        row = manifest.iloc[i]
        if sha(row.path) != row.sha256:
            raise ValueError(f'Audio changed: {row.filename}')
        rows.append(egemaps(row.path, smile))
        if (i+1) % 250 == 0:
            with partial.with_suffix('.partial').open('wb') as handle:
                np.savez_compressed(handle, identity=json.dumps(identity), egemaps=rows)
            partial.with_suffix('.partial').replace(partial)
            print(f'eGeMAPS {i+1}/{len(manifest)}', flush=True)
    values = np.asarray(rows)
    with cache.with_suffix('.partial').open('wb') as handle:
        np.savez_compressed(handle, egemaps=values, filenames=manifest.filename.to_numpy(dtype=str))
    cache.with_suffix('.partial').replace(cache)
    metadata.write_text(json.dumps({'identity': identity, 'names': smile.feature_names,
                                    'shape': list(values.shape)}, indent=2))
    return values


def predict_emotion(audio_path, project_root=None):
    """Predict using trusted local v2 artifacts and the exact saved feature recipe."""
    import joblib
    from .run import identity, load_data
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    out = root / 'experiments/representation_v2'
    frozen = json.loads((out / 'selection.json').read_text())
    assert identity(root) == frozen['identity'], 'Experiment inputs changed'
    record = joblib.load(out / 'winner.joblib')
    vector = egemaps(audio_path)
    if record['representation'] == 'fusion':
        vector = np.r_[extract_file(audio_path, filtered=True)[0], vector]
    return str(record['pipeline'].predict(vector[None, :])[0])
