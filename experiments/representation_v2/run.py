"""Training-only grouped search, validation selection, then one cached test reuse."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from src.experiments import load_inputs, score_predictions, save_artifact
from src.rbf_kernel import TrainingRBFKernel
from .features import extract_all, sha

ROOT = Path(__file__).resolve().parents[2]


def guard(root):
    out = root / 'experiments/representation_v2'
    path = out / 'baseline_hashes.json'
    if not path.exists():
        files = [p for directory in ['src', 'artifacts', 'reports', 'docs', 'notebooks']
                 for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
        files += [root/'README.md', root/'requirements.txt', root/'requirements-lock.txt', root/'data/split_manifest.csv']
        path.write_text(json.dumps({str(p.relative_to(root)): sha(p) for p in files}, indent=2))
    frozen = json.loads(path.read_text())
    assert all((root/p).exists() and sha(root/p) == digest for p, digest in frozen.items()), 'Baseline changed'
    return len(frozen)


def load_data(root):
    manifest, baseline = load_inputs(root)
    extra = extract_all(root, manifest)
    assert extra.shape == (len(manifest), 88) and np.isfinite(extra).all()
    return manifest, {'egemaps': extra, 'fusion': np.c_[baseline['filtered'], extra]}


def identity(root):
    digest = hashlib.sha256()
    for name in ['data/split_manifest.csv', 'artifacts/features.npz', 'src/audio_features.py',
                 'src/experiments.py', 'src/rbf_kernel.py',
                 'experiments/representation_v2/features.py', 'experiments/representation_v2/run.py',
                 'experiments/representation_v2/requirements-lock.txt',
                 'experiments/representation_v2/features.npz', 'experiments/representation_v2/features_metadata.json']:
        digest.update(sha(root/name).encode())
    return digest.hexdigest()


def candidates():
    definitions = []
    for k in [9, 21]:
        definitions.append(('KNN', True, KNeighborsClassifier(n_neighbors=k, n_jobs=1)))
    for lda in [False, True]:
        for c in [0.01, 0.1, 1]:
            definitions.append(('Logistic Regression', lda, LogisticRegression(C=c, max_iter=1500, random_state=42)))
    for smoothing in [1e-9, 1e-5]:
        definitions.append(('GaussianNB', True, GaussianNB(var_smoothing=smoothing)))
    for c in [1, 10, 100]:
        for gamma in ['scale', 0.001]:
            definitions.append(('SVM', False, SVC(C=c, gamma=gamma)))
    for lda in [False, True]:
        for alpha in [0.01, 1]:
            definitions.append(('MLP', lda, MLPClassifier(hidden_layer_sizes=(64,), alpha=alpha,
                                learning_rate_init=0.001, max_iter=300, early_stopping=False, random_state=42)))
    for family, lda, classifier in definitions:
        steps = [('scale', StandardScaler())]
        if lda:
            steps.append(('lda', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')))
        if family == 'SVM':
            steps.append(('kernel', TrainingRBFKernel(gamma=classifier.gamma)))
            classifier = SVC(C=classifier.C, kernel='precomputed', cache_size=128)
        steps.append(('classifier', classifier))
        yield family, lda, Pipeline(steps)


def select(root=ROOT):
    out = root / 'experiments/representation_v2'
    guard(root)
    manifest, matrices = load_data(root)
    fingerprint = identity(root)
    if (out/'selection.json').exists():
        frozen = json.loads((out/'selection.json').read_text())
        assert frozen['identity'] == fingerprint
        return frozen
    training = manifest.split.eq('train').to_numpy()
    validation = manifest.split.eq('validation').to_numpy()
    y = manifest.emotion.to_numpy()
    groups = manifest.loc[training, 'actor'].to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(np.zeros(training.sum()), y[training], groups))
    fold_rows = []
    for fold, (a, b) in enumerate(folds):
        assert not set(groups[a]) & set(groups[b])
        for part, indices in [('fit', a), ('holdout', b)]:
            for actor in sorted(set(groups[indices])):
                fold_rows.append({'fold': fold, 'part': part, 'actor': int(actor)})
    pd.DataFrame(fold_rows).to_csv(out/'grouped_folds.csv', index=False)
    checkpoint = out/'cv_checkpoint.joblib'
    state = joblib.load(checkpoint) if checkpoint.exists() else {'identity': fingerprint, 'rows': []}
    assert state['identity'] == fingerprint
    specs = list(candidates())
    with threadpool_limits(1):
        for rep, matrix in matrices.items():
            x = matrix[training]
            for order, (family, lda, model) in enumerate(specs):
                key = f'{rep}-{order}'
                if any(r['key'] == key for r in state['rows']):
                    continue
                started = time.perf_counter()
                scores, messages = [], []
                for a, b in folds:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        fitted = clone(model).fit(x[a], y[training][a])
                    messages.extend(str(w.message) for w in caught)
                    scores.append(score_predictions(y[training][b], fitted.predict(x[b]))['macro_f1'])
                row = dict(key=key, representation=rep, candidate=order, model=family, lda=lda,
                           cv_macro_f1=float(np.mean(scores)), cv_std=float(np.std(scores)),
                           fold_scores=scores, seconds=time.perf_counter()-started,
                           warnings=' | '.join(sorted(set(messages))), specification=str(model))
                state['rows'].append(row)
                save_artifact(state, checkpoint)
                pd.DataFrame(state['rows']).to_csv(out/'grouped_search.csv', index=False)
                print(f'{len(state["rows"])}/40 {key} {family}: grouped F1={row["cv_macro_f1"]:.4f}', flush=True)
        rows, models = [], {}
        search = pd.DataFrame(state['rows'])
        for (rep, family), frame in search.groupby(['representation', 'model'], sort=False):
            best = frame.sort_values(['cv_macro_f1', 'candidate'], ascending=[False, True]).iloc[0]
            model = clone(specs[int(best.candidate)][2])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(matrices[rep][training], y[training])
            predicted = model.predict(matrices[rep][validation])
            row = {k: best[k] for k in ['key','representation','model','candidate','cv_macro_f1','cv_std']}
            row['candidate'] = int(row['candidate'])
            row.update({'validation_'+k: v for k, v in score_predictions(y[validation], predicted).items()})
            row.update({'train_'+k: v for k, v in score_predictions(y[training], model.predict(matrices[rep][training])).items()})
            row['warnings'] = ' | '.join(str(w.message) for w in caught)
            rows.append(row)
            models[row['key']] = {'pipeline': model, 'representation': rep}
        validation_table = pd.DataFrame(rows)
        ranked = validation_table.sort_values(['validation_macro_f1','validation_accuracy','key'], ascending=[False,False,True])
        winner = ranked.iloc[0]['key']
        frozen = {'identity': fingerprint, 'winner': winner, 'selection': 'Training-only 3-fold actor-grouped macro-F1 per family/representation; validation macro-F1, accuracy, lexical key',
                  'candidates': 40, 'validation_finalists': 10, 'rows': rows,
                  'test_status': 'Previously seen baseline test will be reused once after this freeze; not a new untouched test.'}
        validation_table.to_csv(out/'validation_results.csv', index=False)
        save_artifact(models, out/'selected_models.joblib')
        save_artifact(models[winner], out/'winner.joblib')
        (out/'selection.json').write_text(json.dumps(frozen, indent=2))
        predictions = manifest.loc[validation, ['filename','actor','emotion']].copy()
        for key, record in models.items():
            predictions[key] = record['pipeline'].predict(matrices[record['representation']][validation])
        predictions.to_csv(out/'validation_predictions.csv', index=False)
    guard(root)
    return frozen


def evaluate(root=ROOT):
    out = root/'experiments/representation_v2'
    frozen = json.loads((out/'selection.json').read_text())
    assert frozen['identity'] == identity(root)
    if (out/'test_results.csv').exists():
        return pd.read_csv(out/'test_results.csv')
    manifest, matrices = load_data(root)
    models = joblib.load(out/'selected_models.joblib')
    test = manifest.split.eq('test').to_numpy()
    predictions = manifest.loc[test, ['filename','actor','emotion']].copy()
    rows = []
    # Evaluate only the validation-chosen representative of each required family.
    table = pd.DataFrame(frozen['rows']).sort_values(['validation_macro_f1','validation_accuracy','key'], ascending=[False,False,True])
    with threadpool_limits(1):
        for row in table.drop_duplicates('model').itertuples():
            record = models[row.key]
            predicted = record['pipeline'].predict(matrices[record['representation']][test])
            predictions[row.key] = predicted
            rows.append(dict(key=row.key, model=row.model, **score_predictions(predictions.emotion, predicted)))
    predictions.to_csv(out/'test_predictions.csv', index=False)
    pd.DataFrame(rows).to_csv(out/'test_results.csv', index=False)
    guard(root)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    print(json.dumps(select(), indent=2), flush=True)
    if args.test:
        print(evaluate().to_string(index=False))
