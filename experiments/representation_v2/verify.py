"""Checks tied to real trained artifacts rather than implementation-shaped mocks."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .run import ROOT, guard, load_data, identity
from .features import predict_emotion, egemaps


def verify(root=ROOT):
    out = root/'experiments/representation_v2'
    checks = {}
    checks['baseline_files_preserved'] = guard(root)
    manifest, matrices = load_data(root)
    assert manifest.groupby('actor').split.nunique().max() == 1
    checks['fixed_actor_disjoint_split'] = manifest.groupby('split').actor.nunique().to_dict()
    folds = pd.read_csv(out/'grouped_folds.csv')
    training_actors = set(manifest.loc[manifest.split.eq('train'),'actor'])
    for _, fold in folds.groupby('fold'):
        fit = set(fold.loc[fold.part.eq('fit'),'actor'])
        held = set(fold.loc[fold.part.eq('holdout'),'actor'])
        assert not fit & held and fit | held == training_actors
    checks['three_grouped_folds_disjoint'] = True
    frozen = json.loads((out/'selection.json').read_text())
    assert frozen['identity'] == identity(root)
    table = pd.read_csv(out/'validation_results.csv')
    expected = table.sort_values(['validation_macro_f1','validation_accuracy','key'],ascending=[False,False,True]).iloc[0]['key']
    assert expected == frozen['winner']
    assert len(pd.read_csv(out/'grouped_search.csv')) == 40 and len(table) == 10
    checks['validation_only_winner_and_complete_search'] = True
    models = joblib.load(out/'selected_models.joblib')
    training = manifest.split.eq('train').to_numpy()
    for record in models.values():
        scaler = record['pipeline'].named_steps['scale']
        assert scaler.n_samples_seen_ == training.sum()
        np.testing.assert_allclose(scaler.mean_, matrices[record['representation']][training].mean(axis=0))
        if 'kernel' in record['pipeline'].named_steps:
            assert record['pipeline'].named_steps['kernel'].reference_features_.shape[0] == training.sum()
    checks['all_ten_models_training_only_scaling_and_kernel'] = True
    index = int(np.flatnonzero(manifest.split.eq('validation').to_numpy())[0])
    path = manifest.iloc[index].path
    np.testing.assert_allclose(egemaps(path), matrices['egemaps'][index], atol=1e-7, rtol=1e-7)
    predictions = pd.read_csv(out/'validation_predictions.csv')
    with threadpool_limits(1):
        assert predict_emotion(path, root) == predictions.iloc[0][frozen['winner']]
    checks['raw_wav_cache_and_saved_inference_agree'] = True
    tests = pd.read_csv(out/'test_results.csv')
    assert set(tests.model) == {'SVM','MLP','KNN','Logistic Regression','GaussianNB'}
    assert frozen['winner'] in set(tests.key)
    checks['five_families_evaluated_after_freeze'] = True
    checks['baseline_files_preserved_after_checks'] = guard(root)
    (out/'verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))
    return checks


if __name__ == '__main__':
    verify()
