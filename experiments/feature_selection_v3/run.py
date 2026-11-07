"""Isolated, resumable feature-selection diagnostic. Never opens test targets."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import json
import hashlib
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from src.rbf_kernel import TrainingRBFKernel
from src.experiments import score_predictions, save_artifact
from experiments.representation_v2.run import ROOT, guard, load_data, identity


def specs():
    for k in [32,64,128,256,361]:
        for c in [.001,.01,.1]:
            yield 'Logistic Regression', k, {'C':c}, Pipeline([
                ('scale',StandardScaler()),('select',SelectKBest(f_classif,k=k)),
                ('classifier',LogisticRegression(C=c,max_iter=1500,random_state=42))])
        for c in [1,10]:
            for gamma in ['scale',.001]:
                yield 'SVM', k, {'C':c,'gamma':gamma}, Pipeline([
                    ('scale',StandardScaler()),('select',SelectKBest(f_classif,k=k)),
                    ('kernel',TrainingRBFKernel(gamma=gamma)),
                    ('classifier',SVC(C=c,kernel='precomputed',cache_size=96))])


def run(root=ROOT):
    out=root/'experiments/feature_selection_v3'
    guard(root)
    manifest,matrices=load_data(root)
    fingerprint=hashlib.sha256((identity(root)+Path(__file__).read_text()).encode()).hexdigest()
    if (out/'selection.json').exists():
        frozen=json.loads((out/'selection.json').read_text()); assert frozen['identity']==fingerprint
        return frozen
    train=manifest.split.eq('train').to_numpy(); valid=manifest.split.eq('validation').to_numpy()
    x=matrices['fusion'][train]; y=manifest.loc[train,'emotion'].to_numpy(); actors=manifest.loc[train,'actor'].to_numpy()
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(x,y,actors))
    for a,b in folds: assert not set(actors[a]) & set(actors[b])
    with threadpool_limits(1):
        corr=np.corrcoef(x,rowvar=False)
        pairs=np.argwhere(np.triu(abs(corr),1)>.95)
        metadata=json.loads((root/'experiments/representation_v2/features_metadata.json').read_text())
        names=json.loads((root/'artifacts/feature_schema.json').read_text())+['egemaps_'+n for n in metadata['names']]
        audit={'training_clips':int(train.sum()),'training_actors':len(set(actors)),
               'feature_count':x.shape[1],'feature_pairs_abs_correlation_above_0_95':len(pairs),
               'zero_variance_columns':int(np.sum(x.std(axis=0)<1e-12)),
               'warning':'Correlations are computed on training clips only; redundancy is not proof of causation.'}
        (out/'feature_audit.json').write_text(json.dumps(audit,indent=2))
        pd.DataFrame([{'first':names[a],'second':names[b],'correlation':corr[a,b]} for a,b in pairs]).to_csv(out/'correlated_pairs.csv',index=False)
        checkpoint=out/'checkpoint.joblib'
        state=joblib.load(checkpoint) if checkpoint.exists() else {'identity':fingerprint,'rows':[]}
        assert state['identity']==fingerprint
        definitions=list(specs())
        for order,(family,k,params,model) in enumerate(definitions):
            if order<len(state['rows']): continue
            started=time.perf_counter(); scores=[]; messages=[]
            for a,b in folds:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always'); fitted=clone(model).fit(x[a],y[a])
                messages.extend(str(w.message) for w in caught)
                scores.append(score_predictions(y[b],fitted.predict(x[b]))['macro_f1'])
            row={'candidate':order,'model':family,'k':k,'parameters':json.dumps(params),
                 'cv_macro_f1':float(np.mean(scores)),'cv_std':float(np.std(scores)),
                 'fold_scores':scores,'seconds':time.perf_counter()-started,'warnings':' | '.join(sorted(set(messages)))}
            state['rows'].append(row); save_artifact(state,checkpoint)
            pd.DataFrame(state['rows']).to_csv(out/'grouped_search.csv',index=False)
            print(f'{order+1}/35 {family} k={k}: CV F1={row["cv_macro_f1"]:.4f}',flush=True)
        search=pd.DataFrame(state['rows']); rows=[]; models={}
        predictions=manifest.loc[valid,['filename','actor','emotion']].copy()
        for family,group in search.groupby('model',sort=False):
            best=group.sort_values(['cv_macro_f1','candidate'],ascending=[False,True]).iloc[0]
            order=int(best.candidate); model=clone(definitions[order][3])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); model.fit(x,y)
            predicted=model.predict(matrices['fusion'][valid])
            row={'model':family,'candidate':order,'k':int(best.k),'parameters':best.parameters,'cv_macro_f1':float(best.cv_macro_f1),
                 **{'validation_'+key:value for key,value in score_predictions(manifest.loc[valid,'emotion'],predicted).items()},
                 **{'train_'+key:value for key,value in score_predictions(y,model.predict(x)).items()},
                 'warnings':' | '.join(str(w.message) for w in caught)}
            rows.append(row); models[family]=model; predictions[family]=predicted
            selected=np.asarray(names)[model.named_steps['select'].get_support()]
            pd.Series(selected,name='feature').to_csv(out/f'selected_{family.replace(" ","_")}.csv',index=False)
            assert model.named_steps['scale'].n_samples_seen_==train.sum()
            np.testing.assert_allclose(model.named_steps['scale'].mean_,x.mean(axis=0))
        table=pd.DataFrame(rows); chosen=table.sort_values(['validation_macro_f1','validation_accuracy','candidate'],ascending=[False,False,True]).iloc[0]
        old=json.loads((root/'experiments/representation_v2/selection.json').read_text())
        previous=next(r for r in old['rows'] if r['key']==old['winner'])
        frozen={'identity':fingerprint,'winner':chosen.model,'rows':rows,
                'validation_macro_f1_gain_vs_v2':float(chosen.validation_macro_f1-previous['validation_macro_f1']),
                'test_evaluated':False,'personal_voice_evaluated':False,
                'selection':'Training-only grouped CV per family, then reused validation macro-F1/accuracy/candidate'}
        save_artifact(models,out/'selected_models.joblib'); table.to_csv(out/'validation_results.csv',index=False)
        predictions.to_csv(out/'validation_predictions.csv',index=False)
        (out/'selection.json').write_text(json.dumps(frozen,indent=2))
    guard(root)
    return frozen


if __name__=='__main__':
    print(json.dumps(run(),indent=2),flush=True)
