"""Document positive/negative feature-selection evidence without opening test."""
import json
from pathlib import Path
import nbformat
import pandas as pd
from .run import ROOT, run
from experiments.representation_v2.deliver import markdown_table


def build(root=ROOT):
    out=root/'experiments/feature_selection_v3'
    frozen=run(root)
    audit=json.loads((out/'feature_audit.json').read_text())
    table=pd.read_csv(out/'validation_results.csv')
    search=pd.read_csv(out/'grouped_search.csv')
    grouped=search.groupby(['model','k'],as_index=False).cv_macro_f1.max()
    report=f'''# Optional feature-selection/regularization diagnostic (v3)

This is supplemental investigation requested after the personal-voice failure; it does not replace the core CEP or change its required five-model evaluation.

Hypothesis: redundant/noisy feature columns and insufficient regularization might contribute to poor unseen-speaker performance. Training-only audit found **{audit['feature_pairs_abs_correlation_above_0_95']} pairs with absolute correlation >0.95**, and zero constant columns. Correlation alone does not establish the cause of the generalization gap.

ANOVA SelectKBest inside each training fold compared 32,64,128,256 and 361 features. Thirty-five configurations (Logistic Regression and SVM), each on three actor-grouped folds, made 105 internal fits. Scaling, ANOVA scores and classifiers only saw fit-fold actors. C values included stronger regularization. Hyperparameters/feature count were selected from grouped training CV; just two finalists then used the reused validation set. No new test or personal-voice evaluation was performed.

## Measured outcome

{markdown_table(table[['model','k','parameters','cv_macro_f1','train_accuracy','validation_accuracy','validation_macro_f1']])}

Both families selected **all 361 features**. The selected SVM reproduced v2 exactly: validation accuracy 51.40%, macro-F1 0.5090603; gain versus v2 **{frozen['validation_macro_f1_gain_vs_v2']:.4f}**. This experiment did **not** yield another improvement. V2 remains the extension winner; the baseline stays preserved.

Best internal CV score among predeclared hyperparameters at each k (selected estimates, not nested evaluation):

{markdown_table(grouped)}

## What this says about overfitting and features

The 83.03% training / 51.40% validation gap is consistent with overfitting and actor/domain differences. It does not identify one cause. Having 5,149 clips does not mean 5,149 independent speakers: training contains only 63 actors. Fixed clip summaries can encode speaker/microphone properties while discarding useful frame order. The personal microphone test adds a further recording-domain shift. Its 25% result is a real failure; feature redundancy alone has not explained it.

The tested univariate feature-selection strategy worsened internal grouped scores when it removed many columns. This suggests useful information is distributed across features or their interactions under these models. It does not prove every column is necessary or every feature-selection method would fail. ANOVA ranks features independently; it does not optimize joint subsets. More sophisticated selection and temporal/pretrained representations remain hypotheses requiring fresh, properly controlled evaluation. No promise of 80-90% accuracy or guaranteed personal-voice improvement is made.

Original assignment evidence, v2 gains/uncertainty and failed personal results all remain valid and separate. Full search, selected column names, warnings, training-only scaling/ANOVA checks and validation predictions are saved here. [Protocol](PROTOCOL.md) fixes the tested design. No commit or push occurred.
'''
    (out/'report.md').write_text(report,encoding='utf-8')
    kt='''## Extra investigation: kya features ghalat thay? (v3)

Is sawal ko sirf guess se answer nahi kiya. Training data mein 794 highly correlated feature pairs mili. Phir har training fold ke andar ANOVA feature selection lagayi: 32,64,128,256 ya poore361 features. 35 configurations aur3actor-grouped folds se105 internal fits hue. Logistic Regression aur SVM dono ke liye stronger regularization bhi test hui. Core CEP ke five required classifiers pehle se maujood hain; yeh two-family test alag diagnostic hai.

**Actual result: dono families ne poore361 features select kiye.** Reduced feature subsets ke grouped CV scores kam thay. Selected SVM phir wohi v2 configuration bani: validation51.40%, macro-F10.5091. Naya gain0. Is liye “features kam kar diye aur accuracy improve ho gayi” likhna ghalat hoga. Test aur aapki20 voice clips ko is experiment mein dobara evaluate nahi kiya.

Correlation ka matlab milti-julti information hai; yeh automatically bekaar feature ya overfitting ki pakki wajah nahi. Univariate ANOVA har feature ko alag rank karti hai; useful interactions miss ho sakti hain. Is test se sirf itna pata chala ke tested selection strategy se gain nahi mila. Har possible selection method fail hoga, yeh prove nahi hua.

Overfitting ko training/validation gap se pehchante hain, magar gap ki exact wajah ke liye controlled tests chahiye. 5149training clips sirf63actors se aati hain; actor-related patterns independent new speakers par fail ho sakti hain. Personal microphone recording ka domain bhi dataset se different hai. Voice par25% result ko sirf “features ghalat” ya “aapne emotion sahi nahi bola” keh kar explain nahi kar sakte. Evidence itna strong nahi.

Agla scientific direction stronger temporal/pretrained speech representation ya carefully designed domain-robust training ho sakti hai. Woh abhi implement/train nahi hui; RAM bhi limited hai. Maujooda20recordings par tune karenge to unhein development data kehna hoga aur fresh recordings se confirmation chahiye. Core CEP ka kaam aur optional experiments ki limitations report mein alag hain.
'''
    (out/'KT_ADDENDUM.md').write_text(kt,encoding='utf-8')
    code=nbformat.v4.new_code_cell; md=nbformat.v4.new_markdown_cell
    notebook=nbformat.v4.new_notebook(cells=[md('# Optional v3: feature selection and regularization\nTraining-only grouped selection. No test or personal-voice tuning. This is extra user-requested work, supplementary to core CEP.'),code('''from pathlib import Path
import sys,json
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/audio_features.py').exists())
sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display,Code
from experiments.feature_selection_v3.run import run,specs
OUT=ROOT/'experiments/feature_selection_v3'
frozen=run(ROOT)
display(json.loads((OUT/'feature_audit.json').read_text()))
display(pd.read_csv(OUT/'validation_results.csv'))
import inspect
display(Code(inspect.getsource(specs),language='python'))'''),code('''search=pd.read_csv(OUT/'grouped_search.csv')
best=search.groupby(['model','k'],as_index=False).cv_macro_f1.max()
fig,ax=plt.subplots(figsize=(8,5))
for name,frame in best.groupby('model'):
    ax.plot(frame.k,frame.cv_macro_f1,'o-',label=name)
ax.set(xlabel='Selected features (ANOVA inside each fold)',ylabel='Mean training grouped CV macro-F1',title='Best predeclared configuration per feature count')
ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
fig.savefig(OUT/'feature_selection_curve.png',dpi=140); plt.show()
display(best)
print('Gain versus v2 validation F1:',frozen['validation_macro_f1_gain_vs_v2'])'''),code('''import joblib
from sklearn.feature_selection import f_classif
from experiments.representation_v2.run import load_data,guard
manifest,matrices=load_data(ROOT)
train=manifest.split.eq('train').to_numpy()
models=joblib.load(OUT/'selected_models.joblib')
for model in models.values():
    scale=model.named_steps['scale']
    assert scale.n_samples_seen_==train.sum()
    np.testing.assert_allclose(scale.mean_,matrices['fusion'][train].mean(axis=0))
    scores,_=f_classif(scale.transform(matrices['fusion'][train]),manifest.loc[train,'emotion'])
    np.testing.assert_allclose(scores,model.named_steps['select'].scores_,rtol=1e-7,atol=1e-7)
assert frozen['test_evaluated'] is False and frozen['personal_voice_evaluated'] is False
print('Verified training-only scaler/ANOVA fitting; no test/personal evaluation; baseline files:',guard(ROOT))'''),md(kt)],metadata={'kernelspec':{'display_name':'ML-CEP (.venv Python 3.12)','language':'python','name':'ml-cep'}})
    path=out/'feature_selection.ipynb'; nbformat.write(notebook,path)
    print(path)
    return path


if __name__=='__main__':
    build()
