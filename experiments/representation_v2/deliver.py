"""Build measured comparison, separate Roman Urdu KT addendum and executed notebook."""
from pathlib import Path
import os
import sys
import json
import html
import base64
import numpy as np
import pandas as pd
import nbformat as nbf
from sklearn.metrics import f1_score, accuracy_score, classification_report
from .run import ROOT, guard, select, evaluate


def paired_interval(frame, old, new, repeats=1000):
    rng = np.random.default_rng(42)
    actors = frame.actor.unique()
    positions = {a: np.flatnonzero(frame.actor.to_numpy() == a) for a in actors}
    truth = frame.emotion.to_numpy()
    baseline, improved = frame[old].to_numpy(), frame[new].to_numpy()
    differences = []
    for _ in range(repeats):
        indices = np.concatenate([positions[a] for a in rng.choice(actors, len(actors), replace=True)])
        a, b, c = truth[indices], baseline[indices], improved[indices]
        differences.append([accuracy_score(a,c)-accuracy_score(a,b),
                            f1_score(a,c,average='macro',labels=sorted(set(truth)),zero_division=0)-f1_score(a,b,average='macro',labels=sorted(set(truth)),zero_division=0)])
    return {'resamples': repeats, 'unit': 'paired actor',
            'accuracy_difference_95': np.percentile(differences,[2.5,97.5],axis=0)[:,0].tolist(),
            'macro_f1_difference_95': np.percentile(differences,[2.5,97.5],axis=0)[:,1].tolist()}


def markdown_table(frame):
    copy = frame.copy()
    for col in copy.select_dtypes('number'):
        copy[col] = copy[col].map(lambda x: f'{x:.4f}')
    return '| '+' | '.join(copy.columns)+' |\n| '+' | '.join(['---']*len(copy.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in copy.itertuples(index=False,name=None))


def statistics(root):
    out = root/'experiments/representation_v2'
    frozen = json.loads((out/'selection.json').read_text())
    winner = frozen['winner']
    base = json.loads((root/'artifacts/frozen_selection.json').read_text())
    baseline_valid = base['models'][base['winner']]
    new_valid = pd.read_csv(out/'validation_results.csv').set_index('key').loc[winner]
    baseline_test = pd.read_csv(root/'reports/test_results.csv').set_index('model').loc[base['winner']]
    new_test = pd.read_csv(out/'test_results.csv').set_index('key').loc[winner]
    result = {'winner': winner, 'baseline_winner': base['winner'],
              'validation_macro_f1_gain': float(new_valid.validation_macro_f1-baseline_valid['validation_macro_f1']),
              'validation_accuracy_gain': float(new_valid.validation_accuracy-baseline_valid['validation_accuracy']),
              'reused_test_macro_f1_gain': float(new_test.macro_f1-baseline_test.macro_f1),
              'reused_test_accuracy_gain': float(new_test.accuracy-baseline_test.accuracy)}
    for split, baseline_path, old_column in [('validation','reports/validation_predictions.csv','prediction'),('test','reports/test_predictions.csv',base['winner'])]:
        old = pd.read_csv(root/baseline_path)[['filename',old_column]].rename(columns={old_column:'baseline_prediction'})
        current = pd.read_csv(out/f'{split}_predictions.csv').merge(old,on='filename',validate='one_to_one')
        assert len(current) == (1146 if split=='validation' else 1140)
        result[split+'_paired_interval'] = paired_interval(current,'baseline_prediction',winner)
        if split == 'test':
            (out/'winner_classification.json').write_text(json.dumps(classification_report(current.emotion,current[winner],output_dict=True,zero_division=0),indent=2))
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    return result


def write_docs(root, stats):
    out = root/'experiments/representation_v2'
    valid = pd.read_csv(out/'validation_results.csv')
    test = pd.read_csv(out/'test_results.csv')
    win = valid.set_index('key').loc[stats['winner']]
    score = test.set_index('key').loc[stats['winner']]
    base = json.loads((root/'artifacts/frozen_selection.json').read_text())['models']['SVM']
    result = 'improvement' if stats['validation_macro_f1_gain'] > 0 else 'no improvement'
    report = f'''# Representation v2 - measured comparison

Date: 7 November 2026 (Asia/Karachi). Outcome: **{result} on reused validation actors**. Validation-selected winner: **{stats['winner']} / {win.model}**. Baseline is preserved byte-for-byte; no original report/notebook/KT was replaced.

## Main comparison

| Metric | Original selected SVM | New selected {win.model} | Difference |
| --- | --- | --- | --- |
| Training accuracy | {base['train_accuracy']:.4f} | {win.train_accuracy:.4f} | descriptive |
| Validation accuracy | {base['validation_accuracy']:.4f} | {win.validation_accuracy:.4f} | {100*stats['validation_accuracy_gain']:+.2f} percentage points |
| Validation macro-F1 | {base['validation_macro_f1']:.4f} | {win.validation_macro_f1:.4f} | {stats['validation_macro_f1_gain']:+.4f} |
| Reused test accuracy | 0.5526 | {score.accuracy:.4f} | {100*stats['reused_test_accuracy_gain']:+.2f} percentage points |
| Reused test macro-F1 | 0.5497 | {score.macro_f1:.4f} | {stats['reused_test_macro_f1_gain']:+.4f} |

## Method and selection evidence

Two fixed representations: 88 unfiltered eGeMAPSv02 functionals and a 361-feature fusion with the original 273 filtered features. This compares a package of representation and regularization/search changes; it is not a causal ablation that assigns every gain exclusively to eGeMAPS. Original actor allocation, labels and data exclusions are unchanged. Details, grids, citations, licensing and rationale are in [PROTOCOL.md](PROTOCOL.md).

Forty configurations underwent training-only 3-fold actor-grouped validation (120 fits). One configuration per family and representation was refitted on the 5,149 training clips only. Ten finalists were compared using validation macro-F1, accuracy and lexical key. The winner's grouped training CV macro-F1 is {win.cv_macro_f1:.4f} (fold SD {win.cv_std:.4f}). This is a selected CV estimate, not nested CV or a new external result. No train+validation refit or random-clip early stopping occurred. Warning messages are retained in the search and finalist CSVs; max_iter warnings mean some optimizers did not meet their convergence criterion.

{markdown_table(valid[['key','model','cv_macro_f1','train_macro_f1','validation_macro_f1','validation_accuracy']])}

## Five required families on reused test actors

One validation-selected representation per family was evaluated after selection.json was frozen. Test results never chose the winner. Test output is cached on rerun; do not tune using it.

{markdown_table(test[['key','model','accuracy','macro_f1','balanced_accuracy']])}

## Uncertainty and interpretation

Paired actor bootstrap (1,000 resamples, seed 42), new minus baseline:

- Validation macro-F1 difference 95% percentile interval: {stats['validation_paired_interval']['macro_f1_difference_95']}.
- Reused test macro-F1 difference 95% percentile interval: {stats['test_paired_interval']['macro_f1_difference_95']}.
- Reused test accuracy difference 95% percentile interval: {stats['test_paired_interval']['accuracy_difference_95']}.

Both models are evaluated on the same resampled actors. Intervals condition on fitted models and ignore training/selection uncertainty; the validation interval is especially optimistic because this same validation set selected the winner. A positive point difference is a measured gain on this allocation, not a guarantee or independent significance claim. Only 14 validation and 14 test actors are available. Full numeric intervals are in comparison.json.

The test actors had already been evaluated during the baseline. This is explicitly a **reused test**, not a genuinely untouched test set. The shared validation actors are also a reused development holdout. A new corpus or independently collected speakers would be needed for stronger confirmation. Acted English speech, modified noise and intended filename labels limit generalization to natural speech and other microphones/languages. Voice-quality estimates may be noisy; full utterance functionals still discard frame order. This experiment does not prove why each error occurs, measure clean-reference SNR, or promise 80-90% accuracy.

## Reproducibility and deliverables

Open [improvement.ipynb](improvement.ipynb) in PyCharm using the existing .venv. See [README.md](README.md) for exact commands and inference. New requirements-lock.txt captures the extended environment; baseline dependency files remain unchanged. features_metadata.json lists every eGeMAPS column, grouped_folds.csv fixes internal actor assignments, grouped_search.csv records all fits, selection.json freezes validation choice, and winner.joblib retains training-fitted transforms. Only load trusted local joblib files. [KT_ADDENDUM.md](KT_ADDENDUM.md) explains the change in Roman Urdu. No commit, push or publication occurred.
'''
    personal_reports = sorted((root/'voice_evaluation/sessions').glob('*/evaluation.json'))
    if personal_reports:
        personal = json.loads(personal_reports[-1].read_text())
        report += '\n## Additional real personal-voice evaluation\n\nTwenty user microphone recordings: five each happy, disgust, sad and anger. One speaker/session; four supported target emotions, six possible model outputs. This separate personal evaluation never trains or selects the model.\n\n'
        for name, values in personal['models'].items():
            report += f"- {name}: {values['correct']}/20 correct; accuracy {values['accuracy']:.1%}; four-supported-class macro-F1 {values['macro_f1_four_supported']:.4f}; fixed-six macro-F1 {values['macro_f1_six_fixed']:.4f}.\n"
        report += '\nDo not directly compare four-supported-class F1 with six-supported-class CREMA-D F1. Self-reported intended emotions, one microphone/room, repeated sentence and correlated takes limit conclusions. This is not a multi-speaker external validation. Local evidence: `'+str(personal_reports[-1].parent.relative_to(root))+'`; raw voice files remain ignored by Git.\n'
    else:
        report += '\n## Personal microphone extension\n\nThe user requested twenty real recordings, five each happy/disgust/sad/anger. The [local recorder](../../voice_evaluation/README.md) supports this additional personal evaluation. No personal accuracy is claimed until all twenty real recordings are saved and evaluated. Synthetic safeguard tests do not count as voice recordings.\n'
    (out/'comparison_report.md').write_text(report,encoding='utf-8')
    guide = f'''# Beginner KT addendum: representation v2

7 November 2026, Asia/Karachi. Yeh **alag experiment** hai. Purani 30-chapter KT guide baseline ka dated snapshot hai aur usay badla nahi gaya.

## Kya result mila?

Validation se select hua model **{win.model}**, key **{stats['winner']}** hai. Purane SVM ki validation accuracy 50.17% thi; naye winner ki **{100*win.validation_accuracy:.2f}%** hai. Macro-F1 0.4983 se **{win.validation_macro_f1:.4f}** hua. Accuracy ka farq **{100*stats['validation_accuracy_gain']:+.2f} percentage points** hai. Percentage points seedha do percentages ka subtraction hota hai; isay relative percent increase ke saath mix nahi karna.

Purane selected SVM ki test accuracy 55.26% thi; naye selected winner ki **{100*score.accuracy:.2f}%** hai. Yeh **reused test** hai: hum baseline ke test results pehle dekh chuke thay. Is liye isay bilkul naya untouched test kehna ghalat hoga. Naya winner validation se freeze kiya gaya, test dekh kar badla nahi gaya. Agar kisi doosre model ka test score behtar ho, tab bhi winner automatically change nahi hota.

## Features ko kyun badla?

Baseline har clip ko 273 numbers mein badalta tha. Bohat se numbers frames ka mean/std thay. Mean tumhein average batata hai, lekin pitch kab upar gayi aur voice kitni unstable thi uski poori detail nahi deta. openSMILE eGeMAPSv02 speech research ka established 88-feature set hai. Yeh recording ko measure karta hai; internet se trained emotion labels wala model download nahi hua.

- **F0 / pitch:** awaaz ki bunyadi frequency. Semitone representation, percentiles aur rising/falling slopes awaaz ka range aur movement summarize karte hain. Pitch emotion ki guarantee nahi; speaker bhi pitch ko affect karta hai.
- **Loudness:** perceptual energy ka estimate. Percentiles distribution ko samjhate hain; peaks per second rough rhythm cue dete hain.
- **Jitter / shimmer:** pitch periods aur amplitude ki short-term irregularity ke estimates. Noisy recording mein estimates imperfect ho sakte hain; medical diagnosis nahi hain.
- **HNR:** harmonic aur noise-like components ka ratio estimate. Yeh clean paired reference se measured denoising SNR improvement nahi hai.
- **Formants:** vocal tract ke resonance bands, roughly F1/F2/F3; voice production ki information dete hain.
- **Voiced/unvoiced segments:** voice vibration wale aur baghair vibration wale hisson ki lengths aur counts. Yeh transcript ya sentence meaning nahi dete.

`egemaps` branch mein 88 numbers unfiltered audio se aate hain. `fusion` mein baseline ke 273 filtered numbers aur 88 raw-audio numbers jurtay hain: **273 + 88 = 361**. Low-pass ka existing implementation use hota hai. Raw supplementary features ko filtered keh kar present nahi kiya. eGeMAPS bhi akhir mein clip summaries banata hai, is liye temporal information ka masla poori tarah solve hone ka claim nahi.

## Grouped validation kaise kaam karti hai?

Original training mein 63 actors hain. Unhi actors ko 3 internal folds mein baanta. Har round mein kuch training actors fit ke liye aur alag actors internal holdout ke liye thay. Ek actor ki clips dono taraf nahi gayin. Original 14 validation aur 14 test actors internal search mein shamil nahi thay. Is se hyperparameters ko naye speakers par compare karna random clips ke muqable mein munasib hai.

Har representation par KNN, Logistic Regression, GaussianNB, SVM aur MLP mila kar 20 configurations thay. Dono representations se 40 configurations, har ek ke 3 fits: **120 internal fits**. Har family/representation ka best configuration training CV se chuna, phir 10 finalists ko poore original training par fit kiya. Original validation ne final winner select kiya. Validation par scaler ya classifier fit nahi hua.

StandardScaler ke mean/std sirf fit actors se seekhay gaye. Optional shrinkage LDA label-aware projection hai: covariance estimates ko stable banane ke liye shrinkage lagti hai. Isay har fold ke andar fit karna zaroori hai, warna holdout information leak ho sakti hai. MLP mein random-clip early stopping disable hai. Agar convergence warning ho to iska matlab optimizer apne stopping criterion tak nahi pohncha; model ki predictions phir bhi available hoti hain aur warning CSV mein rehti hai.

## Is result ka kitna bharosa karein?

Macro-F1 har emotion ke F1 ko barabar weight deta hai. Accuracy sirf total correct fraction hai. Dono dekhna behtar hai. Training accuracy **{100*win.train_accuracy:.2f}%** aur validation **{100*win.validation_accuracy:.2f}%** ka gap ab bhi generalization ki limitation dikha sakta hai.

Actor bootstrap mein poora actor resample hota hai, individual clips nahi. Ek actor ki clips related hoti hain, is liye clip bootstrap uncertainty ko chhota dikha sakta hai. Paired comparison mein purana aur naya model bilkul wohi sampled actors dekhte hain. Is se score difference ka descriptive interval milta hai. Lekin yeh training aur winner selection ki uncertainty cover nahi karta. Validation pehle baseline mein bhi use hui thi aur ab winner select karti hai; is liye independent final proof nahi. Naye actors ka external corpus strong next check hoga.

Representation aur regularization/search dono badlay hain. Is experiment se yeh nahi keh sakte ke poora gain sirf ek particular pitch feature ki wajah se hua. 80-90% ki guarantee nahi. Noise filtering se clean SNR improve hua, yeh bhi measure nahi kiya.

## PyCharm mein chalana aur inference

Project `D:\\ML` open karo, interpreter `.venv\\Scripts\\python.exe` select karo. Isi folder ka `improvement.ipynb` kholo aur Run All karo. Verified caches milne par expensive extraction/training repeat nahi hogi; identity mismatch aaye to usay bypass mat karo. `PROTOCOL.md` mein commands aur exact search hai. `comparison_report.md` measured tables deta hai. `features_metadata.json` mein tamam 88 feature names hain.

```python
from experiments.representation_v2.features import predict_emotion
print(predict_emotion('path/to/recording.wav'))
```

Inference wohi feature recipe, training scaler, optional LDA aur saved classifier use karta hai. Baseline inference ab bhi `src.inference.predict_emotion` hai. Dono functions ko confuse mat karo. Naya result baseline artifact ko replace nahi karta. Sirf trusted apne joblib files load karo.

## Files aur next handoff

`features.py` per-recording features aur inference, `run.py` grouped search/freeze/test, `verify.py` leakage aur real-artifact checks, `deliver.py` report/notebook banata hai. `selection.json` ki key final choice hai. `baseline_hashes.json` original files ki identities rakhta hai. New requirements-lock.txt extended environment record karta hai; purana lock unchanged hai. Dataset aur original notebook/report ki allocations preserve hain.

Agli chat ko batao: **v2 complete results isi folder mein hain; pehle selection.json aur comparison_report.md parho, test ko tune karne ke liye dobara use mat karo, aur baseline files mat badlo.** Naya serious experiment alag folder mein ho. Koi commit/push/publication authorize nahi ki gayi. openSMILE educational/research use ke liye free hai; commercial product ki licensing alag hai. References aur Codex assistance ka attribution PROTOCOL.md mein hai.
'''
    guide += '\n## Aapki apni voice ka test\n\nAapki confirmation ke mutabiq 5 happy, 5 disgust, 5 sad aur 5 angry, total20 real recordings ka desktop recorder bhi hai. `voice_evaluation/README.md` mein instructions hain. Mic sirf Record dabane se active hota hai. Tamam clips pehle collect hoti hain; uske baad frozen baseline aur improved model predict karte hain. Recordings training ya hyperparameter tuning mein use nahi hotin. Fear/neutral output aaye to error count hota hai. Chaar supported emotions ka F1 aur six-fixed-label F1 alag hain. Ek shakhs ki 20 takes ko 20 independent speakers mat samajhna. Real personal results comparison report mein alag section mein hain; incomplete session ki accuracy invent nahi hoti.\n'
    (out/'KT_ADDENDUM.md').write_text(guide,encoding='utf-8')
    (out/'README.md').write_text('''# Representation v2

Start with [comparison_report.md](comparison_report.md), [improvement.ipynb](improvement.ipynb), or [Roman Urdu KT addendum](KT_ADDENDUM.html).

All outputs are isolated here. Baseline remains frozen. Protocol, rationale, citations, license and exact reproduction commands: [PROTOCOL.md](PROTOCOL.md).

```python
from experiments.representation_v2.features import predict_emotion
print(predict_emotion('path/to/recording.wav'))
```

Run commands from D:\\ML using the existing .venv. The executed notebook performs resumable extraction/search, cached test reuse, comparison, plots and WAV inference. Do not launch it concurrently with run.py. Large .npz/.joblib files are local and ignored by Git. Retain these for offline inference, or regenerate from pinned requirements and the verified data. verification.json records checks, comparison.json holds exact gains and paired actor intervals. Original baseline notebook/report/KT remain the assignment's baseline deliverable; v2 supplements them. No commit or push occurred.
''',encoding='utf-8')


def notebook(root):
    out = root/'experiments/representation_v2'
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# Stronger acoustic representation: v2

Yeh original baseline ka alag improvement experiment hai. Same actors/split, training-only grouped search, five required classifiers. [Protocol](PROTOCOL.md) aur [Roman Urdu KT](KT_ADDENDUM.md) parhein. **Test reused hai; naya untouched test nahi.** Source cells reusable modules ka code dikhate hain; heavy work verified caches se resume hota hai.'''),
    code('''import os, sys, json
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/audio_features.py').exists())
sys.path.insert(0,str(ROOT))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Code
from threadpoolctl import threadpool_limits
from experiments.representation_v2.run import guard, load_data, select, evaluate, candidates
from experiments.representation_v2.features import predict_emotion
from experiments.representation_v2.verify import verify
OUT = ROOT/'experiments/representation_v2'
print('Original files verified:',guard(ROOT))
manifest, matrices = load_data(ROOT)
display(manifest.groupby('split').agg(clips=('filename','size'),actors=('actor','nunique')))
print({name: value.shape for name,value in matrices.items()})'''),
    md('## Features aur training recipe\n88 eGeMAPS raw waveform se; fusion mein 273 low-pass-filtered baseline features bhi. Fixed acoustic extractor hai, pretrained emotion classifier nahi. Shrinkage LDA aur scaling har training fold ke andar fit hote hain.'),
    code('''display(pd.Series(json.loads((OUT/'features_metadata.json').read_text())['names'],name='88 eGeMAPSv02 columns').to_frame())
display(Code((OUT/'features.py').read_text(),language='python'))
display(Code((OUT/'run.py').read_text(),language='python'))'''),
    md('## Training actors ki 3-fold search\n40 configurations, 120 internal fits; CV se ten finalists. Original validation se winner. Warnings retain ki gayi hain; random-clip early stopping nahi.'),
    code('''frozen = select(ROOT)
search = pd.read_csv(OUT/'grouped_search.csv')
validation = pd.read_csv(OUT/'validation_results.csv')
display(search[['key','model','cv_macro_f1','cv_std','seconds','warnings']])
display(validation)
print('Validation-selected winner:',frozen['winner'])
print('CV candidates with warnings:',search.warnings.fillna('').ne('').sum())'''),
    code('''fig, axes = plt.subplots(1,2,figsize=(13,5))
for rep, frame in validation.groupby('representation'):
    axes[0].scatter(frame.cv_macro_f1,frame.validation_macro_f1,label=rep,s=65)
    for row in frame.itertuples():
        axes[0].annotate(row.model,(row.cv_macro_f1,row.validation_macro_f1),fontsize=7,xytext=(3,3),textcoords='offset points')
axes[0].set(xlabel='Training grouped CV macro-F1',ylabel='Validation macro-F1',title='Ten predeclared finalists')
axes[0].legend()
ranked = validation.sort_values('validation_macro_f1')
axes[1].barh(ranked.key,ranked.validation_macro_f1,color='#207d92')
axes[1].axvline(0.4982648836,color='#bb563b',linestyle='--',label='Baseline selected SVM')
axes[1].set(xlabel='Validation macro-F1',title='Same validation actors')
axes[1].legend()
fig.tight_layout(); fig.savefig(OUT/'validation_comparison.png',dpi=140); plt.show()'''),
    md('## Selection freeze ke baad reused test\nOriginal baseline mein test dekha ja chuka hai. Ab one selected representative per family evaluate hota hai aur cache save hoti hai. Test se model selection/tuning nahi hoti.'),
    code('''test = evaluate(ROOT)
display(test)
display(json.loads((OUT/'comparison.json').read_text()))
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
predictions = pd.read_csv(OUT/'test_predictions.csv')
labels = ['anger','disgust','fear','happy','neutral','sad']
fig, ax = plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(confusion_matrix(predictions.emotion,predictions[frozen['winner']],labels=labels),display_labels=labels).plot(ax=ax,colorbar=False,cmap='Blues')
ax.set_title('Validation-selected winner: reused test actors')
ax.tick_params(axis='x',rotation=30)
fig.tight_layout(); fig.savefig(OUT/'winner_confusion.png',dpi=140); plt.show()
display(pd.DataFrame(json.loads((OUT/'winner_classification.json').read_text())).T)'''),
    md('## Raw WAV inference aur checks\nScaler, LDA aur kernel references training-only hain. File se feature extraction karke saved predictions ke saath consistency check hoti hai.'),
    code('''example = manifest.loc[manifest.split.eq('validation')].iloc[0]
with threadpool_limits(1):
    prediction = predict_emotion(example.path,ROOT)
print(example.filename,'| intended:',example.emotion,'| predicted:',prediction)
display(verify(ROOT))'''),
    md('## Interpretation\nMeasured point gain ko guaranteed future accuracy mat samjhein. Paired actor intervals selection/training uncertainty cover nahi karte. External corpus aur naye speakers par confirmation chahiye. Full utterance functionals ab bhi frame order discard karte hain. Original baseline aur uski KT unchanged hain.'),
    code("display(Markdown((OUT/'comparison_report.md').read_text(encoding='utf-8')))"),
    md('## References\nEyben et al. GeMAPS, [DOI](https://doi.org/10.1109/TAFFC.2015.2457417); [openSMILE usage](https://audeering.github.io/opensmile-python/usage.html) and [license](https://audeering.github.io/opensmile-python/). Implementation assisted by OpenAI Codex; original third-party authors retain attribution. Full provenance, scope and educational-use licensing are in PROTOCOL.md.')]
    n = nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'ML-CEP (.venv Python 3.12)','language':'python','name':'ml-cep'},'language_info':{'name':'python','version':'3.12'}})
    path = out/'improvement.ipynb'
    nbf.write(n,path)
    return path


def execute(path,root):
    from .execute import execute as execute_fresh_kernel
    execute_fresh_kernel(path)


def addendum_html(root):
    import mistune
    out = root/'experiments/representation_v2'
    content = mistune.html((out/'KT_ADDENDUM.md').read_text(encoding='utf-8'))
    for name in ['validation_comparison.png','winner_confusion.png']:
        if (out/name).exists():
            encoded = base64.b64encode((out/name).read_bytes()).decode()
            content += f'<figure><img alt="{name}" src="data:image/png;base64,{encoded}"></figure>'
    page = '<!doctype html><html lang="ur-Latn"><meta charset="utf-8"><title>ML-CEP v2 KT addendum</title><style>body{max-width:960px;margin:48px auto;padding:0 24px;font:18px/1.7 system-ui;color:#18313d;background:#f7fafb}h1,h2{line-height:1.25;color:#125c70}pre{background:#e5edf0;padding:18px;overflow:auto}img{max-width:100%;height:auto}code{font-size:.9em}figure{margin:36px 0}</style><body>'+content+'</body></html>'
    (out/'KT_ADDENDUM.html').write_text(page,encoding='utf-8')


if __name__ == '__main__':
    select(ROOT)
    evaluate(ROOT)
    stats = statistics(ROOT)
    write_docs(ROOT,stats)
    path = notebook(ROOT)
    execute(path,ROOT)
    addendum_html(ROOT)
    guard(ROOT)
