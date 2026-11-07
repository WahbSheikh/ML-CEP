"""Create a readable notebook with executable analysis, not pasted outputs."""

from pathlib import Path
import ast
import json
import uuid

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(
        {
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": text.strip(),
        }
    )


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": text.strip(),
            "execution_count": None,
            "outputs": [],
        }
    )


def include_function(name):
    """Embed this project's own function so the training logic is visible."""
    source = (ROOT / "src/experiments.py").read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    code(ast.get_source_segment(source, function))


md("""# Hearing emotion: a speaker-independent audio classification study
## ML-CEP · Part 2

**Question:** can acoustic features distinguish anger, disgust, fear, happiness, neutrality and sadness in actors not used for training?

This notebook follows the assignment's noisy CREMA-D data, low-pass filtering and five required classifiers. It adds explicit speaker separation, controlled ablations, uncertainty estimates and error analysis. All numerical outputs are produced by this project. See `docs/references.md` for sources and attribution.

**Reading guide:** audit → freeze splits → inspect filtering → extract features → select on validation → diagnose → evaluate held-out test → discuss limitations. No test-informed model tuning is permitted.""")
code("""import os, sys, json, subprocess
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
assert (ROOT / 'src').exists(), 'Open this notebook from the ML project'
sys.path.insert(0, str(ROOT))
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Audio, Markdown, display
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from threadpoolctl import threadpool_limits
from src.audio_features import SETTINGS, load_audio, lowpass, extract_file
from src.experiments import LABELS, run_selection, run_diagnostics, evaluate_test, candidates
sns.set_theme(style='whitegrid', palette='colorblind', context='notebook')
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
FIGURES = ROOT / 'reports/figures'
FIGURES.mkdir(parents=True, exist_ok=True)
print('Python:', sys.version.split()[0], '| Project:', ROOT)
display(pd.Series(SETTINGS.__dict__, name='Audio processing settings'))""")
md("""## 1. Data provenance, audit and split
The specified source is a modified/noisy CREMA-D collection. Filename codes give intended emotions, not crowd-majority perception. We use six emotion classes, not binary sentiment. A raw file hash plus a decoded PCM hash detects exact duplication.

**Split first:** seed 42 assigns 63/14/14 actors to training/validation/test before feature extraction. Actor counts are fixed; clip percentages are approximate. Inspecting validation labels for coverage is not training on validation. Test class counts are shown for integrity only, not used to tune models.""")
code("""if not (ROOT / 'artifacts/features.npz').exists():
    subprocess.run([sys.executable, str(ROOT / 'scripts/extract_features.py')], cwd=ROOT, check=True)
manifest = pd.read_csv(ROOT / 'data/split_manifest.csv')
audit = pd.read_csv(ROOT / 'reports/data_audit.csv')
display(json.loads((ROOT / 'data/provenance.json').read_text()))
display(audit.groupby('status', dropna=False).size().rename('files').to_frame())
display(audit[audit.status.ne('valid')][['filename', 'issue']])
summary = manifest.groupby('split').agg(clips=('filename','size'), actors=('actor','nunique'),
    duration_min=('duration','min'), duration_max=('duration','max'))
summary['clip_percent'] = 100 * summary.clips / len(manifest)
display(summary.loc[['train','validation','test']])
display(pd.crosstab(manifest.split, manifest.emotion)[LABELS])
display(manifest.groupby(['sample_rate','channels']).size().rename('files').to_frame())
assert not manifest.pcm_sha256.duplicated().any()
assert manifest.groupby('actor').split.nunique().max() == 1
assert manifest.groupby('split').emotion.nunique().eq(6).all()
print('Integrity: no exact decoded-audio duplicates; actors are disjoint; all classes present.')""")
code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=manifest, x='emotion', order=LABELS, ax=axes[0], color='#327e9d')
axes[0].set(title='Intended emotion distribution', xlabel='', ylabel='Clips')
sns.histplot(data=manifest, x='duration', hue='split', bins=35, element='step', ax=axes[1])
axes[1].set(title='Clip lengths by partition', xlabel='Duration (seconds)')
fig.tight_layout(); fig.savefig(FIGURES / 'dataset_overview.png', bbox_inches='tight'); plt.show()""")
md("""## 2. What does the low-pass filter change?
We use 16 kHz mono audio and a fourth-order Butterworth design with 5 kHz cutoff. Forward-backward SOS filtering removes phase delay and squares the single-pass magnitude response. The resulting response is approximately -6 dB at the design cutoff. Playback disables automatic normalization so the comparison does not hide amplitude changes.

High-frequency attenuation is measurable. Improved denoising/SNR cannot be established without a paired clean reference. Listen and compare the spectrum before claiming an improvement.""")
code("""from scipy import signal
import librosa, librosa.display
example = manifest[manifest.split.eq('train')].iloc[0]
original = load_audio(example.path)
filtered = lowpass(original)
display(Markdown(f"Training example: **{example.filename}**, intended emotion **{example.emotion}**"))
display(Markdown('Original')); display(Audio(original, rate=SETTINGS.sample_rate, normalize=False))
display(Markdown('Filtered')); display(Audio(filtered, rate=SETTINGS.sample_rate, normalize=False))
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
time = np.arange(len(original)) / SETTINGS.sample_rate
axes[0,0].plot(time, original, alpha=.65, label='Original')
axes[0,0].plot(time, filtered, alpha=.65, label='Filtered')
axes[0,0].set(xlabel='Seconds', ylabel='Amplitude', title='Waveform'); axes[0,0].legend()
for label, samples in [('Original',original),('Filtered',filtered)]:
    frequencies, power = signal.welch(samples, fs=SETTINGS.sample_rate, nperseg=512)
    axes[0,1].plot(frequencies, 10*np.log10(power+1e-20), label=label)
axes[0,1].set(xlabel='Frequency (Hz)', ylabel='PSD (dB/Hz)', title='Power spectral density'); axes[0,1].legend()
for ax, label, samples in [(axes[1,0],'Original',original),(axes[1,1],'Filtered',filtered)]:
    spectrum = librosa.amplitude_to_db(abs(librosa.stft(samples,n_fft=512,hop_length=160)), ref=1.0)
    image = librosa.display.specshow(spectrum, sr=16000, hop_length=160, x_axis='time', y_axis='linear', ax=ax, vmin=-80, vmax=20)
    ax.set_title(label+' spectrogram'); fig.colorbar(image,ax=ax,format='%+2.0f dB')
fig.tight_layout(); fig.savefig(FIGURES/'filter_example.png',bbox_inches='tight'); plt.show()
sos = signal.butter(4,5000,fs=16000,output='sos')
frequencies,response = signal.sosfreqz(sos,fs=16000,worN=2048)
plt.figure(figsize=(8,3)); plt.plot(frequencies,20*np.log10(np.maximum(abs(response)**2,1e-8)))
plt.axvline(5000,color='grey',linestyle='--'); plt.ylim(-100,3)
plt.xlabel('Frequency (Hz)'); plt.ylabel('Forward-backward gain (dB)'); plt.title('Actual two-pass response')
plt.tight_layout(); plt.savefig(FIGURES/'filter_response.png'); plt.show()""")
md("""## 3. From a variable-length recording to one feature vector
Feature families: Fourier-band power, MFCCs and deltas, log-Mel, 12-bin chroma, centroid, bandwidth, contrast and bicoherence. RMS, zero-crossing rate, rolloff, flatness and a pitch proxy add context. Frame means/std produce fixed-length vectors. See `docs/methodology.md` for exact settings and literature rationale.

**Bicoherence:** `E[X1 X2 conj(X1+2)] / sqrt(E[|X1 X2|²] E[|X1+2|²])`. We average windowed segments before normalizing; this is not the raw bispectrum. Coarse frequency pairs limit cost. Ordinary wrapped-phase moments are boundary-sensitive and require cautious interpretation.

Per-clip features require no global fitting. In contrast, scaling and LDA learn from data and must stay inside training pipelines. The feature cache includes source and implementation hashes.""")
code("""bundle = np.load(ROOT / 'artifacts/features.npz')
features = bundle['filtered']; feature_names = bundle['names'].astype(str)
assert np.array_equal(bundle['filenames'], manifest.filename.to_numpy())
assert np.isfinite(features).all() and np.isfinite(bundle['unfiltered']).all()
display(pd.Series(feature_names).str.extract(r'^([a-z]+)')[0].value_counts().rename('columns').to_frame())
print('Filtered matrix:', features.shape, '| Unfiltered matrix:', bundle['unfiltered'].shape)
display(pd.DataFrame({'feature': feature_names, 'first_training_clip': features[manifest.split.eq('train')][0]}).head(12))""")
md("""## 4. Five classifiers, one validation policy
KNN uses local distances; logistic regression learns linear log-odds; GaussianNB assumes class-conditional independent Gaussian features; RBF SVM builds nonlinear margins; MLP learns nonlinear hidden representations. Each is evaluated with training-only scaling, with and without supervised LDA.

We choose within each family by **validation macro-F1**, then accuracy, then predeclared candidate order. The overall winner is selected using the same rule. Macro-F1 gives equal weight to each emotion. Distance-weighted KNN often predicts its own training points perfectly, so training scores are not generalization estimates. A dedicated validation partition remains separate; MLP internal random validation is disabled.""")
md("""### Readable training implementation
The following cells contain this project's actual candidate definitions and training/validation loop, synchronized from our reusable module when the notebook is built. They are executable code, not screenshots. Checkpoints let a completed full search be reopened without retraining; source/data fingerprints prevent stale reuse. Every pipeline fits training data only.

For SVM, `TrainingRBFKernel` computes the ordinary RBF Gram matrix against training references and supplies it to SVC's precomputed interface. This keeps the same kernel, gamma/C grid and train-only fitting while avoiding repeated scalar distance calculations. Equivalence tests compare it against standard RBF SVC; see the methodology for details.""")
code("""import time, warnings, joblib
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.experiments import score_predictions, experiment_identity, load_inputs, save_artifact
from src.rbf_kernel import TrainingRBFKernel""")
include_function("candidates")
include_function("run_selection")
code("""candidate_summary = []
from src.experiments import candidates as reusable_candidates
notebook_specs = [(name, lda, parameters) for name, lda, parameters, _ in candidates()]
source_specs = [(name, lda, parameters) for name, lda, parameters, _ in reusable_candidates()]
assert notebook_specs == source_specs, 'Edit src/experiments.py and rebuild the notebook before changing the search'
for family, lda, parameters, pipeline in candidates():
    candidate_summary.append({'model': family, 'LDA': lda, 'parameters': str(parameters)})
display(pd.DataFrame(candidate_summary).groupby(['model','LDA']).size().rename('candidates').to_frame())
print('Total candidates:', len(candidate_summary))
frozen = run_selection(ROOT)
search = pd.read_csv(ROOT / 'reports/search_results.csv')
validation = pd.read_csv(ROOT / 'reports/validation_results.csv')
display(validation[['model','lda','parameters','train_macro_f1','validation_macro_f1','validation_accuracy','seconds','warnings']])
print('Validation-selected winner:', frozen['winner'])
print('Candidates with recorded warnings:', search.warnings.fillna('').ne('').sum())""")
code("""fig, axes = plt.subplots(1,2,figsize=(14,5))
chart = validation.melt(id_vars='model', value_vars=['train_macro_f1','validation_macro_f1'],var_name='partition',value_name='macro_f1')
chart['partition'] = chart.partition.map({'train_macro_f1':'Training','validation_macro_f1':'Validation'})
sns.barplot(data=chart,x='model',y='macro_f1',hue='partition',ax=axes[0]); axes[0].set_ylim(0,1)
axes[0].tick_params(axis='x',rotation=25); axes[0].set_title('Training versus unseen validation actors')
axes[0].set(xlabel='',ylabel='Macro-F1')
axes[0].legend(title=None,loc='lower left',bbox_to_anchor=(0,1.12),ncol=2,frameon=False)
search_plot = search.assign(representation=np.where(search.lda,'With LDA','Without LDA'))
sns.scatterplot(data=search_plot,x='seconds',y='validation_macro_f1',hue='model',style='representation',ax=axes[1])
axes[1].set(xlabel='Fit + scoring seconds',ylabel='Validation macro-F1',title='Search quality and computational cost')
axes[1].legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
fig.tight_layout(); fig.savefig(FIGURES/'validation_comparison.png',bbox_inches='tight'); plt.show()""")
md("""## 5. Controlled diagnostics on validation data
The ablation holds SVM settings fixed at C=1, gamma=scale and uses no LDA. This isolates the representation more fairly than comparing separately tuned models. The filtered combined representation stays predeclared for final models; these diagnostics do not trigger a test-informed redesign.

The learning curve uses nested actor subsets and unchanged validation actors. It is conditional on the validation-selected configuration, so it is not an independent estimate of model-selection performance.""")
code("""ablation = run_diagnostics(ROOT)
learning = pd.read_csv(ROOT / 'reports/learning_curve.csv')
display(ablation); display(learning)
fig, axes = plt.subplots(1,2,figsize=(12,4))
ablation_plot = ablation.assign(description=ablation.experiment.map({'unfiltered_full':'Unfiltered: all','filtered_mfcc':'Filtered: MFCC','filtered_spectral':'Filtered: spectral','filtered_combined':'Filtered: all'}))
sns.barplot(data=ablation_plot,x='description',y='macro_f1',ax=axes[0],color='#327e9d')
axes[0].tick_params(axis='x',rotation=25); axes[0].set(title='Fixed-SVM feature/filter ablation',ylim=(0,1),ylabel='Validation macro-F1',xlabel='')
axes[1].plot(learning.actors,learning.train_macro_f1,'o-',label='Training')
axes[1].plot(learning.actors,learning.validation_macro_f1,'o-',label='Validation')
axes[1].set(xlabel='Training actors',ylabel='Macro-F1',ylim=(0,1),title='Speaker-based learning curve'); axes[1].legend(loc='upper right',bbox_to_anchor=(1,.85))
fig.tight_layout(); fig.savefig(FIGURES/'diagnostic_experiments.png',bbox_inches='tight'); plt.show()""")
md("""## 6. Where does the validation winner struggle?
Inspect off-diagonal confusions and listen to representative validation errors. Acoustic explanations are hypotheses, not measured causes. Intended labels may differ from perceived emotion. Intensity and duration subgroup scores require their sample sizes; a small subgroup should not drive a broad claim. Subgroup macro-F1 uses the fixed six labels, so absent classes contribute zero; different intensity groups may have different class coverage and are not directly comparable. Balanced accuracy averages recalls over classes present in each subgroup.""")
code("""validation_predictions = pd.read_csv(ROOT / 'reports/validation_predictions.csv')
mistakes = validation_predictions[~validation_predictions.correct]
confusions = mistakes.groupby(['emotion','prediction']).size().sort_values(ascending=False).rename('clips').reset_index()
display(confusions.head(10))
display(pd.read_csv(ROOT / 'reports/validation_by_intensity.csv'))
display(pd.read_csv(ROOT / 'reports/validation_by_duration_bin.csv'))
for pair in confusions.head(2).itertuples():
    row = mistakes[(mistakes.emotion==pair.emotion)&(mistakes.prediction==pair.prediction)].iloc[0]
    display(Markdown(f"**{row.filename}:** intended {row.emotion}, predicted {row.prediction}; duration {row.duration:.2f}s, intensity {row.intensity}. Listen for energy, pitch and voice quality; these alone do not establish why the model erred."))
    display(Audio(load_audio(row.path),rate=16000,normalize=False))""")
md("""## 7. Frozen models meet the held-out actors
Selection metadata has already been written. No hyperparameters, feature choices or fitted preprocessing are changed here. All models remain fitted on training actors only. The majority-class baseline helps judge whether the models learn beyond class frequencies.

The reported champion remains the validation-selected model, regardless of which test row happens to be largest. Cluster bootstrap intervals resample complete test actors 1,000 times; they do not include training randomness or selection uncertainty.""")
code("""test_results = evaluate_test(ROOT)
display(test_results)
intervals = json.loads((ROOT/'reports/bootstrap_intervals.json').read_text())
display(intervals)
test_predictions = pd.read_csv(ROOT/'reports/test_predictions.csv')
fig, axes = plt.subplots(2,3,figsize=(15,9))
for ax, name in zip(axes.ravel(),test_results.model):
    matrix = confusion_matrix(test_predictions.emotion,test_predictions[name],labels=LABELS)
    ConfusionMatrixDisplay(matrix,display_labels=LABELS).plot(ax=ax,colorbar=False,cmap='Blues',values_format='d',im_kw={'vmin':0,'vmax':int(test_predictions.emotion.value_counts().max())})
    ax.grid(False)
    ax.set_title(name); ax.tick_params(axis='x',rotation=35)
fig.tight_layout(); fig.savefig(FIGURES/'test_confusion_matrices.png',bbox_inches='tight'); plt.show()
winner = frozen['winner']
classification = json.loads((ROOT/'reports'/f'classification_{winner.replace(" ","_")}.json').read_text())
display(pd.DataFrame(classification).T)
chosen = test_results[test_results.model.eq(winner)].iloc[0]
display(Markdown(f"Validation-selected **{winner}** obtained test accuracy **{chosen.accuracy:.3f}** and macro-F1 **{chosen.macro_f1:.3f}**. Actor-bootstrap 95% intervals: accuracy **{intervals['accuracy'][0]:.3f}–{intervals['accuracy'][1]:.3f}**, macro-F1 **{intervals['macro_f1'][0]:.3f}–{intervals['macro_f1'][1]:.3f}**."))""")
md("""## 8. Reusable prediction and verification
The saved pipeline contains the scaler, optional LDA and classifier. `predict_emotion` extracts the same filtered feature schema used during training. A prediction on a known held-out example should exactly match the recorded test-stage result. This interface is an offline demonstration, not evidence of accuracy on arbitrary microphones or languages.""")
code("""from src.inference import predict_emotion
prediction_example = test_predictions.iloc[0]
predicted_emotion = predict_emotion(prediction_example.path, ROOT)
assert predicted_emotion == prediction_example[winner]
print('File:', prediction_example.filename, '| Intended:', prediction_example.emotion, '| Predicted:', predicted_emotion)
import joblib
training_rows = manifest.split.eq('train').to_numpy()
for item in joblib.load(ROOT/'artifacts/selected_models.joblib').values():
    scaler = item['pipeline'].named_steps['scale']
    np.testing.assert_allclose(scaler.mean_, features[training_rows].mean(axis=0))
    assert scaler.n_samples_seen_ == training_rows.sum()
print('Verified training-only scaling and saved-model consistency.')""")
md("""## 9. Interpretation and limitations
The final report is generated from these saved experiment tables, not hand-entered scores. Speaker separation is stricter than random clip splitting, so this study must not compare accuracy directly against a project with a different split or data leakage.

Acted emotions, shared sentences, one dataset and intended-emotion labels limit real-world generalization. Clip summaries lose temporal structure; correlated feature families can challenge Naive Bayes; LDA assumes restrictive class structure; MLP convergence and validation selection variability remain limitations. Bicoherence estimates are coarse and phase moments are boundary-sensitive. Pitch is a proxy, not a clinical measurement. Fourteen test actors provide limited population coverage.

Future work should use another corpus and multiple grouped splits or nested grouped validation, followed by an untouched external test set. Repeated test-driven tuning would invalidate the present estimate. No high accuracy is promised or fabricated.

See `docs/methodology.md`, `docs/references.md`, `docs/viva_notes.md`, `docs/requirements_checklist.md`, and `reports/final_report.md` for detailed supporting material.""")

metadata = {
    "kernelspec": {
        "display_name": "ML-CEP (.venv Python 3.12)",
        "language": "python",
        "name": "ml-cep",
    },
    "language_info": {"name": "python", "version": "3.12"},
}
notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "notebooks/audio_emotion_analysis.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8"
)
print("Created notebook:", len(cells), "cells")
