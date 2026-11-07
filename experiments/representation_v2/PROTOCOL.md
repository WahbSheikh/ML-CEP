# Representation v2 protocol (7 November 2026)

Written before observing v2 validation or test results. Original experiment stays frozen.

## Question and resources

Can established paralinguistic measurements improve unseen-actor generalization over the baseline's clip summaries? Initial physical memory: 15.71 GiB total, 2.32 GiB available. Windows CIM access was unavailable; GlobalMemoryStatusEx was used. nvidia-smi was unavailable, so no GPU is assumed. Use one extraction worker and one BLAS thread; do not close applications. Python 3.12 / Windows x64 and openSMILE 2.6.0 passed a real-file extraction smoke check.

WavLM was considered but deferred: torch/transformers were absent, CPU embedding extraction would add significant model/runtime/memory costs, and its pretraining provenance would need separate auditing. No WavLM result is claimed.

## Fixed representations and data

Reuse all 7,435 audited clips and precisely the original 63/14/14 actor allocation. Same mono/16 kHz loader. No augmentation or split changes.

1. `egemaps`: openSMILE eGeMAPSv02, 88 utterance functionals, unfiltered waveform. Includes F0/loudness percentiles and slopes, voice quality, formants and voiced segment statistics. These are fixed signal processing measurements, without a pretrained emotion classifier or learned CREMA-D encoder. This avoids the specific risk of reusing CREMA-D-trained classifier weights, not every possible historical research overlap.
2. `fusion`: original 273 low-pass-filtered features concatenated with the 88 unfiltered eGeMAPS features (361 total). Thus the assignment's filtering remains present in this branch; extra raw-waveform measurements are explicitly an experimental addition. The eGeMAPS-only diagnostic is not presented as implementing the assignment's filtering by itself. There is no new filtering benefit or SNR claim.

The full eGeMAPS representation is still an utterance summary, not a temporal sequence model. Richer slopes and segment statistics only partially address lost temporal information.

## Predeclared search and selection

For each representation, 20 configurations:

- KNN: k=9 or 21, uniform, shrinkage LDA.
- Logistic Regression: C=0.01, 0.1, 1; with/without shrinkage LDA.
- GaussianNB: var_smoothing=1e-9 or 1e-5; shrinkage LDA.
- RBF SVM: C=1, 10, 100 crossed with gamma=scale or 0.001; no LDA.
- MLP: one 64-unit hidden layer; alpha=0.01 or 1; with/without shrinkage LDA; learning rate=0.001, max_iter=300, seed=42, no early stopping.

StandardScaler and optional LDA(solver=eigen, shrinkage=auto) are fitted inside each training fold. Shrinkage stabilizes covariance estimation. RBF uses the baseline's tested, vectorized training-reference transformer. Convergence warnings are retained. No automatic random-clip validation is used.

Use StratifiedGroupKFold(3, shuffle=True, random_state=42) on training actors only. Mean fold macro-F1 selects one configuration per family per representation, candidate order breaks exact ties. Refit these ten finalists on the original training clips only. Select overall and family winners by validation macro-F1, accuracy, lexical candidate key. Do not fit on validation. Save models and selection.json before test evaluation. Forty configurations mean 120 internal fits plus ten finalist fits.

## Evaluation boundaries

Primary improvement evidence is measured validation macro-F1 compared with 0.4982648836, plus descriptive training-only grouped results. The validation actors were also used in the baseline, so this is a reused development holdout; finalist selection introduces optimism. Neither grouped search scores nor validation bootstrap intervals provide an unbiased external performance estimate.

After selection freeze, evaluate one selected representative of each of the five required families on the original test actors once and cache results. The baseline test has already been seen. Label every comparison as **reused test**, never a new untouched test. Do not retune after viewing these results. Paired actor bootstrap differences describe uncertainty conditional on these fitted/selected models, excluding training/selection uncertainty. If no improvement occurs, report it honestly.

## Sources, licensing and attribution

- Eyben et al., GeMAPS, IEEE TAFFC, DOI [10.1109/TAFFC.2015.2457417](https://doi.org/10.1109/TAFFC.2015.2457417), [author-hosted paper](https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf). Rationale for a compact voice parameter set, not a guarantee of gains here.
- Eyben, Wollmer and Schuller (2010), openSMILE - The Munich Versatile and Fast Open-Source Audio Feature Extractor, ACM Multimedia, 1459-1462.
- [Official Python usage](https://audeering.github.io/opensmile-python/usage.html): eGeMAPSv02 functionals API and descriptors.
- [Official license notice](https://audeering.github.io/opensmile-python/): free private/research/educational use; commercial products require a commercial license. This CEP is educational. Installed package contains its license notices; do not relicense openSMILE or raw CREMA-D data.
- Existing assignment dataset provenance and student-reference attribution remain in the baseline docs. No reference-project source is copied. New experiment implementation and Roman Urdu explanations were authored with OpenAI Codex assistance; third-party feature extraction and scikit-learn algorithms remain attributed to their maintainers.

## Reproduce from the project root

```powershell
.\.venv\Scripts\python.exe -m pip install -r experiments\representation_v2\requirements-lock.txt
.\.venv\Scripts\python.exe -m experiments.representation_v2.run
.\.venv\Scripts\python.exe -m experiments.representation_v2.run --test
.\.venv\Scripts\python.exe -m experiments.representation_v2.deliver
.\.venv\Scripts\python.exe -m experiments.representation_v2.verify
.\.venv\Scripts\python.exe -m experiments.representation_v2.assemble
```

Resume reuses checkpoints only with matching source/data identities. No simultaneous training routes. baseline_hashes.json must continue to verify; do not regenerate it to excuse changed baseline files. Trusted-local joblib only. Small source/results/notebook files are shareable; large caches/models stay local. No publication, commit or push is part of these commands.

## Core CEP versus optional extensions

The original assigned Q2 workflow is primary and preserved. This representation experiment and the later user-requested personal microphone test are optional additions. The combined report reproduces the core work first, then clearly marks extra experiments and their negative findings as appendices. Full combined KT preserves the baseline snapshot first and adds optional explanations afterwards. It must not imply the extra features/voice test were assignment requirements. Personal voice design was explicitly confirmed by the user as five happy, five disgust, five sad and five angry recordings (20 total). See voice_evaluation/README.md for separate microphone dependency and commands. After evaluating a completed session, run voice_evaluation.analyze on its folder before assemble. Notebook execution needs local Jupyter sockets; sandboxed agent execution may need local network permission, whereas normal PyCharm execution uses the user's local environment.
