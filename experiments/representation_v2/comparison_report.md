# Representation v2 - measured comparison

Date: 7 November 2026 (Asia/Karachi). Outcome: **improvement on reused validation actors**. Validation-selected winner: **fusion-13 / SVM**. Baseline is preserved byte-for-byte; no original report/notebook/KT was replaced.

## Main comparison

| Metric | Original selected SVM | New selected SVM | Difference |
| --- | --- | --- | --- |
| Training accuracy | 0.9771 | 0.8303 | descriptive |
| Validation accuracy | 0.5017 | 0.5140 | +1.22 percentage points |
| Validation macro-F1 | 0.4983 | 0.5091 | +0.0108 |
| Reused test accuracy | 0.5526 | 0.5772 | +2.46 percentage points |
| Reused test macro-F1 | 0.5497 | 0.5729 | +0.0232 |

## Method and selection evidence

Two fixed representations: 88 unfiltered eGeMAPSv02 functionals and a 361-feature fusion with the original 273 filtered features. This compares a package of representation and regularization/search changes; it is not a causal ablation that assigns every gain exclusively to eGeMAPS. Original actor allocation, labels and data exclusions are unchanged. Details, grids, citations, licensing and rationale are in [PROTOCOL.md](PROTOCOL.md).

Forty configurations underwent training-only 3-fold actor-grouped validation (120 fits). One configuration per family and representation was refitted on the 5,149 training clips only. Ten finalists were compared using validation macro-F1, accuracy and lexical key. The winner's grouped training CV macro-F1 is 0.5561 (fold SD 0.0077). This is a selected CV estimate, not nested CV or a new external result. No train+validation refit or random-clip early stopping occurred. Warning messages are retained in the search and finalist CSVs; max_iter warnings mean some optimizers did not meet their convergence criterion.

| key | model | cv_macro_f1 | train_macro_f1 | validation_macro_f1 | validation_accuracy |
| --- | --- | --- | --- | --- | --- |
| egemaps-1 | KNN | 0.4521 | 0.5364 | 0.4161 | 0.4223 |
| egemaps-4 | Logistic Regression | 0.4956 | 0.5497 | 0.4385 | 0.4459 |
| egemaps-8 | GaussianNB | 0.4302 | 0.4588 | 0.4042 | 0.4197 |
| egemaps-15 | SVM | 0.5026 | 0.6438 | 0.4463 | 0.4546 |
| egemaps-17 | MLP | 0.5061 | 0.6730 | 0.4491 | 0.4555 |
| fusion-1 | KNN | 0.5340 | 0.6499 | 0.4994 | 0.5061 |
| fusion-2 | Logistic Regression | 0.5478 | 0.6422 | 0.4930 | 0.4991 |
| fusion-8 | GaussianNB | 0.5377 | 0.6190 | 0.5043 | 0.5122 |
| fusion-13 | SVM | 0.5561 | 0.8299 | 0.5091 | 0.5140 |
| fusion-19 | MLP | 0.5488 | 0.6440 | 0.5084 | 0.5140 |

## Five required families on reused test actors

One validation-selected representation per family was evaluated after selection.json was frozen. Test results never chose the winner. Test output is cached on rerun; do not tune using it.

| key | model | accuracy | macro_f1 | balanced_accuracy |
| --- | --- | --- | --- | --- |
| fusion-13 | SVM | 0.5772 | 0.5729 | 0.5752 |
| fusion-19 | MLP | 0.5623 | 0.5598 | 0.5604 |
| fusion-8 | GaussianNB | 0.5570 | 0.5522 | 0.5564 |
| fusion-1 | KNN | 0.5421 | 0.5397 | 0.5404 |
| fusion-2 | Logistic Regression | 0.5693 | 0.5644 | 0.5675 |

## Uncertainty and interpretation

Paired actor bootstrap (1,000 resamples, seed 42), new minus baseline:

- Validation macro-F1 difference 95% percentile interval: [-0.006505621646060594, 0.024927317148018366].
- Reused test macro-F1 difference 95% percentile interval: [-0.0024299019284275574, 0.04740824941843105].
- Reused test accuracy difference 95% percentile interval: [-0.000872676758878052, 0.04825898587933251].

Both models are evaluated on the same resampled actors. Intervals condition on fitted models and ignore training/selection uncertainty; the validation interval is especially optimistic because this same validation set selected the winner. A positive point difference is a measured gain on this allocation, not a guarantee or independent significance claim. Only 14 validation and 14 test actors are available. Full numeric intervals are in comparison.json.

The test actors had already been evaluated during the baseline. This is explicitly a **reused test**, not a genuinely untouched test set. The shared validation actors are also a reused development holdout. A new corpus or independently collected speakers would be needed for stronger confirmation. Acted English speech, modified noise and intended filename labels limit generalization to natural speech and other microphones/languages. Voice-quality estimates may be noisy; full utterance functionals still discard frame order. This experiment does not prove why each error occurs, measure clean-reference SNR, or promise 80-90% accuracy.

## Reproducibility and deliverables

Open [improvement.ipynb](improvement.ipynb) in PyCharm using the existing .venv. See [README.md](README.md) for exact commands and inference. New requirements-lock.txt captures the extended environment; baseline dependency files remain unchanged. features_metadata.json lists every eGeMAPS column, grouped_folds.csv fixes internal actor assignments, grouped_search.csv records all fits, selection.json freezes validation choice, and winner.joblib retains training-fitted transforms. Only load trusted local joblib files. [KT_ADDENDUM.md](KT_ADDENDUM.md) explains the change in Roman Urdu. No commit, push or publication occurred.

## Additional real personal-voice evaluation

Twenty user microphone recordings: five each happy, disgust, sad and anger. One speaker/session; four supported target emotions, six possible model outputs. This separate personal evaluation never trains or selects the model.

- baseline: 5/20 correct; accuracy 25.0%; four-supported-class macro-F1 0.1801; fixed-six macro-F1 0.1201.
- improved: 5/20 correct; accuracy 25.0%; four-supported-class macro-F1 0.1042; fixed-six macro-F1 0.0694.

Do not directly compare four-supported-class F1 with six-supported-class CREMA-D F1. Self-reported intended emotions, one microphone/room, repeated sentence and correlated takes limit conclusions. This is not a multi-speaker external validation. Local evidence: `voice_evaluation\sessions\20261107_194332_9e11217c`; raw voice files remain ignored by Git.
