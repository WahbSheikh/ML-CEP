"""Write a factual report from completed experiment artifacts."""

from pathlib import Path
import json
import platform
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(frame):
    frame = frame.copy()
    for column in frame:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = [
        "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def main():
    manifest = pd.read_csv(ROOT / "data/split_manifest.csv")
    audit = pd.read_csv(ROOT / "reports/data_audit.csv")
    search = pd.read_csv(ROOT / "reports/search_results.csv")
    validation = pd.read_csv(ROOT / "reports/validation_results.csv")
    test = pd.read_csv(ROOT / "reports/test_results.csv")
    ablation = pd.read_csv(ROOT / "reports/ablation.csv")
    learning = pd.read_csv(ROOT / "reports/learning_curve.csv")
    frozen = json.loads((ROOT / "artifacts/frozen_selection.json").read_text())
    intervals = json.loads((ROOT / "reports/bootstrap_intervals.json").read_text())
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    schema = json.loads((ROOT / "artifacts/feature_schema.json").read_text())
    winner = frozen["winner"]
    validation_order = validation.sort_values(
        ["validation_macro_f1", "validation_accuracy"], ascending=False
    )
    validation_margin = (
        validation_order.iloc[0].validation_macro_f1
        - validation_order.iloc[1].validation_macro_f1
    )
    runner_up = validation_order.iloc[1]["model"]
    test_leader = test.sort_values("macro_f1", ascending=False).iloc[0]
    selected = test[test.model.eq(winner)].iloc[0]
    class_report = json.loads(
        (
            ROOT / "reports" / f"classification_{winner.replace(' ', '_')}.json"
        ).read_text()
    )
    per_class = pd.DataFrame(
        {key: value for key, value in class_report.items() if isinstance(value, dict)}
    ).T.reset_index(names="class")
    baseline = test[test.model.eq("Majority baseline")].iloc[0]
    errors = pd.read_csv(ROOT / "reports/validation_predictions.csv")
    confusions = (
        errors[~errors.correct]
        .groupby(["emotion", "prediction"])
        .size()
        .sort_values(ascending=False)
        .rename("clips")
        .reset_index()
    )
    split = (
        manifest.groupby("split")
        .agg(clips=("filename", "size"), actors=("actor", "nunique"))
        .reset_index()
    )
    split["percent"] = 100 * split.clips / len(manifest)
    filtered = ablation[ablation.experiment.eq("filtered_combined")].iloc[0]
    unfiltered = ablation[ablation.experiment.eq("unfiltered_full")].iloc[0]
    difference = filtered.macro_f1 - unfiltered.macro_f1
    filter_statement = f"Filtering changed fixed-SVM validation macro-F1 by {difference:+.4f}. This is an observed classification difference, not an SNR measurement."
    report = f"""# Audio emotion recognition on unseen actors
## ML-CEP Part 2 — final experimental report

## Abstract
This project implements six-class acoustic emotion recognition on the assignment-modified CREMA-D collection. It audits {len(audit):,} recordings, retains {len(manifest):,}, splits speakers before signal processing, extracts {len(schema)} acoustic features, and compares five required classifiers through {len(search)} predeclared validation configurations. The validation-selected {winner} achieved held-out accuracy {selected.accuracy:.4f} and macro-F1 {selected.macro_f1:.4f}. The majority baseline achieved accuracy {baseline.accuracy:.4f} and macro-F1 {baseline.macro_f1:.4f}. These are measurements on one fixed speaker split, not promised deployment performance.

## 1. Objective and assignment coverage
The task is to identify intended anger, disgust, fear, happiness, neutrality and sadness from recorded speech. Audio features capture vocal expression; no speech-to-text or language model is used. The project covers noisy-audio filtering, feature literature, extraction, KNN, Logistic Regression, Naive Bayes, SVM and a neural network. See [requirement mapping](../docs/requirements_checklist.md).

## 2. Dataset and audit
Source: [{provenance["repository"]}]({provenance["repository"]}), immutable commit `{provenance["commit"]}`. Actual file hashes and labels are recorded in the manifest. The original dataset paper and the modified assignment collection are separately attributed in [references](../docs/references.md).

Valid/excluded file counts:

{table(audit.groupby("status").size().rename("files").reset_index())}

Excluded files and recorded reasons:

{table(audit[audit.status.ne("valid")][["filename", "issue"]])}

The invalid recording `1076_MTI_SAD_XX.wav` contains only zero-valued samples. All members of each conflicting-label PCM pair were excluded; original raw files remain unchanged.

Split assignment shuffles actors once with seed 42. The 63/14/14 actor allocation makes clip proportions approximate. No actor appears in multiple sets and no exact decoded-audio duplicate is allowed. All six classes appear in every set.

{table(split)}

{table(pd.crosstab(manifest.split, manifest.emotion).reset_index())}

![Dataset distribution](figures/dataset_overview.png)

## 3. Signal processing and features
Mono conversion and 16 kHz resampling precede a fourth-order Butterworth low-pass at 5 kHz, implemented in second-order sections with forward-backward filtering. The two-pass magnitude is squared and phase delay is removed. Means and standard deviations summarize short-time features. Full mathematical definitions and limitations are in [methodology](../docs/methodology.md).

The {len(schema)} features include MFCCs and first/second deltas, log-Mel bands, standard 12-bin chroma, Fourier-band power, spectral centroid/bandwidth/contrast, bicoherence magnitude/phase moments, RMS, zero-crossing rate, rolloff, flatness and a pitch proxy. The feature schema is saved alongside the model. Coarse bicoherence estimates and ordinary wrapped-phase moments are explicitly qualified rather than presented as robust physical measurements.

![Filtering example](figures/filter_example.png)

![Two-pass filter response](figures/filter_response.png)

## 4. Selection protocol and validation results
Scale each candidate using training clips only. Compare with/without training-only supervised LDA. Select by validation macro-F1, then accuracy, then predeclared candidate order. Freeze all configurations before the held-out stage. The MLP uses no internal random validation. Models are not refitted on validation data. The test set is never used to choose the winner.

RBF distances are vectorized against fitted training references, then supplied to SVC's precomputed-kernel interface. This preserves the RBF formula and gamma/C grid; tests compare predictions and decision values against standard RBF SVC. Training references remain in the saved pipeline for inference. All SVM candidates use this implementation; see the methodology for its memory/runtime tradeoff.

{table(validation[["model", "lda", "parameters", "train_macro_f1", "validation_macro_f1", "validation_accuracy", "seconds"]])}

**Validation-selected champion: {winner}.** Recorded-warning candidates: {search.warnings.fillna("").ne("").sum()} of {len(search)}. Full warnings and timings are in `search_results.csv`. A warning means the iteration limit/tolerance may have affected optimization; it is not silently treated as successful convergence. Timings include fitting plus scoring and are machine-specific.

The validation macro-F1 margin over {runner_up} is only {validation_margin:.4f}. This small advantage on one partition does not establish strong or statistically significant superiority.

![Validation comparison](figures/validation_comparison.png)

## 5. Controlled ablation and learning curve
All four ablations use an unchanged SVM (C=1, gamma=scale), no LDA and identical training/validation actors. They diagnose feature/filter choices without changing the predeclared final representation.

{table(ablation)}

{filter_statement}

The learning curve uses nested subsets of training actors with the same validation set and the selected configuration:

{table(learning.drop(columns=["warnings"]))}

At the full training size the training/validation macro-F1 gap is {learning.iloc[-1].train_macro_f1 - learning.iloc[-1].validation_macro_f1:.4f}. A single curve cannot distinguish every source of variance, and its configuration was already selected on this validation set.

![Diagnostic experiments](figures/diagnostic_experiments.png)

## 6. Held-out test results
Every selected family model and the majority baseline are evaluated with their frozen training-only pipelines. The following comparison is descriptive; the largest test row does not replace the validation-selected champion.

{table(test)}

The highest descriptive test macro-F1 belongs to {test_leader["model"]} ({test_leader.macro_f1:.4f}). The saved champion remains {winner}, chosen before these test results were available. No test-driven switch or retuning was performed.

The champion's actor-cluster bootstrap 95% intervals (1,000 resamples, 14 test actors) are accuracy [{intervals["accuracy"][0]:.4f}, {intervals["accuracy"][1]:.4f}] and macro-F1 [{intervals["macro_f1"][0]:.4f}, {intervals["macro_f1"][1]:.4f}]. These percentile intervals do not include model-selection or training variability, and are not a paired significance test.

![Held-out confusion matrices](figures/test_confusion_matrices.png)

Per-class precision/recall/F1 for the validation-selected champion:

{table(per_class)}

All family reports are saved in `classification_*.json`. Raw predictions are retained for reproducibility.

## 7. Error analysis
The most frequent validation confusions are:

{table(confusions.head(10))}

{table(pd.read_csv(ROOT / "reports/validation_by_intensity.csv"))}

{table(pd.read_csv(ROOT / "reports/validation_by_duration_bin.csv"))}

Intensity-group class coverage:

{table(errors.groupby("intensity").emotion.nunique().rename("classes_present").reset_index())}

Subgroup macro-F1 uses all six labels, including zero contributions from absent classes. Balanced accuracy averages recalls of classes actually present. Intensity groups with different class coverage are therefore not directly comparable; these tables are descriptive, not evidence that intensity alone caused a performance difference.

The notebook provides audio examples for the two most frequent confusion directions. Similar energy, pitch or voice quality may be relevant, but listening alone cannot establish the model's causal explanation. Intended emotion is not necessarily perceived emotion; short/low-intensity clips can be ambiguous. Subgroup sample counts matter and the observations are not claims about all speakers.

## 8. Original contribution and limitations
The code and prose were independently developed with coding-assistant support. The supplied student repository was used to understand the scope, not as a source of code or results. The contribution is an auditable implementation: actor-disjoint splits, corrected preprocessing boundaries, a dedicated validation stage, all suggested feature families, controlled experiments, saved inference and actor-level uncertainty. Familiar library algorithms remain attributed to their original sources.

One acted corpus cannot establish robustness to spontaneous emotion, another language, new microphones or background environments. Shared sentence texts mean unseen-sentence generalization is not tested. Summary features lose temporal detail. The large correlated feature set and a single validation split can introduce selection variance. GaussianNB assumptions, coarse bicoherence, wrapped-phase moments and pitch-proxy limitations remain material. Filter attenuation is not proof of denoising. Test scores from leaked or random-clip reference splits are not directly comparable.

## 9. Reproducibility and conclusion
Use Python {platform.python_version()}, the pinned dependency lock, seed 42, the frozen data manifest and the provided fresh-kernel execution command. See [README](../README.md) for setup, artifact paths and execution. Saved-model inference reuses the exact extraction schema and training-fitted transforms.

The experiment answers the stated question on held-out actors with {winner} selected by validation evidence. Its measured accuracy is {selected.accuracy:.4f} and macro-F1 is {selected.macro_f1:.4f}. Further gains should be investigated with grouped repeated/nested validation and an external corpus, without recycling this test set for tuning.

## References
See the [complete bibliography and attribution](../docs/references.md).
"""
    (ROOT / "reports/final_report.md").write_text(report, encoding="utf-8")
    log = f"""# Experiment log

- Dataset immutable commit: `{provenance["commit"]}`; verified WAVs: {len(audit)}.
- Split: seed 42, 63/14/14 actors; manifest frozen before extraction.
- Features: {len(schema)} columns, filtered and unfiltered counterparts.
- Search: {len(search)} candidates; training-only fit; selection metric validation macro-F1.
- Winner frozen before test: **{winner}**.
- Diagnostics: four fixed-SVM ablations, four nested training-actor sizes.
- Final evaluation: five family winners and majority baseline; no validation refit.
- Confidence intervals: 1,000 test-actor bootstrap resamples.
- Full candidate settings, scores, warning text and elapsed seconds: [search_results.csv](search_results.csv).
- Selection fingerprint and configuration: `artifacts/frozen_selection.json`.
- No commits or pushes performed by the implementation workflow.
"""
    (ROOT / "reports/experiment_log.md").write_text(log, encoding="utf-8")
    print("Wrote report and experiment log")


if __name__ == "__main__":
    main()
