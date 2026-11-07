"""Validation-only selection and a separately frozen test stage."""

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
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from .rbf_kernel import TrainingRBFKernel

LABELS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]


def save_artifact(value, destination):
    """Keep the previous usable artifact if serialization is interrupted."""
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        joblib.dump(value, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def score_predictions(truth, predicted):
    """Return complementary metrics with all six emotions in the F1 average."""
    return {
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(
            truth, predicted, labels=LABELS, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            truth, predicted, labels=LABELS, average="weighted", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
    }


def candidates():
    """Yield 76 predeclared pipelines in a deterministic tie-break order."""
    definitions = {
        "KNN": (
            KNeighborsClassifier(n_jobs=1),
            {"n_neighbors": [3, 5, 9, 15, 21], "weights": ["uniform", "distance"]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        "GaussianNB": (GaussianNB(), {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]}),
        "SVM": (
            SVC(kernel="rbf"),
            {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01]},
        ),
        "MLP": (
            MLPClassifier(max_iter=500, early_stopping=False, random_state=42),
            {
                "hidden_layer_sizes": [(128,), (128, 64)],
                "alpha": [0.0001, 0.001],
                "learning_rate_init": [0.0003, 0.001],
            },
        ),
    }
    for family, (estimator, grid) in definitions.items():
        # Predeclared simplicity order for exact metric ties: no LDA first,
        # then ParameterGrid's deterministic order, never test performance.
        for reduction in [False, True]:
            for parameters in ParameterGrid(grid):
                steps = [("scale", StandardScaler())]
                if reduction:
                    steps.append(("lda", LinearDiscriminantAnalysis()))
                if family == "SVM":
                    steps.append(
                        ("kernel", TrainingRBFKernel(gamma=parameters["gamma"]))
                    )
                    classifier = SVC(kernel="precomputed", C=parameters["C"])
                else:
                    classifier = clone(estimator).set_params(**parameters)
                steps.append(("classifier", classifier))
                yield family, reduction, parameters, Pipeline(steps)


def experiment_identity(root):
    """Bind checkpoints to actual data, extractor and experiment definitions."""
    digest = hashlib.sha256()
    # NPZ containers include ZIP timestamps. Hash logical arrays so an identical
    # resumable extraction does not invalidate an otherwise valid checkpoint.
    with np.load(root / "artifacts/features.npz") as features:
        for key in sorted(features.files):
            array = np.ascontiguousarray(features[key])
            digest.update(key.encode())
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
    for path in [
        root / "data/split_manifest.csv",
        root / "src/audio_features.py",
        root / "src/rbf_kernel.py",
        Path(__file__),
    ]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_inputs(root):
    """Reject stale extraction artifacts before exposing aligned arrays."""
    metadata = json.loads((root / "artifacts/features_metadata.json").read_text())
    assert (
        metadata["extractor_sha256"]
        == hashlib.sha256((root / "src/audio_features.py").read_bytes()).hexdigest()
    ), "Stale extractor cache"
    assert (
        metadata["manifest_sha256"]
        == hashlib.sha256((root / "data/split_manifest.csv").read_bytes()).hexdigest()
    ), "Stale data manifest"
    manifest = pd.read_csv(root / "data/split_manifest.csv")
    bundle = np.load(root / "artifacts/features.npz")
    assert np.array_equal(bundle["filenames"], manifest.filename.to_numpy())
    return manifest, bundle


def run_selection(root):
    """Fit on training actors, rank on validation actors, then freeze winners."""
    root = Path(root)
    manifest, features = load_inputs(root)
    identity = experiment_identity(root)
    checkpoint = root / "artifacts/selection_checkpoint.joblib"
    state = (
        joblib.load(checkpoint)
        if checkpoint.exists()
        else {"identity": identity, "rows": [], "best": {}}
    )
    if state["identity"] != identity:
        raise ValueError(
            "Selection checkpoint is stale: archive old outputs before rerunning"
        )
    x = features["filtered"]
    y = manifest.emotion.to_numpy()
    train = manifest.split.eq("train").to_numpy()
    valid = manifest.split.eq("validation").to_numpy()
    with threadpool_limits(limits=1):
        for order, (family, reduction, parameters, pipeline) in enumerate(candidates()):
            if order < len(state["rows"]):
                continue
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(x[train], y[train])
            train_scores = score_predictions(y[train], pipeline.predict(x[train]))
            valid_predictions = pipeline.predict(x[valid])
            valid_scores = score_predictions(y[valid], valid_predictions)
            row = {
                "candidate": order,
                "model": family,
                "lda": reduction,
                "parameters": json.dumps(parameters),
                "seconds": time.perf_counter() - started,
                "warnings": " | ".join(str(w.message) for w in caught),
                **{"train_" + k: v for k, v in train_scores.items()},
                **{"validation_" + k: v for k, v in valid_scores.items()},
            }
            state["rows"].append(row)
            rank = (valid_scores["macro_f1"], valid_scores["accuracy"], -order)
            if family not in state["best"] or rank > state["best"][family]["rank"]:
                state["best"][family] = {
                    "rank": rank,
                    "pipeline": pipeline,
                    "row": row,
                    "validation_predictions": valid_predictions,
                }
            save_artifact(state, checkpoint)
            pd.DataFrame(state["rows"]).to_csv(
                root / "reports/search_results.csv", index=False
            )
            print(
                f"{order + 1}/76 {family} lda={reduction}: validation F1={valid_scores['macro_f1']:.4f}, {row['seconds']:.1f}s",
                flush=True,
            )
    champion_name = max(state["best"], key=lambda name: state["best"][name]["rank"])
    frozen = {
        "identity": identity,
        "selection_metric": "validation macro-F1, accuracy, candidate order",
        "winner": champion_name,
        "models": {name: item["row"] for name, item in state["best"].items()},
    }
    (root / "artifacts/frozen_selection.json").write_text(json.dumps(frozen, indent=2))
    save_artifact(state["best"], root / "artifacts/selected_models.joblib")
    save_artifact(
        state["best"][champion_name]["pipeline"],
        root / "artifacts/winning_pipeline.joblib",
    )
    pd.DataFrame([item["row"] for item in state["best"].values()]).to_csv(
        root / "reports/validation_results.csv", index=False
    )
    return frozen


def run_diagnostics(root):
    """Produce predeclared diagnostics without using test predictions."""
    root = Path(root)
    manifest, features = load_inputs(root)
    x = features["filtered"]
    y = manifest.emotion.to_numpy()
    names = features["names"].astype(str)
    train = manifest.split.eq("train").to_numpy()
    valid = manifest.split.eq("validation").to_numpy()
    best = joblib.load(root / "artifacts/selected_models.joblib")
    winner = json.loads((root / "artifacts/frozen_selection.json").read_text())[
        "winner"
    ]
    rows = []
    with threadpool_limits(limits=1):
        for name, source, columns in [
            ("unfiltered_full", features["unfiltered"], np.ones(len(names), bool)),
            ("filtered_mfcc", x, np.char.startswith(names, "mfcc")),
            (
                "filtered_spectral",
                x,
                np.char.startswith(names, "spectral")
                | np.char.startswith(names, "fourier"),
            ),
            ("filtered_combined", x, np.ones(len(names), bool)),
        ]:
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("kernel", TrainingRBFKernel(gamma="scale")),
                    ("classifier", SVC(C=1, kernel="precomputed")),
                ]
            )
            model.fit(source[train][:, columns], y[train])
            rows.append(
                {
                    "experiment": name,
                    "feature_count": int(columns.sum()),
                    **score_predictions(
                        y[valid], model.predict(source[valid][:, columns])
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(root / "reports/ablation.csv", index=False)
        actors = np.random.default_rng(42).permutation(
            np.sort(manifest.loc[train, "actor"].unique())
        )
        rows = []
        for actor_count in [16, 32, 47, 63]:
            subset = train & manifest.actor.isin(actors[:actor_count]).to_numpy()
            model = clone(best[winner]["pipeline"])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x[subset], y[subset])
            rows.append(
                {
                    "actors": actor_count,
                    "clips": int(subset.sum()),
                    "train_macro_f1": score_predictions(
                        y[subset], model.predict(x[subset])
                    )["macro_f1"],
                    "validation_macro_f1": score_predictions(
                        y[valid], model.predict(x[valid])
                    )["macro_f1"],
                    "warnings": " | ".join(str(w.message) for w in caught),
                }
            )
        pd.DataFrame(rows).to_csv(root / "reports/learning_curve.csv", index=False)
    errors = manifest.loc[valid].copy()
    errors["prediction"] = best[winner]["validation_predictions"]
    errors["correct"] = errors.emotion.eq(errors.prediction)
    errors.to_csv(root / "reports/validation_predictions.csv", index=False)
    for column in ["intensity", "duration_bin"]:
        if column == "duration_bin":
            errors[column] = pd.cut(
                errors.duration, [0, 2, 3, 4, np.inf], right=False
            ).astype(str)
        groups = [
            {
                column: key,
                "clips": len(group),
                **score_predictions(group.emotion, group.prediction),
            }
            for key, group in errors.groupby(column)
        ]
        pd.DataFrame(groups).to_csv(
            root / f"reports/validation_by_{column}.csv", index=False
        )
    return pd.read_csv(root / "reports/ablation.csv")


def evaluate_test(root):
    """Evaluate frozen models once; bootstrap whole actors for the champion."""
    root = Path(root)
    frozen = json.loads((root / "artifacts/frozen_selection.json").read_text())
    if frozen["identity"] != experiment_identity(root):
        raise ValueError("Inputs changed after freezing model selection")
    existing = root / "reports/test_results.csv"
    if existing.exists():
        return pd.read_csv(
            existing
        )  # Never silently retune or refit on a repeat notebook run.
    manifest, features = load_inputs(root)
    x = features["filtered"]
    y = manifest.emotion.to_numpy()
    train = manifest.split.eq("train").to_numpy()
    test = manifest.split.eq("test").to_numpy()
    selected = joblib.load(root / "artifacts/selected_models.joblib")
    baseline = DummyClassifier(strategy="most_frequent").fit(x[train], y[train])
    models = {
        "Majority baseline": baseline,
        **{name: item["pipeline"] for name, item in selected.items()},
    }
    rows = []
    predictions = manifest.loc[test].copy()
    for name, model in models.items():
        predicted = model.predict(x[test])
        predictions[name] = predicted
        rows.append({"model": name, **score_predictions(y[test], predicted)})
        report = classification_report(
            y[test], predicted, labels=LABELS, output_dict=True, zero_division=0
        )
        (root / "reports" / f"classification_{name.replace(' ', '_')}.json").write_text(
            json.dumps(report, indent=2)
        )
    predictions.to_csv(root / "reports/test_predictions.csv", index=False)
    winner = frozen["winner"]
    rng = np.random.default_rng(42)
    actors = predictions.actor.unique()
    bootstrap = []
    for _ in range(1000):
        sampled = rng.choice(actors, size=len(actors), replace=True)
        sample = pd.concat(
            [predictions[predictions.actor.eq(actor)] for actor in sampled],
            ignore_index=True,
        )
        bootstrap.append(
            [
                accuracy_score(sample.emotion, sample[winner]),
                f1_score(
                    sample.emotion,
                    sample[winner],
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                ),
            ]
        )
    bounds = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    (root / "reports/bootstrap_intervals.json").write_text(
        json.dumps(
            {
                "winner": winner,
                "resamples": 1000,
                "unit": "test actor",
                "accuracy": bounds[:, 0].tolist(),
                "macro_f1": bounds[:, 1].tolist(),
            },
            indent=2,
        )
    )
    pd.DataFrame(rows).to_csv(existing, index=False)
    return pd.DataFrame(rows)
