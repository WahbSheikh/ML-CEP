from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from scipy.signal import butter, sosfreqz
from src.audio_features import (
    SETTINGS,
    bicoherence_summary,
    extract_features,
    load_audio,
    lowpass,
)


def test_filter_attenuates_high_frequency_without_time_shift():
    rate = SETTINGS.sample_rate
    time = np.arange(rate) / rate
    low = np.sin(2 * np.pi * 1000 * time)
    high = np.sin(2 * np.pi * 7000 * time)
    filtered = lowpass(low + high)
    interior = slice(100, -100)
    assert np.sqrt(np.mean((filtered[interior] - low[interior]) ** 2)) < 0.03
    _, response = sosfreqz(
        butter(4, 5000, fs=rate, output="sos"), worN=np.array([1000, 7000]), fs=rate
    )
    assert abs(response[0]) ** 2 > 0.95
    assert abs(response[1]) ** 2 < 0.02


def test_bicoherence_detects_phase_coupled_triplet():
    # Independent phase per segment; f3 phase=f1+f2 implies quadratic coupling.
    rng = np.random.default_rng(7)
    frames = []
    for _ in range(100):
        t = np.arange(512) / 16000
        a, b = rng.uniform(-np.pi, np.pi, 2)
        frames.append(
            np.sin(2 * np.pi * 500 * t + a)
            + np.sin(2 * np.pi * 750 * t + b)
            + np.sin(2 * np.pi * 1250 * t + a + b)
        )
    coupled = np.concatenate(frames)
    result, names = bicoherence_summary(coupled)
    assert len(result) == 8 and np.isfinite(result).all()
    assert 0 <= result[0] <= 1
    scaled, _ = bicoherence_summary(coupled * 10)
    np.testing.assert_allclose(result, scaled, rtol=1e-6, atol=1e-6)
    # Independently verify normalization at known exact FFT bins on independent blocks.
    spectrum = np.fft.rfft(np.asarray(frames) * np.hanning(512), axis=1)
    product = spectrum[:, 16] * spectrum[:, 24]
    third = spectrum[:, 40]
    coupling = abs(np.mean(product * third.conj())) / np.sqrt(
        np.mean(abs(product) ** 2) * np.mean(abs(third) ** 2)
    )
    assert coupling > 0.99


def test_features_fixed_finite_and_silence_rejected(tmp_path):
    import soundfile as sf

    t = np.arange(16000) / 16000
    features, names = extract_features(
        np.sin(2 * np.pi * 220 * t)
        + 0.01 * np.random.default_rng(42).normal(size=len(t))
    )
    other, other_names = extract_features(np.sin(2 * np.pi * 300 * t[:8000]))
    assert len(features) == len(names) == len(other)
    assert len(set(names)) == len(names) and names == other_names
    assert np.isfinite(features).all() and np.isfinite(other).all()
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(1600), 16000)
    with pytest.raises(ValueError, match="silent"):
        load_audio(path)


def test_real_split_and_training_only_scaler():
    import pandas as pd
    import joblib

    root = Path(__file__).resolve().parents[1]
    if not (root / "artifacts/selected_models.joblib").exists():
        pytest.skip("Full experiment artifacts not yet available")
    manifest = pd.read_csv(root / "data/split_manifest.csv")
    audit = pd.read_csv(root / "reports/data_audit.csv")
    conflicts = audit[audit.issue.fillna("").str.contains("conflicting emotion labels")]
    assert set(conflicts.pcm_sha256).isdisjoint(manifest.pcm_sha256)
    assert audit.status.eq("valid").sum() == len(manifest)
    assert manifest.groupby("actor").split.nunique().max() == 1
    assert not manifest.pcm_sha256.duplicated().any()
    assert manifest.groupby("split").emotion.nunique().eq(6).all()
    matrix = np.load(root / "artifacts/features.npz")["filtered"]
    expected = matrix[manifest.split.eq("train")].mean(axis=0)
    for item in joblib.load(root / "artifacts/selected_models.joblib").values():
        scaler = item["pipeline"].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, expected)
        assert scaler.n_samples_seen_ == manifest.split.eq("train").sum()
        if "kernel" in item["pipeline"].named_steps:
            training_features = matrix[manifest.split.eq("train")]
            transformed = item["pipeline"][:-2].transform(training_features)
            reference = item["pipeline"].named_steps["kernel"].reference_features_
            np.testing.assert_allclose(reference, transformed)


def test_saved_inference_matches_test_prediction():
    import pandas as pd
    import json
    from src.inference import predict_emotion

    root = Path(__file__).resolve().parents[1]
    if not (root / "reports/test_predictions.csv").exists():
        pytest.skip("Full test evaluation not yet available")
    row = pd.read_csv(root / "reports/test_predictions.csv").iloc[0]
    winner = json.loads((root / "artifacts/frozen_selection.json").read_text())[
        "winner"
    ]
    assert predict_emotion(row.path, root) == row[winner]


def test_checkpoint_identity_ignores_zip_timestamps_but_detects_data_changes(tmp_path):
    import zipfile
    from src.experiments import experiment_identity

    for folder in ["src", "data", "artifacts"]:
        (tmp_path / folder).mkdir()
    (tmp_path / "src/audio_features.py").write_text("fixed extractor")
    (tmp_path / "src/rbf_kernel.py").write_text("fixed kernel")
    (tmp_path / "data/split_manifest.csv").write_text("fixed split")
    path = tmp_path / "artifacts/features.npz"
    np.savez_compressed(path, features=np.arange(6).reshape(2, 3))
    original = experiment_identity(tmp_path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            info = zipfile.ZipInfo(name, date_time=(2001, 1, 1, 0, 0, 0))
            archive.writestr(info, content)
    assert experiment_identity(tmp_path) == original
    np.savez_compressed(path, features=np.arange(6).reshape(2, 3) + 1)
    assert experiment_identity(tmp_path) != original


def test_interrupted_save_preserves_last_valid_checkpoint(tmp_path, monkeypatch):
    import joblib
    from src.experiments import save_artifact

    destination = tmp_path / "checkpoint.joblib"
    save_artifact({"completed_candidates": 3}, destination)

    def interrupted_dump(value, path):
        Path(path).write_bytes(b"incomplete")
        raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(joblib, "dump", interrupted_dump)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            save_artifact({"completed_candidates": 4}, destination)
    assert joblib.load(destination) == {"completed_candidates": 3}
    assert not destination.with_name(destination.name + ".partial").exists()


@pytest.mark.parametrize("gamma", ["scale", 0.01])
def test_vectorized_rbf_matches_standard_svc_and_keeps_training_references(gamma):
    from sklearn.datasets import make_classification
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from threadpoolctl import threadpool_limits
    from src.rbf_kernel import TrainingRBFKernel

    features, labels = make_classification(
        n_samples=180,
        n_features=15,
        n_informative=10,
        n_classes=6,
        n_clusters_per_class=1,
        random_state=42,
    )
    ordinary = Pipeline(
        [("scale", StandardScaler()), ("classifier", SVC(C=3, gamma=gamma))]
    )
    vectorized = Pipeline(
        [
            ("scale", StandardScaler()),
            ("kernel", TrainingRBFKernel(gamma=gamma)),
            ("classifier", SVC(C=3, kernel="precomputed")),
        ]
    )
    with threadpool_limits(limits=1):
        ordinary.fit(features[:120], labels[:120])
        vectorized.fit(features[:120], labels[:120])
        reference = vectorized.named_steps["kernel"].reference_features_.copy()
        np.testing.assert_array_equal(
            ordinary.predict(features[120:]), vectorized.predict(features[120:])
        )
        np.testing.assert_allclose(
            ordinary.decision_function(features[120:]),
            vectorized.decision_function(features[120:]),
            atol=1e-3,
            rtol=1e-3,
        )
    assert vectorized.named_steps["kernel"].gamma_ == pytest.approx(
        ordinary.named_steps["classifier"]._gamma
    )
    assert reference.shape == (120, 15)
    np.testing.assert_array_equal(
        reference, vectorized.named_steps["kernel"].reference_features_
    )
