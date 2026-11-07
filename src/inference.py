"""Load only trusted local model artifacts; joblib uses Python pickle."""

from pathlib import Path
import json
import hashlib
import joblib
import numpy as np
from .audio_features import extract_file


def predict_emotion(audio_path, project_root=None):
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    metadata = json.loads((root / "artifacts/features_metadata.json").read_text())
    if (
        metadata["extractor_sha256"]
        != hashlib.sha256((root / "src/audio_features.py").read_bytes()).hexdigest()
    ):
        raise ValueError("Extractor changed since model training")
    features, names = extract_file(audio_path, filtered=True)
    expected = json.loads((root / "artifacts/feature_schema.json").read_text())
    if names != expected:
        raise ValueError("Feature schema does not match the saved model")
    model = joblib.load(root / "artifacts/winning_pipeline.joblib")
    return str(model.predict(features.reshape(1, -1))[0])
