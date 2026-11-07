"""Resumable per-file extraction, with bounded worker and BLAS counts."""

import os

for variable in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
]:
    os.environ[variable] = "1"
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import hashlib
from dataclasses import asdict
import numpy as np
from joblib import Parallel, delayed
from src.dataset import prepare_dataset
from src.audio_features import SETTINGS, cache_key, extract_file


def process(row):
    target = ROOT / "data/cache" / (cache_key(row.sha256) + ".npz")
    if not target.exists():
        filtered, names = extract_file(row.path, True)
        unfiltered, other_names = extract_file(row.path, False)
        assert names == other_names
        temporary = target.with_suffix(".partial.npz")
        np.savez_compressed(
            temporary, filtered=filtered, unfiltered=unfiltered, names=np.array(names)
        )
        temporary.replace(target)
    return str(target)


def main():
    (ROOT / "data/cache").mkdir(parents=True, exist_ok=True)
    manifest, audit = prepare_dataset(ROOT)
    print("Audit:", audit.status.value_counts().to_dict(), flush=True)
    paths = Parallel(n_jobs=4, verbose=10)(
        delayed(process)(row) for row in manifest.itertuples()
    )
    filtered_rows, unfiltered_rows = [], []
    names = None
    for path in paths:
        with np.load(path) as cached:
            if names is None:
                names = cached["names"]
            assert np.array_equal(names, cached["names"])
            filtered_rows.append(cached["filtered"])
            unfiltered_rows.append(cached["unfiltered"])
    filtered = np.vstack(filtered_rows)
    unfiltered = np.vstack(unfiltered_rows)
    assert np.isfinite(filtered).all() and np.isfinite(unfiltered).all()
    temporary = ROOT / "artifacts/features.partial.npz"
    np.savez_compressed(
        temporary,
        filtered=filtered,
        unfiltered=unfiltered,
        names=names,
        filenames=manifest.filename.to_numpy(dtype=str),
    )
    temporary.replace(ROOT / "artifacts/features.npz")
    (ROOT / "artifacts/feature_schema.json").write_text(
        json.dumps(names.tolist(), indent=2)
    )
    metadata = {
        "settings": asdict(SETTINGS),
        "extractor_sha256": hashlib.sha256(
            (ROOT / "src/audio_features.py").read_bytes()
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (ROOT / "data/split_manifest.csv").read_bytes()
        ).hexdigest(),
    }
    (ROOT / "artifacts/features_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print("Feature matrices:", filtered.shape, unfiltered.shape, flush=True)


if __name__ == "__main__":
    main()
