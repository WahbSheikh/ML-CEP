"""Audit raw audio and freeze speaker-disjoint splits before extraction."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from .audio_features import EMOTIONS


def inspect_file(path):
    """Decode one recording and retain both container and PCM identities."""
    row = {"path": str(path), "filename": path.name, "status": "valid", "issue": ""}
    try:
        actor, sentence, emotion, intensity = path.stem.split("_")
        if not actor.isdigit() or emotion not in EMOTIONS:
            raise ValueError("Invalid actor/emotion code")
        samples, rate = sf.read(path, always_2d=True)
        if (
            len(samples) / rate < 0.04
            or not np.isfinite(samples).all()
            or np.max(abs(samples)) < 1e-9
        ):
            raise ValueError("Short, empty, silent or non-finite audio")
        row.update(
            actor=int(actor),
            sentence=sentence,
            emotion=EMOTIONS[emotion],
            intensity=intensity,
            sample_rate=rate,
            channels=samples.shape[1],
            duration=len(samples) / rate,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pcm_sha256=hashlib.sha256(
                np.asarray(samples, dtype="<f8").tobytes() + str(rate).encode()
            ).hexdigest(),
        )
    except Exception as error:
        row.update(status="excluded", issue=str(error))
    return row


def prepare_dataset(root):
    """Audit all selected clips and persist their fixed actor assignments."""
    report_dir = root / "reports"
    paths = sorted((root / "data/raw/NoiseAudioWAV").glob("*.wav"))
    if len(paths) != 7442:
        raise ValueError(
            f"Expected 7442 WAV files, found {len(paths)}. Run scripts/download_audio.py."
        )
    with ThreadPoolExecutor(max_workers=4) as pool:
        audit = pd.DataFrame(pool.map(inspect_file, paths))
    valid_rows = audit[audit.status.eq("valid")]
    for _, duplicates in valid_rows.groupby("pcm_sha256"):
        if len(duplicates) < 2:
            continue
        if duplicates.emotion.nunique() > 1:
            indices = duplicates.index
            reason = "Identical decoded audio has conflicting emotion labels; all members excluded"
        else:
            indices = duplicates.index[1:]
            reason = "Same-label decoded duplicate; first lexicographic file retained"
        audit.loc[indices, "status"] = "excluded"
        audit.loc[indices, "issue"] = reason
    audit.to_csv(report_dir / "data_audit.csv", index=False)
    usable = audit[audit.status.eq("valid")].copy()
    usable["actor"] = usable.actor.astype(int)
    if usable.pcm_sha256.duplicated().any():
        raise ValueError(
            "Duplicate decoded audio detected; resolve before freezing splits"
        )
    actors = np.sort(usable.actor.unique())
    if len(actors) != 91:
        raise ValueError(f"Expected 91 actors, found {len(actors)}")
    actors = np.random.default_rng(42).permutation(actors)
    groups = {
        int(actor): split
        for split, part in [
            ("train", actors[:63]),
            ("validation", actors[63:77]),
            ("test", actors[77:]),
        ]
        for actor in part
    }
    usable["split"] = usable.actor.map(groups)
    assert usable.groupby("actor").split.nunique().max() == 1
    for split, frame in usable.groupby("split"):
        assert set(frame.emotion) == set(EMOTIONS.values()), split
    manifest_path = root / "data/split_manifest.csv"
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        assert old[["filename", "sha256", "split"]].equals(
            usable[["filename", "sha256", "split"]].reset_index(drop=True)
        ), "Frozen manifest changed"
    usable.to_csv(manifest_path, index=False)
    return usable.reset_index(drop=True), audit
