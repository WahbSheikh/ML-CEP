"""Deterministic signal processing shared by training and inference."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import warnings
import librosa
import numpy as np
import soundfile as sf
from scipy import signal, stats


@dataclass(frozen=True)
class AudioSettings:
    sample_rate: int = 16000
    cutoff_hz: int = 5000
    filter_order: int = 4
    frame_length: int = 512
    hop_length: int = 160
    mfcc_count: int = 20
    mel_count: int = 40
    version: str = "original-v1"


SETTINGS = AudioSettings()
EMOTIONS = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def load_audio(path, settings=SETTINGS):
    """Decode to mono at the target rate while preserving signal amplitude."""
    samples, original_rate = sf.read(path, dtype="float64", always_2d=True)
    if samples.shape[0] == 0 or not np.isfinite(samples).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    samples = samples.mean(axis=1)
    if original_rate != settings.sample_rate:
        samples = librosa.resample(
            samples, orig_sr=original_rate, target_sr=settings.sample_rate
        )
    if len(samples) < settings.frame_length or np.max(np.abs(samples)) < 1e-9:
        raise ValueError(f"Too short or silent audio: {path}")
    return samples


def lowpass(samples, settings=SETTINGS):
    """Apply the same stable, zero-phase low-pass in training and inference."""
    sections = signal.butter(
        settings.filter_order, settings.cutoff_hz, fs=settings.sample_rate, output="sos"
    )
    return signal.sosfiltfilt(sections, samples)


def bicoherence_summary(samples, settings=SETTINGS):
    """Segment-averaged normalized bispectrum, not a raw FFT triple product.

    Use Hann-windowed 512-sample, 50%-overlap frames; coarse FFT indices
    2,4,...,64 (62.5..2000 Hz), f1<=f2, f1+f2<=4000 Hz.
    Phase moments are descriptive wrapped-angle moments, not circular moments.
    """
    frames = librosa.util.frame(samples, frame_length=512, hop_length=256).T
    frames = frames - frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(frames * np.hanning(512), axis=1)
    first, second = np.triu_indices(32)
    bins = np.arange(2, 65, 2)
    product = spectrum[:, bins[first]] * spectrum[:, bins[second]]
    summed = spectrum[:, bins[first] + bins[second]]
    numerator = np.mean(product * summed.conj(), axis=0)
    denominator = np.sqrt(
        np.mean(abs(product) ** 2, axis=0) * np.mean(abs(summed) ** 2, axis=0)
    )
    valid = denominator > max(denominator.max() * 1e-12, 1e-30)
    normalized = numerator[valid] / denominator[valid]
    values = []
    names = []
    for label, array in [
        ("magnitude", np.clip(abs(normalized), 0, 1)),
        ("phase", np.angle(normalized)),
    ]:
        if len(array) == 0:
            raise ValueError("No valid bicoherence frequency pairs")
        spread = np.std(array)
        moments = [
            array.mean(),
            array.var(),
            stats.skew(array) if spread > 1e-12 else 0,
            stats.kurtosis(array, fisher=True) if spread > 1e-12 else 0,
        ]
        values.extend(moments)
        names.extend(
            f"bicoherence_{label}_{stat}"
            for stat in ["mean", "variance", "skewness", "excess_kurtosis"]
        )
    return values, names


def extract_features(samples, settings=SETTINGS):
    """Summarize each acoustic family into one finite, ordered clip vector."""
    values, names = [], []

    def summarize(group, array):
        matrix = np.atleast_2d(array)
        for statistic, summary in [
            ("mean", matrix.mean(axis=1)),
            ("std", matrix.std(axis=1)),
        ]:
            values.extend(summary.tolist())
            names.extend(f"{group}_{i:02d}_{statistic}" for i in range(len(summary)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        transform = librosa.stft(
            samples, n_fft=settings.frame_length, hop_length=settings.hop_length
        )
        magnitude = abs(transform)
        power = magnitude**2
        mel = librosa.feature.melspectrogram(
            S=power, sr=settings.sample_rate, n_mels=settings.mel_count
        )
        logmel = librosa.power_to_db(mel, ref=1.0)
        mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=settings.mfcc_count)
        summarize("mfcc", mfcc)
        summarize("mfcc_delta", librosa.feature.delta(mfcc, mode="nearest"))
        summarize("mfcc_delta2", librosa.feature.delta(mfcc, order=2, mode="nearest"))
        summarize("mel", logmel)
        summarize(
            "chroma",
            librosa.feature.chroma_stft(
                S=power, sr=settings.sample_rate, n_chroma=12, tuning=0
            ),
        )
        summarize(
            "spectral_centroid",
            librosa.feature.spectral_centroid(S=magnitude, sr=settings.sample_rate),
        )
        summarize(
            "spectral_bandwidth",
            librosa.feature.spectral_bandwidth(S=magnitude, sr=settings.sample_rate),
        )
        summarize(
            "spectral_contrast",
            librosa.feature.spectral_contrast(
                S=magnitude, sr=settings.sample_rate, n_bands=5
            ),
        )
        summarize(
            "spectral_rolloff",
            librosa.feature.spectral_rolloff(S=magnitude, sr=settings.sample_rate),
        )
        summarize("spectral_flatness", librosa.feature.spectral_flatness(S=magnitude))
        summarize(
            "rms", librosa.feature.rms(S=magnitude, frame_length=settings.frame_length)
        )
        summarize(
            "zcr",
            librosa.feature.zero_crossing_rate(
                samples,
                frame_length=settings.frame_length,
                hop_length=settings.hop_length,
            ),
        )
        frequencies = librosa.fft_frequencies(
            sr=settings.sample_rate, n_fft=settings.frame_length
        )
        total = np.maximum(power.sum(axis=0), 1e-20)
        for start, end in [
            (0, 250),
            (250, 500),
            (500, 1000),
            (1000, 2000),
            (2000, 4000),
            (4000, 6000),
            (6000, 8001),
        ]:
            summarize(
                f"fourier_{start}_{end}",
                power[(frequencies >= start) & (frequencies < end)].sum(axis=0) / total,
            )
        pitches, strengths = librosa.piptrack(
            S=magnitude, sr=settings.sample_rate, fmin=60, fmax=500
        )
        strongest = strengths.argmax(axis=0)
        pitch = pitches[strongest, np.arange(pitches.shape[1])]
        voiced = pitch[pitch > 0]
        summarize("pitch", voiced if len(voiced) else np.zeros(1))
        values.append(float(np.mean(pitch > 0)))
        names.append("pitch_voiced_fraction")
        higher_order, higher_names = bicoherence_summary(samples, settings)
        values.extend(higher_order)
        names.extend(higher_names)
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite extracted feature")
    return result, names


def extract_file(path, filtered=True, settings=SETTINGS):
    samples = load_audio(path, settings)
    return extract_features(
        lowpass(samples, settings) if filtered else samples, settings
    )


def cache_key(file_digest, settings=SETTINGS):
    payload = {
        "source_sha256": file_digest,
        "settings": asdict(settings),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
