"""Read-only teammate setup checks. No download, fitting or microphone recording."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2', action='store_true', help='Check optional v2 dependencies and inference bundle')
    parser.add_argument('--voice', action='store_true', help='Check sounddevice/Tk availability without recording')
    parser.add_argument('--full-data', action='store_true', help='Check manifest paths for full notebook reproduction')
    parser.add_argument('--verify-frozen', action='store_true', help='Verify all original baseline file hashes')
    args = parser.parse_args()
    errors, warnings = [], []
    print('Project:', ROOT)
    print('Interpreter:', sys.executable)
    print('Python:', sys.version.split()[0])
    if sys.version_info[:2] != (3, 12):
        errors.append('Use Python 3.12 for the recorded environment.')
    if struct.calcsize('P') * 8 != 64:
        errors.append('64-bit Python is required for this verified Windows setup/openSMILE.')
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        warnings.append('Interpreter is not this project .venv. Confirm the IDE interpreter AND notebook kernel.')
    if sys.platform != 'win32':
        warnings.append('This lock and setup guide were verified on Windows; other OSes need dependency adaptation.')
    lock = ROOT / ('experiments/representation_v2/requirements-lock.txt' if args.v2 else 'requirements-lock.txt')
    if lock.exists():
        checked = 0
        for line in lock.read_text(encoding='utf-8-sig').splitlines():
            if '==' not in line or line.lstrip().startswith('#'):
                continue
            name, version = line.strip().split('==', 1)
            checked += 1
            try:
                actual = importlib.metadata.version(name)
                if actual != version:
                    errors.append(f'Package {name}: installed {actual}, expected {version}.')
            except importlib.metadata.PackageNotFoundError:
                errors.append(f'Missing package: {name}=={version}')
        print('Pinned packages checked:', checked)
    else:
        errors.append(f'Missing dependency lock: {lock.relative_to(ROOT)}')
    required = ['src/audio_features.py', 'src/rbf_kernel.py', 'src/inference.py',
                'artifacts/winning_pipeline.joblib', 'artifacts/features_metadata.json',
                'artifacts/feature_schema.json']
    if args.v2:
        required += ['src/experiments.py', 'data/split_manifest.csv', 'artifacts/features.npz',
                     'experiments/representation_v2/features.py', 'experiments/representation_v2/run.py',
                     'experiments/representation_v2/features.npz',
                     'experiments/representation_v2/features_metadata.json',
                     'experiments/representation_v2/selection.json',
                     'experiments/representation_v2/winner.joblib']
    missing = [name for name in required if not (ROOT / name).is_file()]
    errors.extend('Missing handoff file: ' + name for name in missing)
    metadata_path = ROOT / 'artifacts/features_metadata.json'
    if metadata_path.exists() and (ROOT / 'src/audio_features.py').exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['extractor_sha256'] != sha(ROOT / 'src/audio_features.py'):
            errors.append('Baseline extractor source bytes differ from the saved model metadata.')
    if args.v2 and not missing:
        # Reuse the actual fingerprint function; it hashes inputs only and never fits.
        sys.path.insert(0, str(ROOT))
        from experiments.representation_v2.run import identity
        frozen = json.loads((ROOT / 'experiments/representation_v2/selection.json').read_text())
        if frozen['identity'] != identity(ROOT):
            errors.append('V2 input/source fingerprint mismatch. Do not bypass or edit saved identities.')
        else:
            print('V2 input/source fingerprint: matches')
    if args.full_data:
        path = ROOT / 'data/split_manifest.csv'
        if not path.exists():
            errors.append('Missing frozen split manifest for full notebook execution.')
        else:
            with path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            print('Manifest clips:', len(rows))
            missing_audio, different_root = [], []
            for row in rows:
                original = Path(row['path'])
                expected = ROOT / 'data/raw/NoiseAudioWAV' / row['filename']
                if original.resolve() != expected.resolve():
                    different_root.append(row['filename'])
                if not expected.is_file():
                    missing_audio.append(row['filename'])
            if different_root:
                errors.append(f'{len(different_root)} manifest paths point to a different project root. Inference can still use a new WAV; full reproduction needs the original root or a clean new experiment workspace.')
            if missing_audio:
                errors.append(f'Missing {len(missing_audio)} retained audio files under this project root.')
            if len(rows) != 7435:
                errors.append('Expected the original 7,435 retained clips in this snapshot.')
    if args.verify_frozen:
        path = ROOT / 'experiments/representation_v2/baseline_hashes.json'
        if not path.exists():
            errors.append('Missing baseline_hashes.json for the full frozen snapshot.')
        else:
            frozen = json.loads(path.read_text())
            failed = [name for name, digest in frozen.items()
                      if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
            errors.extend('Frozen baseline mismatch/missing: ' + name for name in failed)
            if not failed:
                print('Original baseline files verified:', len(frozen))
    if args.voice:
        try:
            version = importlib.metadata.version('sounddevice')
            if version != '0.5.6':
                errors.append(f'sounddevice {version} installed; expected 0.5.6.')
        except importlib.metadata.PackageNotFoundError:
            errors.append('Missing sounddevice==0.5.6; install voice_evaluation/requirements.txt.')
        try:
            import tkinter
            print('Tcl:', tkinter.Tcl().eval('info patchlevel'))
        except Exception as exc:
            errors.append('Tkinter unavailable: ' + str(exc))
        print('No microphone was opened. Use recorder --check to list devices separately.')
    for warning in warnings:
        print('NOTE:', warning)
    for error in errors:
        print('ERROR:', error)
    if errors:
        print(f'Setup needs attention: {len(errors)} issue(s).')
        return 1
    print('PASS: requested setup checks passed. No training or recording was performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
