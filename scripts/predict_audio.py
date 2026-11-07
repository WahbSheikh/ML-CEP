"""Small CLI demo using the existing baseline or optional v2 inference function."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[name] = '1'
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path, help='Path to an existing WAV recording')
    parser.add_argument('--model', choices=['baseline', 'v2'], default='baseline')
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f'Audio file not found: {args.audio}')
    if args.model == 'baseline':
        from src.inference import predict_emotion
    else:
        from experiments.representation_v2.features import predict_emotion
    print(predict_emotion(args.audio.resolve(), project_root=ROOT))


if __name__ == '__main__':
    main()
