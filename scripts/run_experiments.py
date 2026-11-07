import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import run_selection, run_diagnostics, evaluate_test

if __name__ == "__main__":
    print(run_selection(ROOT), flush=True)
    print(run_diagnostics(ROOT), flush=True)
    print(evaluate_test(ROOT), flush=True)
