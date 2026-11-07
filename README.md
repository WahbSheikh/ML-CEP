# ML-CEP — hearing emotion in unseen speakers

Part 2 of the CEP: six-class acoustic emotion recognition on the assignment's noisy CREMA-D recordings. The project uses independently written processing and experiment code, with citations and explicit coding-assistant attribution. No commit or push is part of any script.

Start with the [executed notebook](notebooks/audio_emotion_analysis.ipynb), [final report](reports/final_report.md), [verification record](reports/verification.md), and [viva notes](docs/viva_notes.md).

New to ML/audio? Read the [complete Roman Urdu beginner KT handbook](docs/KT_BEGINNER_GUIDE.html), or its [Markdown source](docs/KT_BEGINNER_GUIDE.md). It explains the concepts, all five algorithms, actual code/results, operations and handoff in 30 chapters. The HTML contains all six charts for offline reading. Rebuild its presentation/schema appendix with `scripts/build_kt_html.py`; its dated narrative must be reviewed after a different experiment.

## Start in PyCharm

Open `D:\ML` as a project in PyCharm. Select `D:\ML\.venv\Scripts\python.exe` as the existing Python interpreter, then open `notebooks/audio_emotion_analysis.ipynb` and use that environment for its notebook kernel. The environment includes Jupyter Notebook for local server support. Notebook support is assumed available in your PyCharm installation, as you indicated. The standard `.ipynb` also works in VS Code or Jupyter. The notebook is the main analysis deliverable; `reports/final_report.md` is the submission-style write-up. Add your real group details according to the course requirements.

```powershell
# Python 3.12 must be installed; uv may create an equivalent environment.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/download_audio.py
.\.venv\Scripts\python.exe scripts/extract_features.py
.\.venv\Scripts\python.exe scripts/build_notebook.py
.\.venv\Scripts\python.exe scripts/execute_notebook.py
.\.venv\Scripts\python.exe scripts/write_report.py
.\.venv\Scripts\python.exe -m pytest -q
```

`requirements.txt` contains direct pins; `requirements-lock.txt` records the full verified environment. Downloads use one immutable assignment dataset commit, 32 workers and blob-integrity checks. About 606 MB of WAVs are required, plus environments/caches. No GitHub authentication is required to download public data. If rate-limited, wait and rerun; verified existing files are reused.

The notebook runs the model search, diagnostics and test evaluation itself. For a headless alternative that produces the same tables without rendering notebook outputs, run `scripts/run_experiments.py` after feature extraction. Do not launch both training routes concurrently.

## Reproducibility and boundaries

- NumPy seed 42 freezes 63 training, 14 validation and 14 test actors before extraction. Clip proportions are approximate; all six classes must exist in every split.
- Train-only StandardScaler and optional LDA prevent preprocessing leakage. All 76 search candidates use validation macro-F1; test scores never select the winner.
- Resume individual feature caches and candidate checkpoints. Source hashes and implementation fingerprints reject stale reuse. Do not edit frozen data/settings mid-experiment; archive the complete existing artifacts/reports before starting a genuinely new experiment.
- The final test stage reuses already frozen results on subsequent runs. The notebook still executes every analysis cell in a fresh kernel; expensive validated caches are reused rather than silently training on test data.
- Run time depends on CPU and storage. Four feature workers and one BLAS thread per process keep resource use bounded. MLP iteration warnings are recorded, not discarded.

## Project map

| Location | Purpose |
|---|---|
| `notebooks/audio_emotion_analysis.ipynb` | Executed analysis, plots, playback, evaluation and inference |
| `src/audio_features.py`, `src/dataset.py` | Signal processing, feature definitions, integrity and speaker splits |
| `src/experiments.py`, `src/inference.py` | Validation selection, diagnostics, test stage and reusable prediction |
| `scripts/` | Download, feature extraction, experiments, notebook execution and report generation |
| `data/provenance.json`, `data/split_manifest.csv` | Immutable source provenance and exact data allocation |
| `artifacts/` | Feature schema, local matrices, frozen selection and trusted saved models |
| `reports/` | Actual experiment tables, predictions, figures, report and log |
| `docs/` | Methodology, literature references, requirement mapping and viva notes |
| `tests/` | Scientific signal checks and real-artifact leakage/inference checks |

## Offline prediction

```python
from src.inference import predict_emotion
emotion = predict_emotion('path/to/recording.wav')
print(emotion)
```

Only load your own trusted local joblib artifacts. The prediction is one of six intended emotion labels; no calibrated confidence score is claimed.

## Submission

Include the executed notebook, source/scripts, dependency files, Markdown report/docs, figures and small provenance/results files. Raw recordings and `.venv` stay local; retain upstream licensing and attribution when redistributing data. Large model/cache artifacts are ignored by Git, but keep them locally for inference and reproducibility. Any future upload requires the user's explicit authorization.

The original PDF also asks for a group of four and timely submission. These are administrative requirements: the code cannot create real group members or determine a deadline absent from the supplied document. See the checklist for academic and technical evidence.
