# Requirement-to-evidence mapping

The source is the user-provided CEP PDF, Q2. Q1's exact 70/15/15 instruction is not a Q2 requirement; approximately these proportions were chosen as the agreed project design.

| Requirement | Evidence |
|---|---|
| Use the supplied emotion audio dataset | `data/provenance.json`, source inventory and notebook §1 |
| Select files and create train/validation/test before processing | `src/dataset.py`, frozen split manifest; notebook §1 |
| Low-pass filtering | `src/audio_features.py::lowpass`; notebook §2 audio/plots |
| Literature review and feature rationale | `docs/methodology.md`, `docs/references.md`; notebook §3 |
| Fourier, MFCC, Mel, chroma | Feature schema and extraction implementation; notebook §3 |
| Bicoherence magnitude/phase moments | `bicoherence_summary`, estimator description and limitations |
| Centroid, bandwidth and contrast | Feature schema and methodology feature table |
| K-Nearest Neighbors | Search and frozen family winner; notebook §4/7 |
| Logistic Regression | Search and frozen family winner; notebook §4/7 |
| Naive Bayes | GaussianNB search and family winner; notebook §4/7 |
| Support Vector Machines | RBF SVM search and family winner; notebook §4/7 |
| Neural Network | MLP search and family winner; notebook §4/7 |
| Evaluate classifier performance | Validation/test tables, class reports and confusion matrices |
| Project code | Executed notebook, source, scripts, tests and dependency lock |
| Project report | `reports/final_report.md` |
| Group of four students | Student must insert real members/IDs; no identities invented |
| Submit on time | Student action; supplied PDF contains no explicit deadline |
| Additional: prevent speaker/preprocessing leakage | Disjointness assertions and training-only scaler checks |
| Additional: controlled experiments | Ablation and actor learning curve, notebook §5 |
| Additional: uncertainty and error analysis | Bootstrap intervals and validation audio errors, notebook §6/7 |
| Additional: reproducible inference | Saved pipeline, feature schema and consistency test, notebook §8 |

The phrase “synthetic speech attribution” appears in Q2's feature-review paragraph, but the surrounding task, dataset and labels are acted emotion recognition. This project documents that wording mismatch and implements the stated six-emotion classifier; it does not claim to detect synthetic voices.
