# Experiment log

- Dataset immutable commit: `8c764b0b778f90c8b60484afe21715c899d35663`; verified WAVs: 7442.
- Split: seed 42, 63/14/14 actors; manifest frozen before extraction.
- Features: 273 columns, filtered and unfiltered counterparts.
- Search: 76 candidates; training-only fit; selection metric validation macro-F1.
- Winner frozen before test: **SVM**.
- Diagnostics: four fixed-SVM ablations, four nested training-actor sizes.
- Final evaluation: five family winners and majority baseline; no validation refit.
- Confidence intervals: 1,000 test-actor bootstrap resamples.
- Full candidate settings, scores, warning text and elapsed seconds: [search_results.csv](search_results.csv).
- Selection fingerprint and configuration: `artifacts/frozen_selection.json`.
- No commits or pushes performed by the implementation workflow.
