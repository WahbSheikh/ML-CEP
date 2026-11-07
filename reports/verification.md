# Verification record

- Fresh-kernel notebook execution: 14/14 code cells completed; zero error outputs.
- Notebook format validated: nbformat 4.
- Embedded outputs: 6 scientific figures and 4 playable audio examples.
- Full candidate search: 76 configurations across all five required classifier families.
- Dataset: 7,442 audited; 7,435 retained; 7 exclusions documented.
- Speaker split: 63/14/14 actors; no speaker or exact-audio overlap.
- Automated verification: python -m pytest -q — 9 passed, no skips.
- Checks include filtering, bicoherence, finite features, training-only scaling/kernel references, inference consistency, stable cache identity, atomic checkpoint recovery and RBF-SVC equivalence.
- Dependency consistency: python -m pip check passed.
- PyCharm notebook backend: Jupyter Notebook 7.4.3 verified.
- Critical static checks: Ruff E9/F63/F7/F82 passed.
- All six exported figures visually reviewed; local Markdown links checked.
- Final report generated from saved experiment outputs.
- No Git commit or push performed.
