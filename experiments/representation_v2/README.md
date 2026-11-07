# Representation v2

Start with [comparison_report.md](comparison_report.md), [improvement.ipynb](improvement.ipynb), or [Roman Urdu KT addendum](KT_ADDENDUM.html).

Installation aur IDE setup: [project SETUP.md](../../SETUP.md).

All outputs are isolated here. Baseline remains frozen. Protocol, rationale, citations, license and exact reproduction commands: [PROTOCOL.md](PROTOCOL.md).

Final combined deliverables preserve the assigned CEP work first and put optional user-requested research/personal-voice findings in appendices: [core report plus optional extensions](../../reports/CEP_REPORT_WITH_OPTIONAL_EXTENSIONS.md), [full offline KT](../../docs/KT_FULL_WITH_OPTIONAL_EXTENSIONS.html). Run `python -m experiments.representation_v2.assemble` after a completed personal evaluation/analysis to regenerate these combined documents. Successful and unsuccessful findings are both included.

```python
from experiments.representation_v2.features import predict_emotion
print(predict_emotion('path/to/recording.wav'))
```

Run commands from D:\ML using the existing .venv. The executed notebook performs resumable extraction/search, cached test reuse, comparison, plots and WAV inference. Do not launch it concurrently with run.py. Large .npz/.joblib files are local and ignored by Git. Retain these for offline inference, or regenerate from pinned requirements and the verified data. verification.json records checks, comparison.json holds exact gains and paired actor intervals. Original baseline notebook/report/KT remain the assignment's baseline deliverable; v2 supplements them. No commit or push occurred.
