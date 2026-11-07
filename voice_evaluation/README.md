# Personal microphone evaluation

User-confirmed design: **20 real recordings: 5 happy, 5 disgust, 5 sad and 5 angry**. Internal dataset label for angry is `anger`.

Installation aur launch commands: [project SETUP.md](../SETUP.md).

Local desktop window opens; nothing is recorded until you click **Record**. Default Intel microphone is selected when available. Windows microphone permission, if requested, must be handled by you. No network service is used. Sounddevice (MIT license) records through PortAudio; [official documentation](https://python-sounddevice.readthedocs.io/).

## Roman Urdu instructions

1. **Start session** dabayein. Har screen par intended emotion diya hoga. Session mein five shuffled blocks hain; har block mein chaaron emotions ek baar hain.
2. Har baar **The airplane is almost full.** kahein, lekin dikhaye gaye emotion mein. Emotion ka naam mat bolein. Microphone distance aur room ko same rakhein. Five seconds ki recording mein pehle normal sa pause, phir clear sentence kaafi hai.
3. **Record / retake (5 sec)** dabane par microphone 5 seconds active hoga, phir band. **Listen** se playback karein. Sirf recording quality check karein; models ke answers abhi nahi dikhte.
4. Draft theek ho to **Save & next** dabayein. Isse WAV aur intended label locally save hota hai. Retake save se pehle karein. Saved/evaluated attempts ko model result dekh kar replace nahi karna.
5. Tamam **20/20** save hone par **Evaluate all 20 recordings** dabayein. Agar v2 training abhi chal rahi ho to wait karein; recordings mehfooz rehti hain. Koi training aapki voice par nahi hogi.

Resume an interrupted session using the exact folder displayed in the window:

```powershell
.\.venv\Scripts\python.exe -m voice_evaluation.recorder --resume voice_evaluation\sessions\YOUR_SESSION_FOLDER
```

Each session contains WAVs, session.json (intended labels/audio hashes), frozen_recordings.json (recording/model identities before prediction), predictions.csv (both frozen models), evaluation.json and report.md. All sessions and voice data are ignored by Git. No upload, commit or push occurs.

After evaluation, build the executed personal analysis notebook and combined core-plus-optional documents:

```powershell
.\.venv\Scripts\python.exe -m voice_evaluation.analyze voice_evaluation\sessions\YOUR_SESSION_FOLDER
.\.venv\Scripts\python.exe -m experiments.representation_v2.assemble
```

The personal microphone feature is an optional user-requested addition. The core CEP requirements and original verified experiment remain primary and unchanged. The completed first real session has twenty recordings and both models obtained 5/20 (25%); the v2 model predicted happy 19 times. These negative results are preserved, not used for retuning.

Models still predict six emotions. Fear/neutral predictions on these four requested emotions count as errors. Four-supported-class macro-F1 averages happy/disgust/sad/anger. Fixed-six-class macro-F1 also includes the two unsupported classes as zero, and is provided separately to avoid confusing it with a six-emotion test with support for every class.

Twenty repetitions from one speaker/session are correlated; this is descriptive personal performance, not a 20-speaker test or evidence of broad population accuracy. Labels describe what you intended to act, not independent listener annotations. Personal results must remain separate from CREMA-D validation/reused-test results. All twenty samples must be collected before predictions are exposed, and no tuning uses these samples.

Implementation: session.py manages labels, saved WAVs, immutable evaluation and metrics; recorder.py provides the interactive desktop window. Only trusted local saved joblib models are loaded. Synthetic test fixtures are never counted as real user recordings.
