# Setup — Windows

Python **3.12 (64-bit)** install karein. Complete project `D:\ML` mein copy karein, including `data`, `artifacts` aur saved experiment models. Purani `.venv` copy na karein.

## Install

PowerShell mein:

```powershell
cd D:\ML
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments\representation_v2\requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r voice_evaluation\requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name ml-cep --display-name "ML-CEP Python 3.12"
.\.venv\Scripts\python.exe -m pip check
```

## VS Code

1. **Open Folder → `D:\ML`**.
2. Microsoft **Python** aur **Jupyter** extensions install karein.
3. **Python: Select Interpreter** → `D:\ML\.venv\Scripts\python.exe`.
4. Notebook ka **Select Kernel** bhi isi environment par set karein.

## PyCharm

1. **Open → `D:\ML`**.
2. **Settings → Python Interpreter → Add Local Interpreter → Existing environment**.
3. `D:\ML\.venv\Scripts\python.exe` select karein; notebook bhi isi environment se run karein.

## Run

- Core notebook: `notebooks/audio_emotion_analysis.ipynb`
- Optional v2: `experiments/representation_v2/improvement.ipynb`

Notebook mein **Restart Kernel → Run All**. Ek waqt mein ek notebook.

Microphone:

```powershell
.\.venv\Scripts\python.exe -m voice_evaluation.recorder
```

WAV prediction — actual file path dein:

```powershell
.\.venv\Scripts\python.exe scripts/predict_audio.py "C:\path\recording.wav" --model v2
```
