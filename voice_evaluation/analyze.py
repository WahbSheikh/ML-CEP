"""Execute a notebook from completed real voice results, without retraining."""
from pathlib import Path
import os
import argparse
import json
import nbformat as nbf
from .session import ROOT, read_session


def analyze(folder):
    folder=Path(folder).resolve()
    assert len(read_session(folder)['clips'])==20
    assert (folder/'evaluation.json').exists(), 'Evaluate real recordings first'
    cells=[nbf.v4.new_markdown_cell('''# Meri voice - 20 real recordings

5 happy, 5 disgust, 5 sad, 5 angry. One speaker/session. Intended emotions are user-provided, not independently rated perceived emotions. No training/tuning uses these recordings. Models still predict all six labels. Personal results remain separate from CREMA-D results.'''),
    nbf.v4.new_code_cell(f'''from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import ConfusionMatrixDisplay
FOLDER=Path({str(folder)!r})
results=json.loads((FOLDER/'evaluation.json').read_text())
predictions=pd.read_csv(FOLDER/'predictions.csv')
display(predictions[['filename','intended_emotion','baseline_prediction','improved_prediction']])
display(pd.DataFrame({{name:{{key:value for key,value in metrics.items() if key in ['correct','accuracy','macro_f1_four_supported','macro_f1_six_fixed']}} for name,metrics in results['models'].items()}}).T)'''),
    nbf.v4.new_code_cell('''fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,(name,metrics) in zip(axes,results['models'].items()):
    ConfusionMatrixDisplay(np.asarray(metrics['confusion_matrix_six']),display_labels=metrics['confusion_label_order']).plot(ax=ax,colorbar=False,cmap='Blues')
    ax.set_title(name+' - personal voice'); ax.tick_params(axis='x',rotation=35)
fig.tight_layout(); fig.savefig(FOLDER/'confusion_matrices.png',dpi=140); plt.show()'''),
    nbf.v4.new_markdown_cell('''Four-supported-class macro-F1 averages only the four requested emotions. Fixed-six-class macro-F1 additionally includes zero support for fear/neutral. Fear/neutral predictions still count as errors. Twenty takes from one speaker are correlated, so these results cannot establish population generalization. Models and recordings were frozen before prediction; do not replace failed takes after seeing answers.'''),
    nbf.v4.new_code_cell('''for name,metrics in results['models'].items():
    print(name)
    display(pd.DataFrame(metrics['classification_report']).T)
print(results['limits'])''')]
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'ML-CEP (.venv Python 3.12)','name':'ml-cep','language':'python'}})
    path=folder/'voice_analysis.ipynb'
    nbf.write(notebook,path)
    from experiments.representation_v2.execute import execute
    execute(path)
    print(path)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('session',type=Path); args=parser.parse_args(); analyze(args.session)
