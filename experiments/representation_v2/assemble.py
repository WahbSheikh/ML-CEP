"""Core CEP stays primary; measured extensions become clearly marked appendices."""
from pathlib import Path
import base64
import json
import re
import numpy as np
import pandas as pd
import mistune
from .run import ROOT, guard
from .deliver import markdown_table


def assemble(root=ROOT):
    out=root/'experiments/representation_v2'
    guard(root)
    personal_paths=sorted((root/'voice_evaluation/sessions').glob('*/evaluation.json'))
    assert personal_paths, 'Complete real personal evaluation before assembling full documents'
    folder=personal_paths[-1].parent
    personal=json.loads(personal_paths[-1].read_text())
    predictions=pd.read_csv(folder/'predictions.csv')
    metrics=json.loads((out/'comparison.json').read_text())
    validation=pd.read_csv(out/'validation_results.csv')
    egemaps_best=validation[validation.representation.eq('egemaps')].validation_macro_f1.max()
    table=predictions[['filename','intended_emotion','baseline_prediction','improved_prediction']]
    per_class=[]
    for emotion,frame in predictions.groupby('intended_emotion'):
        per_class.append({'emotion':emotion,'recordings':len(frame),
                          'baseline_correct':int(frame.intended_emotion.eq(frame.baseline_prediction).sum()),
                          'improved_correct':int(frame.intended_emotion.eq(frame.improved_prediction).sum())})
    detail=f'''## Optional personal-voice extension: complete evidence

User-confirmed design: **20 actual microphone recordings, five each happy, disgust, sad and angry (internal label anger)**. Session `{folder.name}`. Same sentence, "The airplane is almost full", recorded with the local Intel Microphone Array. Each clip is five seconds; actual device rate is 44,100 Hz, then the shared loader resamples to mono/16 kHz. The UI only records after a user click; audio never uploads. No synthetic test fixture contributes to these twenty samples.

All intended labels were assigned before prediction. Draft retakes were allowed for quality before saving; no evaluated clip was replaced in response to a prediction. Twenty saved WAVs and their SHA-256 hashes were frozen before evaluation. V2 selection was completed independently on CREMA-D; personal recordings never participated in its fitting or selection. Raw voice data remains under the ignored local sessions folder.

{markdown_table(pd.DataFrame([{'model':name,'correct_out_of_20':v['correct'],'accuracy':v['accuracy'],'macro_f1_four_supported':v['macro_f1_four_supported'],'macro_f1_six_fixed':v['macro_f1_six_fixed']} for name,v in personal['models'].items()]))}

**No improvement on the user's own voice:** both models achieved 5/20 correct (25%). The v2 model's four-supported-class macro-F1 fell from 0.1801 to 0.1042. V2 predicted happy on 19 recordings and fear on one. It did not correctly classify any angry, disgust or sad clip. A constant prediction of any one supported class would also achieve 5/20 on this balanced session. This is an important practical failure, despite the modest CREMA-D point gain. Do not present v2 as robust for this microphone/user.

{markdown_table(pd.DataFrame(per_class))}

### Every recorded attempt

{markdown_table(table)}

### Quality checks and limitations

Saved capture peaks ranged from {predictions.peak.min():.4f} to {predictions.peak.max():.4f}; all stored clipped-sample fractions were zero. Thus silence/amplitude clipping checks do not explain away the failure. These checks do not establish clean audio or validate emotion expression. Intended labels are self-reported rather than independent listener ratings. Different microphone/accent, room noise, recording timing/silence and the acted-corpus domain are plausible contributors, not measured causes of each error. No extra trimming, adaptation or retuning was performed after seeing these twenty scores.

One speaker and correlated repeated takes cannot establish multi-speaker robustness. There is no support for fear/neutral targets in this personal test; predictions of those labels still count as errors. Four-supported-class F1 must not be directly compared with a CREMA-D six-supported-class F1. Fixed-six F1 includes absent target classes as zero. Full class reports, six-label confusion matrices and predictions are in the session folder, with `voice_analysis.ipynb` for analysis.
'''
    diagnostics=folder/'diagnostics.md'
    diagnostics.write_text('# Personal voice diagnostics\n\n'+detail,encoding='utf-8')
    report_path=folder/'report.md'
    simple=report_path.read_text(encoding='utf-8').split('\n## Complete evidence')[0]
    report_path.write_text(simple+'\n## Complete evidence\n\n'+detail,encoding='utf-8')
    scope='''# ML-CEP Q2: core report with optional extensions

**Scope and priority.** The original verified notebook, report and artifacts implement the assigned Q2 project. Its main purpose is emotion classification using the specified noisy audio, low-pass filtering, justified audio features, separate train/validation/test sets and the five required classifiers. Exact 70/15/15 percentages belong to Q1, not Q2. The original baseline is the core CEP evidence and remains unchanged.

The eGeMAPS/fusion study and personal microphone experiment below are **additional user-requested extensions**. They are not assignment requirements, do not replace the core evaluation and do not change its original saved winner/results. The core account is reproduced first; optional findings, including failures, follow in appendices.

## Part I: original verified CEP experiment

'''
    baseline=(root/'reports/final_report.md').read_text(encoding='utf-8')
    extension=(out/'comparison_report.md').read_text(encoding='utf-8')
    # Resolve local Markdown links from the extension's folder into reports/.
    def relink(match):
        label,target=match.group(1),match.group(2)
        if '://' in target or target.startswith('#'):
            return match.group(0)
        import os
        dest=(out/target).resolve()
        return '['+label+']('+os.path.relpath(dest,root/'reports').replace('\\','/')+')'
    extension=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',relink,extension)
    appendix=f'''\n\n# Part II: optional extensions requested by the user

These additions are supplemental research and a practical personal test. The assignment's required low-pass/features/classifier evidence remains in Part I and the original executed notebook. The eGeMAPS-only raw-audio branch is diagnostic; it is not substituted for the core filtered pipeline.

## Appendix A: representation/generalization study

{extension}

## Appendix B: negative findings and statistical limits

eGeMAPS-only did **not** beat the core validation result: the best of its five finalists reached macro-F1 {egemaps_best:.4f}, versus baseline 0.4983. Fusion SVM (361 features, C=10, gamma=0.001, no LDA) was selected instead. It reduced the training accuracy from 97.71% to 83.03%, but a sizeable train/validation gap remains. Representation and regularization/search changed together; this is not a causal attribution to any single feature.

Both paired actor-bootstrap 95% macro-F1 difference intervals include zero: validation {metrics['validation_paired_interval']['macro_f1_difference_95']} and reused test {metrics['test_paired_interval']['macro_f1_difference_95']}. Accordingly, report a modest measured point gain, **not conclusive statistical superiority**. There is no new untouched CREMA-D test. No SNR gain, 80-90% accuracy, WavLM experiment or clinical/real-world reliability is claimed.

## Appendix C: personal microphone evaluation

{detail}

## Appendix D: reproducibility, verification and scope boundaries

Original baseline source/artifacts/report/notebook/KT and split manifest were SHA-256 checked against 55 saved file identities. The original nine tests passed; four new synthetic-fixture voice safeguard tests passed separately. These fixtures never counted as real recordings. Critical Ruff checks and dependency consistency passed. Final executed notebook and artifact checks are recorded in the extension folder's verification files. No original data split or baseline source was changed. All extra outputs are isolated in experiments/representation_v2 and voice_evaluation; no commit/push/publication occurred.

Fresh-kernel execution completed seven improvement-notebook code cells and three personal-voice-notebook code cells, with zero error outputs. Three new charts were exported and visually inspected. An initial notebook-execution attempt stalled because the agent sandbox blocked local socket connections; the same notebooks executed successfully with permitted localhost Jupyter sockets. This was an execution-environment issue, not a model result, and no model tuning was done to resolve it.

Initial free RAM was 2.32 GiB, later varied substantially; one extraction worker and one BLAS thread were used. GPU acceleration was not assumed because nvidia-smi was unavailable. CIM inspection was denied, so physical memory was read through the Windows memory API/psutil. Network-restricted package download was retried through the permitted download path. openSMILE 2.6.0 (educational/research license) and sounddevice 0.5.6 (MIT) were added without replacing baseline package pins. WavLM was deferred due to runtime/memory/provenance audit costs. Original method/literature/dataset/student-reference attribution remains in Part I and core docs; new openSMILE/GeMAPS citations and Codex assistance attribution are in PROTOCOL.md.
'''
    v3_report=root/'experiments/feature_selection_v3/report.md'
    if v3_report.exists():
        appendix+='\n\n# Appendix E: optional feature-selection follow-up\n\n'+v3_report.read_text(encoding='utf-8').replace('(PROTOCOL.md)','(../experiments/feature_selection_v3/PROTOCOL.md)')
    review=root/'experiments/reference_review/review.md'
    if review.exists():
        appendix+='\n\n# Appendix F: reference repository review\n\n'+review.read_text(encoding='utf-8')
    combined=root/'reports/CEP_REPORT_WITH_OPTIONAL_EXTENSIONS.md'
    combined.write_text(scope+baseline+appendix,encoding='utf-8')
    guide=(out/'KT_ADDENDUM.md').read_text(encoding='utf-8')
    marker='\n## Scope: asal CEP aur extra kaam alag hain\n'
    guide=guide.split(marker)[0]
    guide+=marker+'''
Asal CEP Q2 ka maqsad assigned audio se emotions classify karna hai: pehle split, phir low-pass filtering aur justified features, phir five required classifiers aur evaluation. Purana verified notebook aur baseline report yeh core evidence hain. Abhi humne jo eGeMAPS/fusion research aur aapki microphone ki 20 takes ki hain woh **aapki request par extra kaam** hai; teacher ki extra mandatory requirement keh kar present nahi karna. Main report mein core pehle, yeh extensions appendices mein hain.

### Har experiment successful nahi tha

eGeMAPS-only ke best finalist ki validation macro-F1 '''+f'{egemaps_best:.4f}'+''' thi, jo baseline 0.4983 se kam hai. Fusion SVM ne chhota point gain diya. Bootstrap intervals zero ko cross karte hain, is liye statistically pakki superiority claim nahi. WavLM implement/train nahi hua. Temporal information ka masla poori tarah solve nahi hua. No clean-reference SNR improvement measure ki gayi.

### Aapki real voice par kya hua?

Aapne khud 20 real recordings ki: har requested emotion ki five takes. Dono models 5/20 yani **25%** correct thay. Naya model 19 clips ko happy aur ek ko fear keh raha tha. Usne sad, disgust aur angry ki koi clip sahi nahi ki. Is liye microphone use par improvement ka claim ghalat hoga. Four-supported-class macro-F1 baseline 0.1801 aur v2 0.1042 hai. Accuracy same hone ka matlab dono equally useful nahi; class-wise predictions bhi dekhni hoti hain.

Constant happy kehne wala model bhi is balanced session mein 25% le leta. Yeh real-use limitation ka saboot hai. Recording peaks/clipping checks theek hona emotion generalization ki guarantee nahi. Mic/accent/room/acting aur silence distribution possible reasons hain; actual causal test nahi kiya, is liye kisi ek ko pakki wajah nahi kehte. Scores dekh kar recordings trim, replace ya model tune nahi kiya. Future adaptation karni ho to in takes ko development data kehna hoga aur alag fresh evaluation recordings collect karni hongi.

'''+markdown_table(table)+'''

Voice data local ignored sessions folder mein hai. Dono models six labels predict karte hain; fear/neutral ki true clips is test mein nahi hain, lekin unki predictions errors hoti hain. Yeh one-speaker personal test hai, multi-speaker validation nahi. Complete per-class tables, metrics aur caveats combined report mein hain.
'''
    if (root/'experiments/feature_selection_v3/KT_ADDENDUM.md').exists():
        guide+='\n'+(root/'experiments/feature_selection_v3/KT_ADDENDUM.md').read_text(encoding='utf-8')
    if review.exists():
        guide+='\n## Reference project ko kaise compare karein\n\n'+review.read_text(encoding='utf-8')
    (out/'KT_ADDENDUM.md').write_text(guide,encoding='utf-8')
    original_kt=(root/'docs/KT_BEGINNER_GUIDE.md').read_text(encoding='utf-8')
    full_kt=root/'docs/KT_FULL_WITH_OPTIONAL_EXTENSIONS.md'
    full_kt.write_text('''# Complete KT: core CEP first, optional extensions afterwards

Original dated baseline guide is reproduced below, followed by a clearly marked optional addendum. Core assignment scope is unchanged. All successful and unsuccessful new findings, including the 20-recording personal test, are included. Original guide files remain untouched. Concise VS Code/PyCharm setup commands are in [SETUP.md](../SETUP.md).

'''+original_kt+'\n\n# Optional user-requested extensions\n\n'+guide,encoding='utf-8')
    extra_html=mistune.create_markdown(plugins=['table'])(guide)
    for path in [out/'validation_comparison.png',out/'winner_confusion.png',folder/'confusion_matrices.png',root/'experiments/feature_selection_v3/feature_selection_curve.png']:
        if path.exists():
            extra_html+='<figure><img style="max-width:100%;height:auto" alt="'+path.stem+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"></figure>'
    baseline_html=(root/'docs/KT_BEGINNER_GUIDE.html').read_text(encoding='utf-8')
    html_page=baseline_html.replace('</body>','<section style="border-top:6px solid #22768a;margin-top:64px;padding-top:32px"><h1>Optional user-requested extensions</h1>'+extra_html+'</section></body>')
    html_page=html_page.replace('<title>','<title>Core CEP + optional extensions | ',1)
    (root/'docs/KT_FULL_WITH_OPTIONAL_EXTENSIONS.html').write_text(html_page,encoding='utf-8')
    (out/'KT_ADDENDUM.html').write_text('<!doctype html><meta charset="utf-8"><title>Optional CEP extensions KT</title><style>body{max-width:960px;margin:48px auto;padding:0 24px;font:18px/1.7 system-ui;color:#18313d}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:8px;border:1px solid #bcced5}pre{overflow:auto;background:#eef4f6;padding:16px}img{max-width:100%}</style>'+extra_html,encoding='utf-8')
    (out/'SCOPE.md').write_text('Core CEP evidence remains in the original notebook/report/KT. Optional extension evidence is supplemental. The combined report is reports/CEP_REPORT_WITH_OPTIONAL_EXTENSIONS.md; combined KT is docs/KT_FULL_WITH_OPTIONAL_EXTENSIONS.html. Original files are untouched. Both successes and failures are reported.\n')
    guard(root)
    print(combined); print(full_kt)


if __name__=='__main__':
    assemble()
