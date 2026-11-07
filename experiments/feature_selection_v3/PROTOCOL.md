# Optional v3 diagnostic: do redundant features / regularization hurt?

Written before viewing v3 validation results. The assigned CEP remains the original frozen experiment; this is a further user-requested extension after the disappointing personal-voice result.

Current v2: training accuracy 83.03%, validation 51.40%, reused CREMA-D test 57.72%; personal-voice test 5/20. These known results motivate a hypothesis, not a proven cause. Do not use the twenty personal labels or test predictions to choose features, hyperparameters or a winner.

Use the exact existing 361-feature fusion, original actor allocation and training-only StratifiedGroupKFold(3, shuffle=True, random_state=42). In each fold fit StandardScaler, SelectKBest(f_classif), then the classifier. ANOVA ranks individual features using only fit-fold labels. It can discard useful interacting features, so benefit is empirical, not guaranteed. Compare k=32,64,128,256,361 (all features control). Fit scaling/ranking independently inside each fold.

Two diagnostic families (the core CEP and v2 still retain all five required classifiers):

- Logistic Regression C=0.001,0.01,0.1 crossed with all five k values (15).
- RBF SVM C=1,10 crossed with gamma=scale,0.001 and all five k values (20). The tested vectorized training-reference kernel is reused.

Total 35 configurations / 105 internal fits. Mean grouped training CV macro-F1 selects one configuration per family, candidate order breaks ties. Only these two finalists are refitted on original training clips and evaluated on the existing validation actors. Compare against the frozen v2 validation macro-F1 0.5090603328. Save source/data identities, warnings, selected feature names and validation predictions. No automatic test/personal evaluation is part of this diagnostic. No train+validation refitting. Reused validation remains development evidence and is not an unbiased estimate.

Correlations/feature rankings are descriptive. Strongly correlated columns and single-feature F scores cannot prove the cause of overfitting, identify a clinically reliable emotion cue or establish microphone-domain generalization. If validation does not improve, retain v2 as the measured extension winner and report the negative finding. No retuning after viewing validation.

One BLAS thread, sequential fits; current free RAM ~0.56 GiB, so no new large encoder/deep-runtime installation. No application closing, paid service, commit or push. openSMILE/GeMAPS/scikit-learn attribution remains in the v2 protocol. SelectKBest/f_classif is the installed scikit-learn implementation, not a new algorithm. Code assisted by OpenAI Codex.

Run from D:\ML: `.\.venv\Scripts\python.exe -u -m experiments.feature_selection_v3.run`.
