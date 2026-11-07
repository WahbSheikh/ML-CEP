# Optional feature-selection/regularization diagnostic (v3)

This is supplemental investigation requested after the personal-voice failure; it does not replace the core CEP or change its required five-model evaluation.

Hypothesis: redundant/noisy feature columns and insufficient regularization might contribute to poor unseen-speaker performance. Training-only audit found **794 pairs with absolute correlation >0.95**, and zero constant columns. Correlation alone does not establish the cause of the generalization gap.

ANOVA SelectKBest inside each training fold compared 32,64,128,256 and 361 features. Thirty-five configurations (Logistic Regression and SVM), each on three actor-grouped folds, made 105 internal fits. Scaling, ANOVA scores and classifiers only saw fit-fold actors. C values included stronger regularization. Hyperparameters/feature count were selected from grouped training CV; just two finalists then used the reused validation set. No new test or personal-voice evaluation was performed.

## Measured outcome

| model | k | parameters | cv_macro_f1 | train_accuracy | validation_accuracy | validation_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 361.0000 | {"C": 0.01} | 0.5478 | 0.6442 | 0.4991 | 0.4930 |
| SVM | 361.0000 | {"C": 10, "gamma": 0.001} | 0.5561 | 0.8303 | 0.5140 | 0.5091 |

Both families selected **all 361 features**. The selected SVM reproduced v2 exactly: validation accuracy 51.40%, macro-F1 0.5090603; gain versus v2 **0.0000**. This experiment did **not** yield another improvement. V2 remains the extension winner; the baseline stays preserved.

Best internal CV score among predeclared hyperparameters at each k (selected estimates, not nested evaluation):

| model | k | cv_macro_f1 |
| --- | --- | --- |
| Logistic Regression | 32.0000 | 0.4201 |
| Logistic Regression | 64.0000 | 0.4575 |
| Logistic Regression | 128.0000 | 0.5205 |
| Logistic Regression | 256.0000 | 0.5474 |
| Logistic Regression | 361.0000 | 0.5478 |
| SVM | 32.0000 | 0.4378 |
| SVM | 64.0000 | 0.4774 |
| SVM | 128.0000 | 0.5250 |
| SVM | 256.0000 | 0.5542 |
| SVM | 361.0000 | 0.5561 |

## What this says about overfitting and features

The 83.03% training / 51.40% validation gap is consistent with overfitting and actor/domain differences. It does not identify one cause. Having 5,149 clips does not mean 5,149 independent speakers: training contains only 63 actors. Fixed clip summaries can encode speaker/microphone properties while discarding useful frame order. The personal microphone test adds a further recording-domain shift. Its 25% result is a real failure; feature redundancy alone has not explained it.

The tested univariate feature-selection strategy worsened internal grouped scores when it removed many columns. This suggests useful information is distributed across features or their interactions under these models. It does not prove every column is necessary or every feature-selection method would fail. ANOVA ranks features independently; it does not optimize joint subsets. More sophisticated selection and temporal/pretrained representations remain hypotheses requiring fresh, properly controlled evaluation. No promise of 80-90% accuracy or guaranteed personal-voice improvement is made.

Original assignment evidence, v2 gains/uncertainty and failed personal results all remain valid and separate. Full search, selected column names, warnings, training-only scaling/ANOVA checks and validation predictions are saved here. [Protocol](PROTOCOL.md) fixes the tested design. No commit or push occurred.
