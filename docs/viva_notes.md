# Viva preparation

Read the executed notebook and actual final report before presenting. Understand the decisions; do not memorize invented explanations for every error.

## Task and dataset

**What are you predicting?** Six intended emotions encoded in filenames: anger, disgust, fear, happy, neutral and sad. This is acoustic emotion recognition, not word-level sentiment analysis.

**Why no transcription?** The task emphasizes intonation and spectral/prosodic features. The same sentence can express several emotions, so text alone is insufficient.

**Why split by actor?** A random clip split allows one speaker into multiple sets. A model could exploit identity; held-out actors test a stronger generalization claim.

**Why approximately 70/15/15?** Q2 asks for suitable three-way sets without specifying ratios. 63/14/14 actors was chosen before processing. Clips per actor differ, so exact clip percentages differ.

**What is the validation set for?** Hyperparameters and the winning model. The test set estimates performance only after selection. Models remain fitted on training actors only.

## Signals and features

**What does the filter do?** It attenuates high frequencies above 5 kHz. A fourth-order Butterworth design is smooth; SOS improves numerical stability; forward/backward filtering removes phase delay but squares the magnitude response. It does not guarantee better classification or denoising.

**Why not report improved SNR?** We lack paired clean ground truth. We measure attenuation and validation classification changes instead.

**What are MFCCs?** Log-Mel spectral energies transformed by a cosine transform, compactly describing spectral shape. Deltas approximate changes across nearby frames. We retain frame means/std, which lose sequence order.

**Mel versus MFCC?** Mel retains perceptually spaced spectral bands; MFCC transforms log-Mel bands into cepstral coefficients. They overlap in information and are not independent.

**Centroid versus bandwidth?** Centroid is the spectrum's weighted frequency center; bandwidth measures weighted spread, not simply maximum minus minimum frequency.

**Bicoherence versus power spectrum?** Power uses second-order information. Normalized bispectrum examines phase coupling between f1, f2 and f1+f2 across segments. Our coarse estimator is noisy on short clips and does not prove significant coupling.

**Why qualify phase moments?** Angles wrap at ±π, so ordinary means/variance can mislead. Circular statistics would be a useful extension; we report the requested descriptive moments with that limitation.

## Models and leakage

**Why scale?** Distance/margin/gradient-based models are sensitive to feature units. Scaling statistics must come from training data only.

**Why is pre-split LDA wrong?** LDA uses class labels. Fitting on all observations uses validation/test labels to construct the representation. Our pipeline fits it only on training actors.

**Why GaussianNB?** Our real-valued summaries include negative scaled values. GaussianNB models continuous class-conditional densities; MultinomialNB expects nonnegative count-like inputs.

**Why macro-F1?** It balances each emotion's precision and recall and weights classes equally. Accuracy can hide weak minority-class recall. In single-label multiclass classification, micro-F1 equals accuracy.

**Why does the SVM say precomputed?** The preceding transformer calculates the ordinary RBF Gram matrix against training references. SVC optimizes the same RBF classification problem, while vectorized distance computation improves runtime. Gamma is fitted from training features only. Equivalence tests compare it against standard RBF SVC; this does not change the validation grid or introduce test fitting.

**Why can KNN training accuracy be perfect?** Distance-weighted neighbors include the point itself. This is not evidence of perfect unseen-speaker performance.

**What does an MLP warning mean?** The solver reached its maximum iterations before its convergence condition. Report it and interpret the result; do not silently delete it or claim convergence.

**What if the test-best model differs from the validation winner?** Keep the validation winner. Choosing by test results would turn the test set into another validation set.

## Additional experiments

**Why fixed-SVM ablation?** Changing both features and tuned classifiers would confound the comparison. Holding SVM settings and actors constant isolates representation differences more clearly.

**Why actor bootstrap?** Clips within one speaker are related. Resampling actors retains that grouping. Intervals are conditional on one trained model and 14 held-out actors, not total uncertainty.

**What is original here?** Independently written code/prose, reproducible actor split, leakage safeguards, feature estimator, controlled experiments, uncertainty and actual error analysis. Standard algorithms are correctly credited. Variable renaming alone is not originality.

## Before the presentation

- State the actual validation winner and its held-out scores from `reports/final_report.md`.
- Explain whether filtering helped in the fixed-SVM ablation, using the actual difference.
- Identify the largest observed validation confusion and listen to its example.
- Explain the measured training/validation gap without claiming a single proven cause.
- Show the frozen split and scaler check, then demonstrate `predict_emotion`.
- Add real group details and follow the course's disclosure/submission rules.
