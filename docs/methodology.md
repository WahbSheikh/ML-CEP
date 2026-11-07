# Methodology and literature review

## Research question
Can acoustic summaries classify six acted emotions for speakers not encountered during training? This is acoustic emotion recognition, not transcription-based positive/negative sentiment classification. The same spoken sentences appear with different emotions, so lexical content alone cannot resolve the labels.

## Data and evaluation design
The assignment supplies a modified CREMA-D dataset through `naweedfatima03/SentimentAnalysisDataset`. Use that noisy version, not an unnoticed substitution with clean CREMA-D. The original CREMA-D paper describes 7,442 clips from 91 actors [1]. Labels here are the intended emotions encoded in filenames, rather than crowd-majority perceived-emotion labels. That distinction limits interpretation.

Shuffle sorted actor identifiers once with NumPy `default_rng(42)`. Allocate 63/14/14 actors to training/validation/test. Audio decoding for integrity auditing is not feature fitting. Save the split manifest before filtering or extraction. Report actual clip percentages rather than asserting an exact 70/15/15 clip split. Q2 requires three sets but specifies no ratio; the chosen proportions are a project decision. Actor separation evaluates speaker generalization and avoids memorizing speaker identity. Shared sentence texts across actors are allowed; the experiment does not claim unseen-sentence generalization.

Raw-file SHA-256 and decoded-PCM SHA-256 detect exact duplicates. These do not detect every near duplicate. If identical decoded audio has conflicting emotion labels, exclude all members rather than choosing a label arbitrarily. Same-label duplicates retain only the first lexicographic filename. Reject invalid labels, non-finite, silent or shorter-than-40ms audio; audit exclusions. Freeze the manifest and fail if inputs subsequently change. The audit found three conflicting pairs (six files) and one all-zero recording, leaving 7,435 clips across all 91 actors.

## Filtering
Convert channels to mono and resample to 16 kHz. Apply a fourth-order Butterworth low-pass filter at 5 kHz in second-order sections with `sosfiltfilt` [3]. Forward-backward application has zero phase and squared magnitude response (effective eighth-order rolloff); its response at the design cutoff is approximately -6 dB, not the single-pass -3 dB. Audio is not peak-normalized for feature extraction, preserving relative energy. Notebook playback uses `normalize=False` for a fair comparison.

Filtering attenuates high-frequency components and may remove useful cues as well as noise. A spectral difference is not proof of improved signal-to-noise ratio. No clean paired reference exists, so no SNR improvement claim is made. The four-way ablation measures whether this choice helps validation classification.

## Feature rationale
Use 512-sample Hann STFT windows (32 ms), 160-sample hops (10 ms), and frame means/standard deviations to obtain clip vectors. No population statistics are learned during per-file extraction.

| Family | Implementation | Rationale and limitation |
|---|---|---|
| Fourier transform | Relative power in seven fixed frequency bands, frame mean/std | Broad spectral energy; discards within-band structure. |
| MFCC | 20 cepstra from 40 log-Mel bands, plus first/second deltas, mean/std | Compact spectral envelope and its change; the representation originated in speech recognition [2], not an emotion guarantee. |
| Mel spectrogram | 40 log-power bands, mean/std | Perceptually spaced spectral detail; related to MFCC, so features are correlated. |
| Chroma | Standard 12 pitch classes, mean/std | Coarse harmonic information; motivated by pitch-class representations [5], with weaker theoretical relevance to unpitched speech. |
| Spectral centroid | Magnitude-weighted frequency mean | Spectral brightness; can reflect noise as well as emotion. |
| Spectral bandwidth | Magnitude-weighted spread around centroid | Dispersion, not literally max minus min frequency. |
| Spectral contrast | Six peak/valley contrast bands, mean/std | Distinguishes harmonic peaks from spectral valleys [6]. |
| Bicoherence | Eight magnitude/phase moments of normalized bispectrum | Quadratic phase coupling beyond second-order spectra [7]; noisy for short signals. |
| Supplementary | Rolloff, flatness, RMS, zero-crossing rate, strongest spectral pitch peak and nonzero-pitch fraction | Additional brightness, energy and pitch proxies. `piptrack` is not a validated voiced/unvoiced detector. |

### Bicoherence estimator
Use 512-sample Hann-windowed frames with 50% overlap and remove each frame's DC component. For frame FFTs X, compute

`b(f1,f2) = mean[X(f1) X(f2) conj(X(f1+f2))] / sqrt(mean[|X(f1)X(f2)|²] mean[|X(f1+f2)|²])`.

Use f1<=f2 on 32 coarse bins at 62.5 Hz spacing from 62.5 to 2000 Hz; summed frequencies reach 4000 Hz. Omit denominator-negligible pairs. This controls cost and does not claim full-resolution bispectral estimation. Summarize magnitude and wrapped phase by mean, variance, skewness and excess kurtosis. Wrapped-phase ordinary moments are boundary-dependent; they satisfy the suggested descriptive feature list but are not circular statistics or proof of physiological coupling. Finite-sample estimates have bias and no coupling significance test is claimed.

## Training and model selection
All five classifiers use `StandardScaler` fitted on training clips only. Compare no reduction against supervised LDA (at most five dimensions for six classes). LDA uses labels, so fitting it before splitting would contaminate evaluation [4]. A pipeline enforces the boundary. GaussianNB remains valid with signed scaled features; MultinomialNB would not.

Evaluate the 76 predeclared candidates in `src/experiments.py`. Choose by validation macro-F1, then accuracy, then deterministic candidate order. This last order is an operational tie-break, not a universal complexity measure. MLP uses Adam, seed 42, no automatic internal validation, and at most 500 iterations. Record convergence warnings rather than hiding them. Macro-F1 weights each class equally; accuracy and weighted-F1 complement it. Micro-F1 would equal accuracy for this single-label multiclass task, so it adds little information.

Save all family winners and freeze selection metadata before opening test predictions. Models remain fitted on the training set; validation is not folded into final fitting. The overall winner is chosen on validation, even if another family obtains a better test result. Test comparisons are descriptive and cannot justify retuning.

### Efficient, equivalent RBF computation

The SVM pipeline evaluates `exp(-gamma * squared_distance)` using vectorized pairwise calculations and passes the Gram matrix to `SVC(kernel='precomputed')`. This is the same RBF classifier and parameter grid, not a linear-kernel substitution or a new hyperparameter. `TrainingRBFKernel` retains only training references, computes `gamma='scale'` from that training representation, and transforms validation/test points against those references without fitting. The saved pipeline retains these references for new-audio inference. Tests compare predictions, decision values and the fitted gamma against ordinary `SVC(kernel='rbf')` for both numeric and scale gamma. This optimization trades a temporary training Gram matrix (about 202 MiB for 5,149 clips) for substantially less repeated scalar distance computation. All SVM candidates were evaluated with the vectorized implementation before final selection; unchanged non-SVM candidates were retained only after their specifications and data identity were verified.

## Additional experiments

### Classifier mathematics

For standardized feature vector x, KNN votes among the k closest training vectors under Euclidean distance; distance weights use inverse distance with special handling for exact matches. Multinomial logistic regression models `P(y=c|x)=exp(w_c·x+b_c)/sum_j exp(w_j·x+b_j)` and minimizes cross-entropy with L2 regularization; C is inverse regularization strength. GaussianNB chooses the largest `log P(c) + sum_j log N(x_j; μ_cj, σ²_cj)` under a conditional-independence assumption. RBF SVM uses `K(x,z)=exp(-γ||x-z||²)` and soft-margin control C, with multiclass handling supplied by scikit-learn. The MLP composes ReLU hidden transformations and a softmax output, optimized using Adam and L2 regularization alpha. The neural network is a tabular-feature MLP, not a convolutional network on spectrogram images.

LDA projects onto directions maximizing between-class relative to within-class scatter, with at most C_classes−1 dimensions. This is supervised dimensionality reduction, not an unsupervised visualization. Training-only fitting is essential.

For each class, precision=`TP/(TP+FP)`, recall=`TP/(TP+FN)`, and F1=`2PR/(P+R)`. Undefined divisions are reported as zero. Macro-F1 is the unweighted mean of six class F1 values; weighted-F1 uses class support; balanced accuracy averages class recall. Rows of confusion matrices are actual labels, columns predicted labels.
Four ablations use the same fixed SVM (C=1, gamma=scale), training/validation actors and no LDA: unfiltered full, filtered MFCC-only, filtered spectral-only, and filtered full features. The spectral-only group contains centroid/bandwidth/contrast/rolloff/flatness and Fourier bands. The MFCC-only group includes deltas. Ablations are diagnostic and do not change the predeclared final filtered representation.

Learning curves use nested fixed subsets of 16,32,47,63 training actors with unchanged validation actors and the winning configuration. The curve is conditional on that selected configuration and is not an independent evaluation.

Bootstrap test actors, not individual clips, 1,000 times with seed 42. Repeated actors contribute all their clips each time. Percentile 95% intervals describe uncertainty across these 14 held-out actors; they do not include training/selection variability or prove significance between classifiers. Duration/intensity breakdowns and example errors are descriptive, with group sample sizes shown.

Subgroup macro-F1 retains the same six-label average as the primary metric. Some intensity groups do not contain every emotion, so absent classes contribute zero and these scores are not directly comparable across intensity groups. Subgroup balanced accuracy averages recall over classes actually present in that subgroup. Inspect class coverage, support and the full confusion patterns before interpreting differences.

## References
See [references.md](references.md) for primary sources and official API references. The reference student repository inspired the comparison target, not the implementation or reported numerical results.
