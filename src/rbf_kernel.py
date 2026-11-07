"""Vectorized RBF evaluation against training references only."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_array, check_is_fitted


class TrainingRBFKernel(TransformerMixin, BaseEstimator):
    """Give SVC the ordinary RBF kernel through its precomputed interface.

    Gamma='scale' is calculated from the fitted training representation, just as
    in SVC. Validation/test transforms never fit or change the reference matrix.
    Vectorized pairwise distances avoid repeated scalar feature loops in libsvm.
    Training references are retained so the saved pipeline handles new audio.
    """

    def __init__(self, gamma="scale"):
        self.gamma = gamma

    def fit(self, features, y=None):
        features = check_array(features, dtype=np.float64)
        self.reference_features_ = features.copy()
        self.n_features_in_ = features.shape[1]
        if self.gamma == "scale":
            variance = features.var()
            self.gamma_ = 1 / (features.shape[1] * variance) if variance else 1.0
        else:
            self.gamma_ = float(self.gamma)
            if self.gamma_ <= 0:
                raise ValueError("RBF gamma must be positive")
        return self

    def transform(self, features):
        check_is_fitted(self, ["reference_features_", "gamma_"])
        features = check_array(features, dtype=np.float64)
        if features.shape[1] != self.n_features_in_:
            raise ValueError("RBF input feature count changed")
        return rbf_kernel(features, self.reference_features_, gamma=self.gamma_)
