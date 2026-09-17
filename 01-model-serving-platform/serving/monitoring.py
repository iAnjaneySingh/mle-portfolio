"""Covariate drift detection: is the feature distribution the model is
seeing in production still statistically similar to what it was trained on?

Uses the two-sample Kolmogorov-Smirnov test per feature — a standard,
well-understood choice for comparing two empirical distributions without
assuming a parametric form. If any feature's incoming distribution has
drifted significantly from the training-time reference, requests are
flagged (not blocked — flagging first, blocking is a policy decision a
team makes deliberately, not something to bake silently into a monitoring
library).
"""

from typing import List

import numpy as np
from scipy import stats


class DriftDetector:
    def __init__(self, reference_samples: List[List[float]], feature_names: List[str],
                 p_value_threshold: float = 0.05):
        self.reference = np.array(reference_samples)
        self.feature_names = feature_names
        self.p_value_threshold = p_value_threshold

    def check(self, incoming_samples: List[List[float]]) -> dict:
        """Compare a batch of incoming feature vectors against the reference
        distribution, per feature. Returns per-feature p-values and an
        overall drift flag.

        A single request is too small a sample for a meaningful KS test —
        this is designed to be called periodically against a rolling batch
        of recent requests, not on every individual prediction.
        """
        incoming = np.array(incoming_samples)
        if len(incoming) < 2:
            raise ValueError("Need at least 2 incoming samples for a meaningful drift check")

        results = {}
        any_drift = False
        for i, name in enumerate(self.feature_names):
            ref_col = self.reference[:, i]
            incoming_col = incoming[:, i]
            ks_stat, p_value = stats.ks_2samp(ref_col, incoming_col)
            drifted = bool(p_value < self.p_value_threshold)
            any_drift = any_drift or drifted
            results[name] = {"ks_statistic": float(ks_stat), "p_value": float(p_value), "drifted": drifted}

        return {"per_feature": results, "any_drift": bool(any_drift)}
