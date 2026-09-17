import numpy as np
import pytest

from serving.monitoring import DriftDetector


def _reference_samples(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n, 3)).tolist()


def test_no_drift_when_same_distribution():
    reference = _reference_samples()
    rng = np.random.default_rng(1)
    incoming = rng.normal(loc=0.0, scale=1.0, size=(50, 3)).tolist()

    detector = DriftDetector(reference, ["f0", "f1", "f2"])
    result = detector.check(incoming)

    assert result["any_drift"] is False


def test_drift_detected_on_shifted_distribution():
    reference = _reference_samples()
    rng = np.random.default_rng(2)
    # shifted mean + larger spread -> should trip the KS test
    incoming = rng.normal(loc=5.0, scale=3.0, size=(50, 3)).tolist()

    detector = DriftDetector(reference, ["f0", "f1", "f2"])
    result = detector.check(incoming)

    assert result["any_drift"] is True
    assert all(result["per_feature"][f]["drifted"] for f in ["f0", "f1", "f2"])


def test_raises_on_too_few_samples():
    reference = _reference_samples()
    detector = DriftDetector(reference, ["f0", "f1", "f2"])
    with pytest.raises(ValueError):
        detector.check([[0.1, 0.2, 0.3]])
