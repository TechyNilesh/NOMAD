"""
SEAD: Streaming Ensemble of Anomaly Detectors

Implementation of Algorithm 1 from:
"SEAD: Unsupervised Ensemble of Streaming Anomaly Detectors"
Shah et al., ICML 2025.

Core idea: Maintain N base detectors with multiplicative weights.
Detectors that consistently output lower scores (i.e., fewer false alarms)
get higher weights. Anomaly score is the weighted sum of normalized scores.
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')

from pysad_rust import (
    IForestASD,
    RobustRandomCutForest,
    OnlineIsolationForest,
)

try:
    from pysad_rust import XStream
except ImportError:
    XStream = None

EPS = 1e-8


# ── Streaming quantile via sliding window ────────────────────────────────────

class StreamingQuantile:
    """Estimates quantile of a score relative to past W scores."""

    def __init__(self, window_size: int = 1000):
        self._window = deque(maxlen=window_size)

    def quantile(self, score: float) -> float:
        """Return rank of score in [0,1] relative to window."""
        self._window.append(score)
        if len(self._window) < 2:
            return 0.5
        arr = np.array(self._window)
        return float(np.mean(arr < score))


# ── Base detector wrapper ─────────────────────────────────────────────────────

class BaseDetector:
    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.quantile_estimator = StreamingQuantile(window_size=1000)

    def score(self, x: np.ndarray) -> Tuple[float, float]:
        """Return (raw_score, normalized_score)."""
        raw = self.model.fit_score_partial(x)
        if raw is None or not np.isfinite(raw):
            raw = 0.0
        norm = self.quantile_estimator.quantile(raw)
        return raw, norm


# ── SEAD ─────────────────────────────────────────────────────────────────────

class SEAD:
    """
    SEAD: Streaming Ensemble of Anomaly Detectors (Algorithm 1).

    Parameters
    ----------
    feature_mins : np.ndarray
        Per-feature minimum values (required for HST-based detectors).
    feature_maxes : np.ndarray
        Per-feature maximum values.
    eta : float
        Learning rate for multiplicative weights update (default: 1.0).
    lam : float
        KL regularization strength (default: 1e-6). With uniform prior,
        this reduces to negative entropy regularization (MWU).
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        feature_mins: np.ndarray = None,
        feature_maxes: np.ndarray = None,
        eta: float = 1.0,
        lam: float = 1e-6,
        random_state: int = 42,
    ):
        self.eta = eta
        self.lam = lam
        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)

        self.detectors: List[BaseDetector] = self._build_detectors(
            feature_mins, feature_maxes, random_state
        )
        self.N = len(self.detectors)

        # Log-space weights (unconstrained), initialized to 0 (uniform)
        self.log_weights = np.zeros(self.N)

        self.step = 0

    def _build_detectors(self, feature_mins, feature_maxes, seed) -> List[BaseDetector]:
        """Build N=13 base detectors: RRCF×4, IForestASD×4, SIF×4, SPC×1."""
        detectors = []

        # RRCF with 4 window sizes (tree_size in paper)
        for tree_size in [32, 64, 128, 256]:
            model = RobustRandomCutForest(
                num_trees=40,
                shingle_size=4,
                tree_size=tree_size,
                random_state=seed,
            )
            detectors.append(BaseDetector(model, f"RRCF_w{tree_size}"))

        # IForestASD with 4 window sizes
        for window_size in [256, 512, 1024, 2048]:
            model = IForestASD(
                initial_window_x=None,
                window_size=window_size,
                random_state=seed,
            )
            detectors.append(BaseDetector(model, f"IForestASD_w{window_size}"))

        # OnlineIsolationForest with 4 configs
        for num_trees, window_size in [(8, 256), (16, 512), (32, 512), (64, 1024)]:
            model = OnlineIsolationForest(
                num_trees=num_trees,
                max_leaf_samples=32,
                growth_criterion="adaptive",
                subsample=0.5,
                branching_factor=2,
                split="axisparallel",
                window_size=window_size,
            )
            detectors.append(BaseDetector(model, f"OIF_t{num_trees}_w{window_size}"))

        # SPC: statistical process control (mean-based z-score)
        detectors.append(BaseDetector(_SPCDetector(), "SPC"))

        return detectors

    def fit_score_partial(self, x: np.ndarray) -> float:
        """
        Process one instance: score it then update weights.

        Returns weighted ensemble anomaly score in [0, 1].
        """
        self.step += 1
        x = np.asarray(x, dtype=np.float64)

        # Step 1: get normalized scores from all detectors
        norm_scores = np.zeros(self.N)
        for i, det in enumerate(self.detectors):
            _, norm_scores[i] = det.score(x)

        # Step 2: compute softmax weights
        weights = self._softmax(self.log_weights)

        # Step 3: output weighted score (line 10 of Algorithm 1)
        ensemble_score = float(np.dot(weights, norm_scores))

        # Step 4: gradient update on log_weights (lines 11-12)
        # Loss = weighted_score + lambda * KL(weights || uniform)
        # Gradient w.r.t. log_w_i = w_i * (s_i - ensemble_score) + lambda * (w_i - 1/N)
        grad = weights * (norm_scores - ensemble_score) + self.lam * (weights - 1.0 / self.N)
        self.log_weights -= self.eta * grad

        return ensemble_score

    def _softmax(self, log_w: np.ndarray) -> np.ndarray:
        shifted = log_w - log_w.max()
        exp_w = np.exp(shifted)
        return exp_w / (exp_w.sum() + EPS)

    def get_weights(self) -> dict:
        """Return current detector weights for analysis."""
        weights = self._softmax(self.log_weights)
        return {det.name: float(w) for det, w in zip(self.detectors, weights)}


# ── Simple SPC detector ───────────────────────────────────────────────────────

class _SPCDetector:
    """
    Statistical Process Control: z-score based anomaly detector.
    Score = mean of per-dimension |z-scores|.
    """

    def __init__(self):
        self._mean = None
        self._m2 = None
        self._count = 0

    def fit_score_partial(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        d = len(x)

        if self._mean is None:
            self._mean = np.zeros(d)
            self._m2 = np.zeros(d)

        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

        if self._count < 2:
            return 0.0

        std = np.sqrt(self._m2 / (self._count - 1) + EPS)
        z_scores = np.abs((x - self._mean) / std)
        return float(z_scores.mean())
