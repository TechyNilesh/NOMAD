"""
NOMAD: Neural Online Meta-learning for Automated Anomaly Detection

This module implements NOMAD, a neural meta-learning framework that learns
which algorithm-hyperparameter configurations perform best under specific stream
conditions for streaming anomaly detection.

Key Innovation: Algorithm-Aware Meta-Network (AAMN) with algorithm-specific
prediction heads. Each algorithm has its own specialized head that processes
the shared stream representation along with algorithm-specific hyperparameters.

Architecture:
    Shared Encoder: meta_features(10) → Linear(10 → 32) + ReLU → stream_repr(32)

    Algorithm-Specific Heads (one per algorithm):
        [stream_repr(32) + algo_hp_vector] → Linear → ReLU → Linear → reward

    During prediction: Select appropriate head based on algorithm type

Benefits over single MLP:
    - Each algorithm learns its own hyperparameter-reward relationship
    - No padding needed for different HP dimensions
    - Better capacity utilization per algorithm
    - More interpretable (can analyze per-algorithm behavior)
    - Scales better when adding new algorithms

Authors: Nilesh Verma, Albert Bifet, Bernhard Pfahringer, Maroua Bahri
License: MIT
"""

import numpy as np
import random
import hashlib
import time as time_module
from collections import deque, Counter
from typing import Dict, List, Tuple, Any
import warnings

# PyTorch for neural network (CPU only)
import torch
import torch.nn as nn
import torch.optim as optim

# Force CPU usage
DEVICE = torch.device('cpu')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Numerical stability constant
EPS = 1e-8

# Import anomaly detection models from pysad_rust
from pysad_rust import (
    HalfSpaceTrees,
    IForestASD,
    RobustRandomCutForest,
    LODA,
    OnlineIsolationForest,
    XStream,
    StreamingIsolationForest,
)

# Import drift detectors
from .drift_detectors import ADWIN, KSWIN, DDM


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class PostProcessor:
    """
    Tracks min/max/mean/variance and returns a score normalized to [0,1].
    Used for normalizing raw anomaly scores from different detectors.

    Note: Uses fixed window style - statistics reset when window is full.
    """

    def __init__(self, max_history: int = 1000):
        self._min = float("inf")
        self._max = float("-inf")
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0
        self.max_history = max_history

    def process(self, score: float) -> float:
        """Normalize a score to [0, 1] range based on historical min/max."""
        if score < self._min:
            self._min = score
        if score > self._max:
            self._max = score

        self._count += 1
        self._sum += score
        self._sum_sq += score ** 2

        span = self._max - self._min
        normalized = 0.0 if span == 0 else (score - self._min) / span

        # Reset statistics when reaching max_history (fixed window behavior)
        if self._count >= self.max_history:
            self.reset()

        return normalized

    def reset(self):
        """Reset all statistics for a new fixed window."""
        self._min = float("inf")
        self._max = float("-inf")
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def mean(self) -> float:
        """Return mean of all processed scores."""
        return 0.0 if self._count == 0 else self._sum / self._count

    def variance(self) -> float:
        """Return variance of all processed scores."""
        if self._count < 2:
            return 0.0
        mean = self.mean()
        return (self._sum_sq / self._count) - mean**2


class PageHinkley:
    """
    Page-Hinkley test for detecting ABRUPT changes in mean.

    Designed to detect sudden shifts while ignoring gradual drift.
    Only triggers reset when there's a significant abrupt change.

    The test monitors the cumulative difference between observed values
    and their mean, triggering when this cumulative sum exceeds a threshold.

    Reference:
        Page, E. S. (1954). Continuous inspection schemes.
        Biometrika, 41(1/2), 100-115.

    Args:
        delta: Minimum magnitude of change to detect (default: 0.005)
               Higher delta = less sensitive to small changes
        threshold: Detection threshold (default: 50)
                   Higher threshold = only detect very abrupt changes
        alpha: Forgetting factor for gradual adaptation (default: 0.9999)
               Closer to 1 = longer memory, less adaptation to gradual drift
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.9999):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        # Running statistics
        self.n = 0
        self.mean = 0.0
        self.sum_ph = 0.0           # Page-Hinkley sum (for increase)
        self.sum_ph_min = 0.0       # Minimum of PH sum
        self.sum_ph_neg = 0.0       # Page-Hinkley sum (for decrease)
        self.sum_ph_max = 0.0       # Maximum of PH sum (negative direction)

        # Tracking
        self.drift_detected = False
        self.drift_count = 0
        self.total_samples = 0

    def update(self, value: float) -> bool:
        """
        Update with new value and check for abrupt drift.

        Args:
            value: New data point (scalar). For arrays, pass np.mean(X).

        Returns:
            True if ABRUPT drift detected, False otherwise
        """
        # Handle array input
        if isinstance(value, np.ndarray):
            value = float(np.mean(value))

        self.n += 1
        self.total_samples += 1
        self.drift_detected = False

        # Update running mean incrementally
        old_mean = self.mean
        self.mean = old_mean + (value - old_mean) / self.n

        # Page-Hinkley test for INCREASE
        self.sum_ph = self.alpha * self.sum_ph + (value - self.mean - self.delta)
        self.sum_ph_min = min(self.sum_ph_min, self.sum_ph)

        # Page-Hinkley test for DECREASE
        self.sum_ph_neg = self.alpha * self.sum_ph_neg + (self.mean - value - self.delta)
        self.sum_ph_max = min(self.sum_ph_max, self.sum_ph_neg)

        # Detect abrupt change (either direction)
        ph_increase = self.sum_ph - self.sum_ph_min
        ph_decrease = self.sum_ph_neg - self.sum_ph_max

        if ph_increase > self.threshold or ph_decrease > self.threshold:
            self.drift_detected = True
            self.drift_count += 1
            self._soft_reset()
            return True

        return False

    def _soft_reset(self):
        """Reset PH statistics after drift detection."""
        self.sum_ph = 0.0
        self.sum_ph_min = 0.0
        self.sum_ph_neg = 0.0
        self.sum_ph_max = 0.0
        self.n = max(1, self.n // 4)

    def reset(self):
        """Full reset of all statistics."""
        self.n = 0
        self.mean = 0.0
        self.sum_ph = 0.0
        self.sum_ph_min = 0.0
        self.sum_ph_neg = 0.0
        self.sum_ph_max = 0.0
        self.drift_detected = False

    def get_stats(self) -> Dict[str, Any]:
        """Get drift detection statistics."""
        return {
            "total_samples": self.total_samples,
            "drift_count": self.drift_count,
            "current_mean": self.mean,
            "ph_statistic_increase": self.sum_ph - self.sum_ph_min,
            "ph_statistic_decrease": self.sum_ph_neg - self.sum_ph_max,
            "threshold": self.threshold
        }


class ConsensusRewardCalculator:
    """
    Incremental reward calculator based on consensus between models.
    Uses running statistics (Welford-style) to compute correlation incrementally.

    Rewards models based on consensus: Strong agreement with other models
    indicates genuine anomalies and reliable detection.
    """

    def __init__(self, window_size: int = 1000,
                 min_updates_for_correlation: int = 10,
                 reward_mode: str = "consensus",
                 ema_decay: float = 0.995):
        self.n_models = None
        self.window_size = window_size
        self.min_updates_for_correlation = min_updates_for_correlation
        self.reward_mode = reward_mode
        self.n_updates = 0
        self.rewards = None

        # FIX 2: EMA decay instead of cumulative sums divided by global n.
        # Cumulative sums cause new models (added via evolution) to have near-zero
        # estimated stats for thousands of steps because sum/n_global ≈ 0.
        # EMA gives equal weight to recent history regardless of when a model
        # joined the pool — a replaced model catches up in ~1/(1-decay) steps.
        self.ema_decay = ema_decay
        self.ema_x = None    # EMA of scores per model
        self.ema_x2 = None   # EMA of scores^2 per model
        self.ema_xy = None   # EMA of outer product (for correlation)

        # Recent window for selection (Bug 3 fix): separate short-horizon
        # accumulator used only for argmax model selection, not AAMN training.
        self.recent_sum = None
        self.recent_count = 0

    def update_scores(self, normalized_scores: list):
        scores = np.array(normalized_scores)

        if self.n_models is None:
            self.n_models = len(scores)
            self.rewards = np.zeros(self.n_models)
            self.ema_x  = np.zeros(self.n_models)
            self.ema_x2 = np.zeros(self.n_models)
            self.ema_xy = np.zeros((self.n_models, self.n_models))
            self.recent_sum = np.zeros(self.n_models)

        d = self.ema_decay
        self.ema_x  = d * self.ema_x  + (1 - d) * scores
        self.ema_x2 = d * self.ema_x2 + (1 - d) * scores ** 2
        self.ema_xy = d * self.ema_xy + (1 - d) * np.outer(scores, scores)

        self.recent_sum   += scores
        self.recent_count += 1
        self.n_updates    += 1

    def calculate_rewards(self) -> np.ndarray:
        if self.n_updates < self.min_updates_for_correlation:
            if self.n_models is not None:
                self.rewards = np.ones(self.n_models) / self.n_models
            return self.rewards

        # FIX 2: stats from EMA — correctly estimated for all models regardless
        # of when they were added to the pool.
        mean_x = self.ema_x
        var_x  = self.ema_x2 - mean_x ** 2
        std_x  = np.sqrt(np.maximum(var_x, 1e-10))

        if self.reward_mode == "consensus":
            cov_xy    = self.ema_xy - np.outer(mean_x, mean_x)
            std_outer = np.maximum(np.outer(std_x, std_x), 1e-10)
            corr_matrix = np.clip(
                np.nan_to_num(cov_xy / std_outer, nan=0.0), -1.0, 1.0
            )
            np.fill_diagonal(corr_matrix, 0)

            # FIX 1: mean(positive correlations) not abs(mean(all correlations)).
            # abs(mean) penalises models that disagree with bad detectors.
            # mean(clip≥0) only rewards genuine positive consensus.
            self.rewards = np.mean(np.maximum(corr_matrix, 0.0), axis=1)
        else:
            m3   = self.ema_x2 * mean_x - mean_x ** 3  # simplified skew from EMA
            skew = m3 / np.maximum(std_x ** 3, 1e-10)
            self.rewards = np.maximum(skew, 0.0)

        reward_sum = np.sum(self.rewards)
        if reward_sum > 0:
            self.rewards = self.rewards / reward_sum
        else:
            self.rewards = np.ones(self.n_models) / self.n_models

        return self.rewards

    def get_recent_best(self) -> int:
        """FIX 3: argmax over recent window scores, not long-horizon EMA rewards.
        Used for model selection only (not AAMN training) to avoid incumbency bias."""
        if self.recent_count == 0 or self.n_models is None:
            return 0
        return int(np.argmax(self.recent_sum / self.recent_count))

    def reset_model(self, index: int):
        if self.n_models is None or index >= self.n_models:
            return
        # Zero out the replaced model's EMA stats — they will rebuild within
        # ~1/(1-ema_decay) steps, giving new models a fair chance immediately.
        self.ema_x[index]     = 0.0
        self.ema_x2[index]    = 0.0
        self.ema_xy[index, :] = 0.0
        self.ema_xy[:, index] = 0.0
        self.recent_sum[index] = 0.0

    def reset(self):
        self.n_updates = 0
        self.recent_count = 0
        if self.n_models is not None:
            self.ema_x     = np.zeros(self.n_models)
            self.ema_x2    = np.zeros(self.n_models)
            self.ema_xy    = np.zeros((self.n_models, self.n_models))
            self.recent_sum = np.zeros(self.n_models)
            self.rewards   = np.zeros(self.n_models)


# ══════════════════════════════════════════════════════════════════════════════
# STREAM FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class StreamFeatureExtractor:
    """
    Extracts statistical meta-features from the data stream using incremental statistics.
    Uses Welford's algorithm for numerically stable running mean/variance/skewness/kurtosis.

    Features (10 total):
        - mean, std, median, p25, p75, range, skewness, kurtosis, min, max

    All updates are O(1) per data point.

    Note: Uses fixed window style (feature windowing) - when the window fills,
    the completed window's features are snapshotted and the statistics restart,
    so features track the current regime rather than the full history.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.n = 0

        # Welford's algorithm variables
        self.mean = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0

        # Min and max tracking
        self.min_val = float('inf')
        self.max_val = float('-inf')

        # P2 quantile estimators
        self._init_p2_estimators()
        self._cached_features = None

        # Features of the last completed window, served during the new
        # window's warm-up so extraction never degrades to zeros mid-stream
        self._last_window_features = None

    def _init_p2_estimators(self):
        """Initialize P2 quantile estimators for p25, median, p75."""
        self.q_heights = {
            'p25': [float('inf')] * 5,
            'p50': [float('inf')] * 5,
            'p75': [float('inf')] * 5,
        }
        self.q_positions = {
            'p25': [1, 2, 3, 4, 5],
            'p50': [1, 2, 3, 4, 5],
            'p75': [1, 2, 3, 4, 5],
        }
        self.q_desired = {
            'p25': [0, 0.25/2, 0.25, (1+0.25)/2, 1],
            'p50': [0, 0.25, 0.5, 0.75, 1],
            'p75': [0, (1+0.75)/2 - 0.5, 0.75, (1+0.75)/2, 1],
        }
        self.q_initialized = False
        self.init_buffer = []

    def update(self, X: np.ndarray) -> float:
        """Incrementally update all running statistics - O(1)."""
        if isinstance(X, np.ndarray):
            value = float(np.mean(X))
        else:
            value = float(X)

        # Feature windowing: when the window is full, snapshot its features
        # and restart the statistics for a new window
        if self.n >= self.window_size:
            self._last_window_features = self.extract_meta_features().copy()
            self._start_new_window()

        self.n += 1
        self._cached_features = None

        if value < self.min_val:
            self.min_val = value
        if value > self.max_val:
            self.max_val = value

        # Welford's algorithm
        n = self.n
        delta = value - self.mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * (n - 1)

        self.mean += delta_n

        self.M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + \
                   6 * delta_n2 * self.M2 - 4 * delta_n * self.M3
        self.M3 += term1 * delta_n * (n - 2) - 3 * delta_n * self.M2
        self.M2 += term1

        self._update_quantile_estimates(value)

        return value

    def _update_quantile_estimates(self, value: float):
        """Update P2 quantile estimates - simplified version."""
        if self.n <= 5:
            self.init_buffer.append(value)
            if self.n == 5:
                sorted_vals = sorted(self.init_buffer)
                for key in ['p25', 'p50', 'p75']:
                    self.q_heights[key] = sorted_vals.copy()
                    self.q_positions[key] = [1, 2, 3, 4, 5]
                self.q_initialized = True
            return

        if not self.q_initialized:
            return

        for key, p in [('p25', 0.25), ('p50', 0.5), ('p75', 0.75)]:
            heights = self.q_heights[key]
            positions = self.q_positions[key]

            k = 0
            if value < heights[0]:
                heights[0] = value
                k = 0
            elif value >= heights[4]:
                heights[4] = value
                k = 3
            else:
                for i in range(1, 5):
                    if value < heights[i]:
                        k = i - 1
                        break

            for i in range(k + 1, 5):
                positions[i] += 1

            n = self.n
            desired = [1, 1 + (n - 1) * p / 2, 1 + (n - 1) * p,
                      1 + (n - 1) * (1 + p) / 2, n]

            for i in range(1, 4):
                d = desired[i] - positions[i]
                if (d >= 1 and positions[i + 1] - positions[i] > 1) or \
                   (d <= -1 and positions[i - 1] - positions[i] < -1):
                    d_sign = 1 if d > 0 else -1

                    qi = heights[i]
                    qim1 = heights[i - 1]
                    qip1 = heights[i + 1]
                    ni = positions[i]
                    nim1 = positions[i - 1]
                    nip1 = positions[i + 1]

                    q_new = qi + d_sign / (nip1 - nim1) * (
                        (ni - nim1 + d_sign) * (qip1 - qi) / (nip1 - ni) +
                        (nip1 - ni - d_sign) * (qi - qim1) / (ni - nim1)
                    )

                    if qim1 < q_new < qip1:
                        heights[i] = q_new
                    else:
                        if d_sign > 0:
                            heights[i] = qi + (qip1 - qi) / (nip1 - ni)
                        else:
                            heights[i] = qi - (qim1 - qi) / (nim1 - ni)

                    positions[i] += d_sign

    def _start_new_window(self):
        """Restart running statistics for a new window (keeps the last-window snapshot)."""
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self._init_p2_estimators()
        self._cached_features = None

    def reset(self):
        """Reset all statistics for fresh start (e.g., after drift detection)."""
        self._start_new_window()
        self._last_window_features = None

    def extract_meta_features(self) -> np.ndarray:
        """Extract meta-features from running statistics - O(1)."""
        if self._cached_features is not None:
            return self._cached_features

        if self.n < 5:
            if self._last_window_features is not None:
                return self._last_window_features
            return np.zeros(10)

        variance = self.M2 / self.n if self.n > 0 else 0.0
        std_val = np.sqrt(max(variance, 1e-10))

        if self.M2 > 1e-10:
            skewness = np.sqrt(self.n) * self.M3 / (self.M2 ** 1.5)
        else:
            skewness = 0.0

        if self.M2 > 1e-10:
            kurtosis = self.n * self.M4 / (self.M2 ** 2) - 3.0
        else:
            kurtosis = 0.0

        if self.q_initialized:
            median = self.q_heights['p50'][2]
            p25 = self.q_heights['p25'][2]
            p75 = self.q_heights['p75'][2]
        else:
            median = self.mean
            p25 = self.mean
            p75 = self.mean

        range_val = self.max_val - self.min_val if self.n > 0 else 0.0

        meta_features = np.array([
            self.mean,
            std_val,
            median,
            p25,
            p75,
            range_val,
            skewness,
            kurtosis,
            self.min_val if self.n > 0 else 0.0,
            self.max_val if self.n > 0 else 0.0,
        ])

        self._cached_features = meta_features
        return meta_features


# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM-AWARE META-NETWORK (AAMN)
# ══════════════════════════════════════════════════════════════════════════════

class AlgorithmHead(nn.Module):
    """PyTorch module for algorithm-specific prediction head.

    head_type='mlp' (default): Linear(input->hidden) -> ReLU -> Linear(hidden->1).
    head_type='perceptron': single Linear(input->1), no hidden layer, no ReLU.
        Used by the NOMAD-Perceptron ablation (a linear meta-learner that is
        faster and lighter than the full AAMN).
    """

    def __init__(self, input_dim: int, hidden_dim: int, head_type: str = "mlp"):
        super().__init__()
        self.head_type = head_type

        if head_type == "perceptron":
            self.fc1 = nn.Linear(input_dim, 1)
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.fc1.bias)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)
            self.relu = nn.ReLU()

            # Initialize weights
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.fc1.bias)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "perceptron":
            return self.fc1(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class AlgorithmAwareMetaNetwork(nn.Module):
    """
    Algorithm-Aware Meta-Network (AAMN) with algorithm-specific prediction heads.

    PyTorch implementation (CPU only) for efficient neural network operations.

    This is the key innovation of NOMAD: instead of a single MLP that treats
    algorithm type as just another input feature, we use:

    1. A SHARED ENCODER that processes stream meta-features into a
       "stream representation" that captures current data characteristics.

    2. ALGORITHM-SPECIFIC HEADS - each algorithm has its own small network
       that takes the stream representation + that algorithm's hyperparameters
       and predicts the expected reward.

    Architecture:
        ┌─────────────────────────────────────────────────────────────────┐
        │                 ALGORITHM-AWARE META-NETWORK                     │
        └─────────────────────────────────────────────────────────────────┘

                            meta_features (10)
                                   │
                                   ▼
                        ┌───────────────────┐
                        │  Shared Encoder   │
                        │  10 → 32 + ReLU   │
                        └─────────┬─────────┘
                                  │
                                  ▼
                        stream_repr (32-dim)
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
        ┌──────────┐       ┌──────────┐       ┌──────────┐
        │ HST Head │       │RRCF Head │       │LODA Head │  ... (6 heads)
        │ 32+3→16  │       │ 32+3→16  │       │ 32+2→16  │
        │   →1     │       │   →1     │       │   →1     │
        └──────────┘       └──────────┘       └──────────┘

    Benefits:
        - Each algorithm learns its own HP-to-reward mapping
        - No padding needed (each head sees only its own HPs)
        - Better capacity utilization
        - More interpretable
        - Only update relevant head during training

    Args:
        algorithm_names: List of algorithm names
        all_configs: List of (algo_name, hp_dict) tuples
        meta_feature_dim: Dimension of stream meta-features (10)
        shared_dim: Shared encoder output dimension (32)
        head_hidden_dim: Hidden dimension in each head (16)
        K: Number of configs to sample during selection
        diversity_weight: Weight for diversity penalty
        learning_rate: Learning rate
        momentum: Momentum for SGD
    """

    def __init__(
        self,
        algorithm_names: List[str],
        all_configs: List[Tuple[str, Dict]],
        meta_feature_dim: int = 10,
        shared_dim: int = 32,
        head_hidden_dim: int = 16,
        K: int = 20,
        diversity_weight: float = 0.3,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        head_type: str = "mlp"
    ):
        super().__init__()

        self.algorithm_names = algorithm_names
        self.n_algorithms = len(algorithm_names)
        self.algo_name_to_idx = {name: idx for idx, name in enumerate(algorithm_names)}
        self.all_configs = all_configs
        self.meta_feature_dim = meta_feature_dim
        self.shared_dim = shared_dim
        self.head_hidden_dim = head_hidden_dim
        self.K = K
        self.diversity_weight = diversity_weight
        self.lr = learning_rate
        self.momentum = momentum
        self.head_type = head_type

        # Compute HP dimensions per algorithm
        self.algo_hp_dims = self._compute_algo_hp_dims()
        self.hp_normalization = self._compute_hp_normalization()

        # ════════════════════════════════════════════════════════════════════
        # SHARED ENCODER: meta_features(10) → Linear(10 → 32) + ReLU
        # Perceptron mode drops the ReLU so the whole meta-learner is linear.
        # ════════════════════════════════════════════════════════════════════
        if head_type == "perceptron":
            self.shared_encoder = nn.Sequential(
                nn.Linear(meta_feature_dim, shared_dim)
            )
        else:
            self.shared_encoder = nn.Sequential(
                nn.Linear(meta_feature_dim, shared_dim),
                nn.ReLU()
            )
        # Initialize shared encoder weights
        nn.init.normal_(self.shared_encoder[0].weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.shared_encoder[0].bias)

        # ════════════════════════════════════════════════════════════════════
        # ALGORITHM-SPECIFIC HEADS
        # Each head: [stream_repr(32) + hp_vector(hp_dim)] → Linear → ReLU → Linear → 1
        # ════════════════════════════════════════════════════════════════════
        self.heads = nn.ModuleDict()
        for algo_name in self.algorithm_names:
            hp_dim = self.algo_hp_dims.get(algo_name, 3)
            input_dim = shared_dim + hp_dim
            # Use sanitized name for ModuleDict (replace invalid chars)
            safe_name = algo_name.replace("-", "_")
            self.heads[safe_name] = AlgorithmHead(input_dim, head_hidden_dim, head_type=head_type)

        # Store mapping for safe names
        self.algo_to_safe_name = {name: name.replace("-", "_") for name in algorithm_names}

        # Setup optimizer (SGD with momentum)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        self.loss_fn = nn.MSELoss()

        # Cache for pool member vectors
        self.pool_vectors_cache = {}

        # Ensure CPU
        self.to(DEVICE)

    def _compute_algo_hp_dims(self) -> Dict[str, int]:
        """Compute HP dimension for each algorithm."""
        algo_hp_dims = {}
        for algo_name, hp_config in self.all_configs:
            if algo_name not in algo_hp_dims:
                algo_hp_dims[algo_name] = len(hp_config)
            else:
                algo_hp_dims[algo_name] = max(algo_hp_dims[algo_name], len(hp_config))
        return algo_hp_dims

    def _compute_hp_normalization(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Compute min/max values for each HP across all configs per algorithm."""
        hp_ranges = {}

        algo_configs = {}
        for algo_name, hp_config in self.all_configs:
            if algo_name not in algo_configs:
                algo_configs[algo_name] = []
            algo_configs[algo_name].append(hp_config)

        for algo_name, configs in algo_configs.items():
            hp_ranges[algo_name] = {}
            all_hp_names = set()

            for hp_config in configs:
                all_hp_names.update(hp_config.keys())

            for hp_name in all_hp_names:
                values = []
                for hp_config in configs:
                    if hp_name in hp_config:
                        val = hp_config[hp_name]
                        if isinstance(val, (int, float)):
                            values.append(float(val))
                        elif isinstance(val, str):
                            hash_val = int(hashlib.md5(val.encode()).hexdigest(), 16) % 1000
                            values.append(float(hash_val))

                if len(values) > 0:
                    hp_ranges[algo_name][hp_name] = (min(values), max(values))
                else:
                    hp_ranges[algo_name][hp_name] = (0.0, 1.0)

        return hp_ranges

    def reset_network(self):
        """Reset all network weights to start learning from scratch."""
        # Re-initialize shared encoder
        nn.init.normal_(self.shared_encoder[0].weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.shared_encoder[0].bias)

        # Re-initialize all heads
        for safe_name in self.heads:
            head = self.heads[safe_name]
            nn.init.normal_(head.fc1.weight, mean=0.0, std=0.1)
            nn.init.zeros_(head.fc1.bias)
            if head.head_type != "perceptron":
                nn.init.normal_(head.fc2.weight, mean=0.0, std=0.1)
                nn.init.zeros_(head.fc2.bias)

        # Reset optimizer state
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)

        self.pool_vectors_cache.clear()

    def clear_stale_cache(self, current_pool_indices: List[int]):
        """Remove cache entries not in current pool."""
        stale_keys = [k for k in self.pool_vectors_cache if k not in current_pool_indices]
        for k in stale_keys:
            del self.pool_vectors_cache[k]

    def _encode_hp_config(self, algo_name: str, hp_config: Dict) -> np.ndarray:
        """Encode and normalize HP config into vector (NO PADDING - algorithm-specific size)."""
        hp_dim = self.algo_hp_dims.get(algo_name, len(hp_config))
        hp_vector = np.zeros(hp_dim)
        idx = 0

        for hp_name, hp_value in sorted(hp_config.items()):
            if idx >= hp_dim:
                break

            if isinstance(hp_value, (int, float)):
                val = float(hp_value)
            elif isinstance(hp_value, str):
                val = float(int(hashlib.md5(hp_value.encode()).hexdigest(), 16) % 1000)
            else:
                val = 0.0

            if algo_name in self.hp_normalization and hp_name in self.hp_normalization[algo_name]:
                min_val, max_val = self.hp_normalization[algo_name][hp_name]
                if max_val > min_val:
                    val = (val - min_val) / (max_val - min_val)
                else:
                    val = 0.5

            hp_vector[idx] = val
            idx += 1

        return hp_vector

    def _forward_shared(self, meta_features: np.ndarray) -> torch.Tensor:
        """Forward pass through shared encoder."""
        x = torch.tensor(meta_features, dtype=torch.float32, device=DEVICE)
        return self.shared_encoder(x)

    def _forward_head(self, algo_name: str, stream_repr: torch.Tensor, hp_vector: np.ndarray) -> float:
        """Forward pass through algorithm-specific head."""
        hp_tensor = torch.tensor(hp_vector, dtype=torch.float32, device=DEVICE)
        head_input = torch.cat([stream_repr, hp_tensor])

        safe_name = self.algo_to_safe_name[algo_name]
        out = self.heads[safe_name](head_input)

        return float(out.item())

    def forward(self, meta_features: np.ndarray, algo_name: str, hp_vector: np.ndarray) -> torch.Tensor:
        """Complete forward pass: shared encoder → algorithm-specific head."""
        x = torch.tensor(meta_features, dtype=torch.float32, device=DEVICE)
        stream_repr = self.shared_encoder(x)

        hp_tensor = torch.tensor(hp_vector, dtype=torch.float32, device=DEVICE)
        head_input = torch.cat([stream_repr, hp_tensor])

        safe_name = self.algo_to_safe_name[algo_name]
        return self.heads[safe_name](head_input)

    def predict(self, meta_features: np.ndarray, current_pool_indices: List[int]) -> int:
        """
        Predict best config index using diversity-aware Top-K sampling.

        For each candidate:
        1. Pass meta_features through shared encoder
        2. Use algorithm-specific head based on candidate's algorithm
        3. Apply diversity penalty
        4. Select best scoring config
        """
        all_indices = list(range(len(self.all_configs)))

        # Sample K configs
        if len(all_indices) > self.K:
            candidate_indices = np.random.choice(all_indices, self.K, replace=False).tolist()
        else:
            candidate_indices = all_indices

        # Pre-compute stream representation (shared for all candidates)
        with torch.no_grad():
            stream_repr = self._forward_shared(meta_features)

        best_idx = None
        best_score = -np.inf

        for config_idx in candidate_indices:
            algo_name, hp_config = self.all_configs[config_idx]
            hp_vector = self._encode_hp_config(algo_name, hp_config)

            # Use algorithm-specific head
            with torch.no_grad():
                predicted_reward = self._forward_head(algo_name, stream_repr, hp_vector)

            # Compute diversity penalty
            diversity_penalty = self._compute_diversity_penalty(
                algo_name, hp_vector, current_pool_indices, meta_features
            )

            score = predicted_reward - self.diversity_weight * diversity_penalty

            if score > best_score:
                best_score = score
                best_idx = config_idx

        if best_idx is None:
            best_idx = np.random.randint(len(self.all_configs))

        return best_idx

    def _get_cached_pool_vectors(self, pool_idx: int) -> Tuple[str, np.ndarray]:
        """Get cached encoded vectors for a pool member."""
        if pool_idx not in self.pool_vectors_cache:
            pool_algo_name, pool_hp_config = self.all_configs[pool_idx]
            pool_hp_vector = self._encode_hp_config(pool_algo_name, pool_hp_config)
            self.pool_vectors_cache[pool_idx] = (pool_algo_name, pool_hp_vector)
        return self.pool_vectors_cache[pool_idx]

    def _compute_diversity_penalty(self, candidate_algo: str, candidate_hp_vector: np.ndarray,
                                   current_pool_indices: List[int], meta_features: np.ndarray) -> float:
        """
        Compute diversity penalty based on algorithm type and HP similarity.

        Penalty is higher if:
        1. Same algorithm type as existing pool member
        2. Similar hyperparameters to existing pool member
        """
        if len(current_pool_indices) == 0:
            return 0.0

        max_similarity = 0.0

        for pool_idx in current_pool_indices:
            pool_algo, pool_hp_vector = self._get_cached_pool_vectors(pool_idx)

            # Algorithm similarity: 1.0 if same, 0.0 if different
            algo_similarity = 1.0 if candidate_algo == pool_algo else 0.0

            # HP similarity (only meaningful if same algorithm)
            if candidate_algo == pool_algo:
                min_len = min(len(candidate_hp_vector), len(pool_hp_vector))
                if min_len > 0:
                    candidate_hp = candidate_hp_vector[:min_len]
                    pool_hp = pool_hp_vector[:min_len]

                    norm_candidate = np.linalg.norm(candidate_hp)
                    norm_pool = np.linalg.norm(pool_hp)

                    if norm_candidate > 1e-8 and norm_pool > 1e-8:
                        hp_similarity = np.dot(candidate_hp, pool_hp) / (norm_candidate * norm_pool)
                        hp_similarity = max(0.0, hp_similarity)
                    else:
                        hp_similarity = 0.5
                else:
                    hp_similarity = 0.5

                similarity = 0.5 * algo_similarity + 0.5 * hp_similarity
            else:
                similarity = 0.2 * algo_similarity

            if similarity > max_similarity:
                max_similarity = similarity

        return max_similarity

    def update(self, meta_features: np.ndarray, config_idx: int, reward: float):
        """
        Update meta-network with observed reward using PyTorch autograd.

        Key: Only update the shared encoder AND the relevant algorithm-specific head.
        Other heads remain unchanged (natural regularization).
        """
        algo_name, hp_config = self.all_configs[config_idx]
        hp_vector = self._encode_hp_config(algo_name, hp_config)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        pred = self.forward(meta_features, algo_name, hp_vector)

        # Compute loss
        target = torch.tensor([reward], dtype=torch.float32, device=DEVICE)
        loss = self.loss_fn(pred, target)

        # Backward pass
        loss.backward()

        # Update weights
        self.optimizer.step()

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown."""
        shared_params = sum(p.numel() for p in self.shared_encoder.parameters())

        head_params = {}
        total_head_params = 0
        for algo_name in self.algorithm_names:
            safe_name = self.algo_to_safe_name[algo_name]
            params = sum(p.numel() for p in self.heads[safe_name].parameters())
            head_params[algo_name] = params
            total_head_params += params

        return {
            "shared_encoder": shared_params,
            "heads": head_params,
            "total_heads": total_head_params,
            "total": shared_params + total_head_params
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN NOMAD FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

class NOMAD:
    """
    NOMAD: Neural Online Meta-learning for Automated Anomaly Detection

    Complete Online Meta-Learning AutoML system for streaming anomaly detection.
    Uses Algorithm-Aware Meta-Network (AAMN) for improved configuration selection.

    Key Innovation: Algorithm-specific prediction heads that learn distinct
    patterns for each detector type, rather than treating algorithm as just
    another input feature.

    Features:
        - Algorithm-Aware Meta-Network with shared encoder + algorithm-specific heads
        - Diversity-aware Top-K candidate sampling
        - Rich meta-feature extraction from stream characteristics
        - Consensus-based reward calculation
        - Evolutionary model pool management
        - Direct online learning (no replay buffer)
        - Page-Hinkley-based drift detection

    Args:
        n_pool_models: Number of models in the pool (default: 10)
        n_candidate_configs: Number of random configs to generate (default: 100)
        random_state: Random seed for reproducibility
        verbose: Print evolution details
        evolution_interval: Steps between evolution events (default: 1000)
        shared_dim: Shared encoder output dimension (default: 32)
        head_hidden_dim: Hidden dimension in algorithm-specific heads (default: 16)
        K: Number of configs to sample during selection (default: 20)
        diversity_weight: Weight for diversity penalty (default: 0.3)
        feature_window_size: Window size for feature extraction (default: 100)
        score_max_history: Max history for score normalization (default: 1000)
        reward_min_updates: Min updates before correlation (default: 10)
        meta_lr: Learning rate for meta-learner (default: 0.01)
        meta_momentum: Momentum for meta-learner (default: 0.9)
        drift_delta: Minimum magnitude of change to detect (default: 0.005)
        drift_threshold: Detection threshold (default: 50.0)
        drift_alpha: Forgetting factor (default: 0.9999)
        drift_detector_type: Drift detector type: "page_hinkley", "adwin", "kswin", "ddm" (default: "page_hinkley")
    """

    def __init__(
        self,
        n_pool_models: int = 10,
        n_candidate_configs: int = 100,
        random_state: int = 42,
        verbose: bool = False,
        evolution_interval: int = 1000,
        shared_dim: int = 32,
        head_hidden_dim: int = 16,
        K: int = 20,
        diversity_weight: float = 0.3,
        feature_window_size: int = 100,
        score_max_history: int = 1000,
        reward_min_updates: int = 10,
        meta_lr: float = 0.01,
        meta_momentum: float = 0.9,
        drift_delta: float = 0.005,
        drift_threshold: float = 50.0,
        drift_alpha: float = 0.9999,
        drift_detector_type: str = "page_hinkley",
        meta_arch: str = "aamn",
        selection_mode: str = "meta",
        reward_mode: str = "consensus",
        output_score: str = "normalized",
        **kwargs
    ):
        # reward_mode: "quality" (skewness-based, selects for detection quality)
        # or "consensus" (legacy correlation-with-herd reward).
        self.reward_mode = reward_mode
        # output_score: "normalized" (legacy, per-window reset min-max) or "raw"
        # (selected model's raw score - preserves global ranking like baselines).
        self.output_score = output_score
        # meta_arch: "aamn" (default nonlinear heads) or "perceptron" (linear meta-learner).
        # selection_mode: "meta" (default, AAMN-selected best model) or
        #   "ensemble_avg" (RandomSearchEnsemble: mean of the random pool's scores,
        #   no meta selection/evolution).
        self.meta_arch = meta_arch
        self.selection_mode = selection_mode
        self.n_pool_models = n_pool_models
        self.n_candidate_configs = n_candidate_configs
        self.random_state = random_state
        self.verbose = verbose
        self.evolution_interval = evolution_interval
        self.shared_dim = shared_dim
        self.head_hidden_dim = head_hidden_dim
        self.K = K
        self.diversity_weight = diversity_weight

        # Store hyperparameters
        self.feature_window_size = feature_window_size
        self.score_max_history = score_max_history
        self.reward_min_updates = reward_min_updates
        self.meta_lr = meta_lr
        self.meta_momentum = meta_momentum

        # Drift detection parameters
        # Drift detection is a core, always-on component of NOMAD, so it is no
        # longer an exposed constructor argument. The flag is retained (defaulting
        # to True) only so the ablation study can disable it via kwargs.
        self.enable_drift_detection = kwargs.get("enable_drift_detection", True)
        self.drift_delta = drift_delta
        self.drift_threshold = drift_threshold
        self.drift_alpha = drift_alpha
        self.drift_detector_type = drift_detector_type

        # Single-family pool for the consensus-collapse stress test. When set to an
        # algorithm name, the entire configuration library (and hence the initial
        # pool and all evolution candidates) is restricted to that one family, so
        # every pool member shares the same bias and the consensus reward degrades.
        self.pool_family = kwargs.get("pool_family", None)

        np.random.seed(random_state)
        random.seed(random_state)

        self.feature_mins = kwargs.get("feature_mins")
        self.feature_maxes = kwargs.get("feature_maxes")

        # Algorithm types
        self.algorithm_types = [
            "HalfSpaceTrees",
            "IForestASD",
            "RobustRandomCutForest",
            "LODA",
            "OnlineIsolationForest",
            "XStream",
            "StreamingIsolationForest",
        ]

        # Get hyperparameter options
        self.hp_options = self._get_hyperparameter_options()

        # Generate random configs
        self.all_configs = self._generate_random_configs(n_candidate_configs)

        # Initialize components
        self.feature_extractor = StreamFeatureExtractor(window_size=self.feature_window_size)

        # Create Algorithm-Aware Meta-Network (AAMN)
        self.meta_learner = AlgorithmAwareMetaNetwork(
            algorithm_names=self.algorithm_types,
            all_configs=self.all_configs,
            meta_feature_dim=10,
            shared_dim=shared_dim,
            head_hidden_dim=head_hidden_dim,
            K=K,
            diversity_weight=diversity_weight,
            learning_rate=meta_lr,
            momentum=meta_momentum,
            head_type=("perceptron" if meta_arch == "perceptron" else "mlp")
        )

        # Model pool
        self.models = []
        self.model_configs = []
        self.scores = []

        # Initialize pool
        self._init_pool()

        # Reward calculator
        self.reward_calculator = ConsensusRewardCalculator(
            window_size=self.score_max_history,
            min_updates_for_correlation=self.reward_min_updates,
            reward_mode=self.reward_mode
        )

        # Drift detector
        if self.enable_drift_detection:
            if self.drift_detector_type == "page_hinkley":
                self.drift_detector = PageHinkley(
                    delta=self.drift_delta,
                    threshold=self.drift_threshold,
                    alpha=self.drift_alpha
                )
            elif self.drift_detector_type == "adwin":
                self.drift_detector = ADWIN(delta=self.drift_delta)
            elif self.drift_detector_type == "kswin":
                self.drift_detector = KSWIN(alpha=self.drift_alpha)
            elif self.drift_detector_type == "ddm":
                self.drift_detector = DDM(
                    min_instances=30,
                    warning_level=2.0,
                    drift_level=3.0,
                )
            else:
                raise ValueError(
                    f"Unknown drift_detector_type: {self.drift_detector_type}. "
                    "Must be one of: page_hinkley, adwin, kswin, ddm"
                )
        else:
            self.drift_detector = None

        # Tracking
        self.step = 0
        self.best_model_idx = 0
        self.drift_count = 0
        self.last_drift_step = None

        # Window-based drift detection accumulators
        self._window_sum = 0.0
        self._window_count = 0

    def _get_hyperparameter_options(self) -> Dict[str, Dict[str, List]]:
        """Get hyperparameter search space for each algorithm."""
        return {
            "HalfSpaceTrees": {
                "num_trees": [8, 16, 32, 64, 128],
                "max_depth": [5, 8, 10, 12, 15],
                "window_size": [50, 100, 150, 200, 250],
            },
            "IForestASD": {
                "window_size": [512, 1024, 2048, 4096],
                "n_estimators": [16, 32, 64, 128],
                "max_samples": [64, 128, 256, 512],
                "contamination": [0.05, 0.1, 0.15, 0.2],
                "anomaly_rate_threshold": [0.1, 0.2, 0.3, 0.4],
            },
            "RobustRandomCutForest": {
                "num_trees": [8, 16, 32, 64, 128],
                "tree_size": [64, 128, 256, 512],
                "shingle_size": [1, 2, 3],
            },
            "LODA": {
                "num_bins": [50, 100, 150, 200],
                "num_random_cuts": [5, 10, 25, 32, 50],  # 32 = baseline default
            },
            "OnlineIsolationForest": {
                "branching_factor": [2, 3, 4],
                "split": ["axisparallel", "hyperplane"],
                "num_trees": [8, 16, 32, 64, 128],
                "max_leaf_samples": [16, 32, 64],
                "window_size": [512, 1024, 2048, 4096],
                "growth_criterion": ["adaptive", "depth", "size"],
                "subsample": [0.1, 0.25, 0.5, 0.75, 1.0],
            },
            # XStream: baseline default is n_components=8, n_chains=8, depth=8
            "XStream": {
                "n_components": [4, 8, 16, 32],
                "n_chains": [4, 8, 16, 32],
                "depth": [6, 8, 10, 12],
            },
            # SIF: baseline default is n_trees=100, window_size=256
            "StreamingIsolationForest": {
                "n_trees": [50, 100, 150, 200],
                "window_size": [128, 256, 512, 1024],
            },
        }

    def _get_default_configs(self) -> List[Tuple[str, Dict]]:
        """Return one config per algorithm that exactly matches each baseline's defaults.
        These are prepended to the random search space so the pool always has
        at least one strong candidate from each algorithm family."""
        return [
            ("HalfSpaceTrees",         {"num_trees": 32,  "max_depth": 15, "window_size": 250}),
            ("IForestASD",             {"window_size": 2048, "n_estimators": 32, "max_samples": 256, "contamination": 0.1, "anomaly_rate_threshold": 0.3}),
            ("RobustRandomCutForest",  {"num_trees": 32,  "tree_size": 256, "shingle_size": 1}),
            ("LODA",                   {"num_bins": 100,  "num_random_cuts": 32}),
            ("OnlineIsolationForest",  {"branching_factor": 2, "split": "axisparallel", "num_trees": 32, "max_leaf_samples": 32, "window_size": 2048, "growth_criterion": "adaptive", "subsample": 1.0}),
            ("XStream",                {"n_components": 8, "n_chains": 8, "depth": 8}),
            ("StreamingIsolationForest", {"n_trees": 100, "window_size": 256}),
        ]

    def _generate_random_configs(self, n_configs: int) -> List[Tuple[str, Dict]]:
        """Generate configs: 7 baseline-matching anchors first (indices 0–6),
        then n_configs-7 random configs.

        Having all 7 algorithm families in the initial pool (first n_pool_models
        configs) lets the consensus reward immediately compare all detector types
        and identify the best family for the current stream — critical on datasets
        where the best algorithm (e.g. SIF on shuttle) differs from the default.

        When ``pool_family`` is set, every config is drawn from that single family,
        producing a monoculture pool for the consensus-collapse stress test.
        """
        if self.pool_family is not None:
            if self.pool_family not in self.algorithm_types:
                raise ValueError(f"Unknown pool_family: {self.pool_family}")
            hp_space = self.hp_options[self.pool_family]
            return [
                (self.pool_family,
                 {k: np.random.choice(v)
                  for k, v in hp_space.items()
                  if k not in ("feature_mins", "feature_maxes")})
                for _ in range(n_configs)
            ]

        configs = self._get_default_configs()   # 7 anchors at indices 0-6

        for _ in range(n_configs - len(configs)):
            algo_name = np.random.choice(self.algorithm_types)
            hp_space = self.hp_options[algo_name]
            hp_config = {k: np.random.choice(v)
                         for k, v in hp_space.items()
                         if k not in ("feature_mins", "feature_maxes")}
            configs.append((algo_name, hp_config))

        return configs

    def _create_model(self, algo_name: str, hp_config: Dict) -> object:
        """Create model instance from algorithm name and HP config."""
        params = hp_config.copy()
        params["random_state"] = self.random_state

        if algo_name == "HalfSpaceTrees":
            params["feature_mins"] = self.feature_mins
            params["feature_maxes"] = self.feature_maxes
            return HalfSpaceTrees(**params)
        elif algo_name == "IForestASD":
            return IForestASD(**params)
        elif algo_name == "RobustRandomCutForest":
            return RobustRandomCutForest(**params)
        elif algo_name == "LODA":
            return LODA(**params)
        elif algo_name == "OnlineIsolationForest":
            return OnlineIsolationForest(**params)
        elif algo_name == "XStream":
            # XStream does not accept random_state
            params.pop("random_state", None)
            return XStream(**params)
        elif algo_name == "StreamingIsolationForest":
            # pysad_rust SIF does not accept random_state
            params.pop("random_state", None)
            return StreamingIsolationForest(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

    def _init_pool(self):
        """Initialize model pool with the first n_pool_models configs."""
        for idx in range(self.n_pool_models):
            algo_name, hp_config = self.all_configs[idx]
            model = self._create_model(algo_name, hp_config)
            self.models.append(model)
            self.model_configs.append((algo_name, idx, hp_config))
            self.scores.append(PostProcessor(max_history=self.score_max_history))

    def fit_score_partial(self, X: np.ndarray) -> float:
        """
        Main online learning step.

        All models are trained on every instance. Best model from last window
        is used for predictions in current window.

        Args:
            X: Input data point

        Returns:
            Anomaly score from best performing model
        """
        self.step += 1

        # ══════════════════════════════════════════════════════════════════════
        # EVERY INSTANCE: Model scoring
        # ══════════════════════════════════════════════════════════════════════
        normalized_scores = []
        raw_scores = []

        for i, model in enumerate(self.models):
            raw_score = model.fit_score_partial(X)
            raw_scores.append(raw_score)
            normalized_score = self.scores[i].process(raw_score)
            normalized_scores.append(normalized_score)

        # Diagnostics hook (behavior-neutral): expose the current per-model
        # scores so instrumented runs can build reward/selection figures.
        self.last_raw_scores = raw_scores
        self.last_normalized_scores = normalized_scores

        # The score used for the global AUROC. Normalized scores are min-max'd
        # per window with resets, which is fine for internal cross-model
        # comparison (reward/consensus) but corrupts the global ranking. The
        # baselines feed the detector's raw score to the metric, so output the
        # selected model's raw score to match (output_score="raw").
        out_scores = raw_scores if self.output_score == "raw" else normalized_scores

        # RandomSearchEnsemble: average the random pool's scores (no selection).
        # Default NOMAD: use the meta-learner's currently selected best model.
        if self.selection_mode == "ensemble_avg":
            best_score = float(np.mean(out_scores))
        else:
            best_score = out_scores[self.best_model_idx]

        # ══════════════════════════════════════════════════════════════════════
        # EVERY INSTANCE: Incremental updates
        # ══════════════════════════════════════════════════════════════════════
        data_value = self.feature_extractor.update(X)
        self.reward_calculator.update_scores(normalized_scores)

        if self.drift_detector is not None:
            self._window_sum += data_value
            self._window_count += 1

        # ══════════════════════════════════════════════════════════════════════
        # AT INTERVALS: Meta-learning operations
        # ══════════════════════════════════════════════════════════════════════
        if self.step % self.evolution_interval == 0 and self.selection_mode != "ensemble_avg":
            rewards = self.reward_calculator.calculate_rewards()
            meta_features = self.feature_extractor.extract_meta_features()

            # Train meta-learner
            for i, (_, config_idx, _) in enumerate(self.model_configs):
                self.meta_learner.update(meta_features, config_idx, rewards[i])

            # Evolution step
            self._evolve(meta_features, rewards)

            # FIX 3: select best model using recent-window mean score, not the
            # long-horizon EMA reward used for AAMN training. Long-horizon rewards
            # favour incumbents; recent-window selection reacts to current behaviour.
            self.best_model_idx = self.reward_calculator.get_recent_best()

            # Reset recent window for the next interval
            self.reward_calculator.recent_sum[:] = 0.0
            self.reward_calculator.recent_count  = 0

            if self.verbose:
                best_algo = self.model_configs[self.best_model_idx][0]
                print(f"[Step {self.step}] Best model: {best_algo} (idx={self.best_model_idx}, reward={rewards[self.best_model_idx]:.4f})")

            # Check for drift
            if self.drift_detector is not None and self._window_count > 0:
                window_mean = self._window_sum / self._window_count
                abrupt_drift = self.drift_detector.update(window_mean)

                self._window_sum = 0.0
                self._window_count = 0

                if abrupt_drift:
                    self.drift_count += 1
                    self.last_drift_step = self.step

                    if self.verbose:
                        print(f"\n{'!'*70}")
                        print(f"ABRUPT DRIFT DETECTED at step {self.step}")
                        print(f"Resetting meta-learner...")
                        print(f"{'!'*70}\n")

                    self.meta_learner.reset_network()
                    self.feature_extractor.reset()
                    self.reward_calculator.reset()

        return best_score

    def _evolve(self, meta_features: np.ndarray, rewards: np.ndarray):
        """Evolution step: Replace worst model with meta-learner's suggestion."""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"NOMAD EVOLUTION at step {self.step}")
            print(f"{'='*70}")

        worst_idx = np.argmin(rewards)
        worst_reward = rewards[worst_idx]

        current_pool_indices = [config_idx for _, config_idx, _ in self.model_configs]

        suggested_config_idx = self.meta_learner.predict(meta_features, current_pool_indices)
        suggested_algo, suggested_hp_config = self.all_configs[suggested_config_idx]

        already_in_pool = suggested_config_idx in current_pool_indices

        if not already_in_pool:
            new_model = self._create_model(suggested_algo, suggested_hp_config)

            old_algo = self.model_configs[worst_idx][0]
            self.models[worst_idx] = new_model
            self.model_configs[worst_idx] = (suggested_algo, suggested_config_idx, suggested_hp_config)
            self.scores[worst_idx] = PostProcessor(max_history=self.score_max_history)
            self.reward_calculator.reset_model(worst_idx)

            updated_pool_indices = [config_idx for _, config_idx, _ in self.model_configs]
            self.meta_learner.clear_stale_cache(updated_pool_indices)

            if self.verbose:
                print(f"\nReplaced model {worst_idx}:")
                print(f"  Old: {old_algo:25s} (reward: {worst_reward:.4f})")
                print(f"  New: {suggested_algo:25s} (Config idx: {suggested_config_idx})")

        if self.verbose:
            print(f"\nCurrent pool composition:")
            algo_counts = Counter([algo for algo, _, _ in self.model_configs])
            for algo, count in sorted(algo_counts.items()):
                print(f"  {algo:25s}: {count}")
            print(f"{'='*70}\n")

    def get_model_pool_info(self) -> Dict[str, Any]:
        """Get detailed information about current model pool."""
        pool_info = {
            "step": self.step,
            "n_total_configs": len(self.all_configs),
            "models": []
        }

        for i, (algo_name, config_idx, hp_config) in enumerate(self.model_configs):
            model_info = {
                "index": i,
                "algorithm": algo_name,
                "config_idx": config_idx,
                "hyperparameters": hp_config,
                "score_mean": float(self.scores[i].mean()),
                "score_variance": float(self.scores[i].variance())
            }
            pool_info["models"].append(model_info)

        meta_features = self.feature_extractor.extract_meta_features()
        current_pool_indices = [config_idx for _, config_idx, _ in self.model_configs]
        suggested_config_idx = self.meta_learner.predict(meta_features, current_pool_indices)
        suggested_algo, suggested_hp_config = self.all_configs[suggested_config_idx]

        pool_info["meta_learner_suggestion"] = {
            "algorithm": suggested_algo,
            "config_idx": suggested_config_idx,
            "hyperparameters": suggested_hp_config
        }

        # Add parameter count
        pool_info["meta_network_params"] = self.meta_learner.get_parameter_count()

        return pool_info

    def get_current_best_model(self) -> Tuple[str, Dict, float]:
        """Get information about the current best performing model."""
        rewards = self.reward_calculator.calculate_rewards()
        best_idx = np.argmax(rewards)

        algo_name, _, hp_config = self.model_configs[best_idx]
        mean_score = self.scores[best_idx].mean()

        return algo_name, hp_config, mean_score


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Example usage of NOMAD on synthetic streaming data.
    """

    print("="*70)
    print("NOMAD: Neural Online Meta-learning for Automated Anomaly Detection")
    print("="*70)
    print()

    # Generate synthetic streaming data with concept drift
    np.random.seed(42)
    n_samples = 5000
    n_features = 10

    X_stream = []
    for i in range(n_samples):
        if i < 2500:
            X = np.random.randn(n_features) * 0.5
        else:
            X = np.random.randn(n_features) * 0.5 + 2.0

        if np.random.rand() < 0.05:
            X += np.random.randn(n_features) * 3.0

        X_stream.append(X)

    X_array = np.array(X_stream)
    feature_mins = X_array.min(axis=0)
    feature_maxes = X_array.max(axis=0)

    print("Initializing NOMAD with Algorithm-Aware Meta-Network (AAMN)...")
    print("Architecture:")
    print("  - Shared Encoder: meta_features(10) → Linear(32) + ReLU")
    print("  - Algorithm-Specific Heads (6 algorithms):")
    print("    - Each head: [stream_repr(32) + hp_vector] → Linear(16) + ReLU → Linear(1)")
    print()

    nomad = NOMAD(
        n_pool_models=10,
        n_candidate_configs=50,
        random_state=42,
        verbose=True,
        evolution_interval=500,
        shared_dim=32,
        head_hidden_dim=16,
        K=20,
        diversity_weight=0.3,
        feature_mins=feature_mins,
        feature_maxes=feature_maxes
    )

    # Print parameter count
    param_info = nomad.meta_learner.get_parameter_count()
    print(f"Meta-Network Parameters:")
    print(f"  - Shared Encoder: {param_info['shared_encoder']}")
    print(f"  - Algorithm Heads:")
    for algo, count in param_info['heads'].items():
        print(f"      {algo}: {count}")
    print(f"  - Total: {param_info['total']}")
    print()

    print(f"Processing {n_samples} samples from streaming data...\n")

    anomaly_scores = []
    for t, X in enumerate(X_stream):
        score = nomad.fit_score_partial(X)
        anomaly_scores.append(score)

        if (t + 1) % 1000 == 0:
            print(f"Processed {t + 1}/{n_samples} samples")

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    best_algo, best_hp, best_score = nomad.get_current_best_model()
    print(f"\nBest performing algorithm: {best_algo}")
    print(f"Hyperparameters: {best_hp}")
    print(f"Mean score: {best_score:.4f}")

    pool_info = nomad.get_model_pool_info()
    print(f"\nFinal pool composition:")
    algo_counts = Counter([m['algorithm'] for m in pool_info['models']])
    for algo, count in sorted(algo_counts.items()):
        print(f"  {algo:25s}: {count}")

    print(f"\nMeta-learner's suggestion for current stream state:")
    suggestion = pool_info['meta_learner_suggestion']
    print(f"  Algorithm: {suggestion['algorithm']}")
    print(f"  Hyperparameters: {suggestion['hyperparameters']}")

    print("\n" + "="*70)
    print("NOMAD execution completed successfully!")
    print("="*70)
