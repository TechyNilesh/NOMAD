import numpy as np
from collections import deque
from typing import Any, Dict, List


class ADWIN:
    """
    Adaptive Windowing (ADWIN) for drift detection.

    Maintains a variable-size window of data values and drops the oldest
    bucket when two sub-windows have significantly different means, as
    determined by the Hoeffding bound.

    Reference:
        Bifet, A., & Gavalda, R. (2007). Learning from time-changing data
        with adaptive windowing. SDM.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._buckets: List[List[float]] = []
        self._total = 0.0
        self._variance = 0.0
        self._n = 0
        self.drift_detected = False
        self.drift_count = 0
        self.total_samples = 0

    def update(self, value: float) -> bool:
        if isinstance(value, np.ndarray):
            value = float(np.mean(value))

        self.total_samples += 1
        self._n += 1
        self._total += value
        self._insert_into_buckets(value)
        self.drift_detected = self._detect_drift()
        if self.drift_detected:
            self.drift_count += 1
        return self.drift_detected

    def _insert_into_buckets(self, value: float) -> None:
        bucket = [value, 1, value, value * value]
        self._buckets.insert(0, bucket)
        self._merge_buckets()

    def _merge_buckets(self) -> None:
        max_size = 5
        i = 0
        while i < len(self._buckets) - 1:
            if len([b for b in self._buckets if len(b) >= 4]) == 0:
                break
            if self._buckets[i][1] == self._buckets[i + 1][1] or (
                self._buckets[i][1] < max_size
                and self._buckets[i + 1][1] < max_size
            ):
                b1 = self._buckets.pop(i)
                b2 = self._buckets.pop(i)
                merged = [
                    b1[0] + b2[0],
                    b1[1] + b2[1],
                    b1[2] + b2[2],
                    b1[3] + b2[3],
                ]
                self._buckets.insert(i, merged)
            i += 1

    def _detect_drift(self) -> bool:
        total_n = self._n
        if total_n < 3:
            return False
        for i in range(len(self._buckets) - 1):
            n0 = sum(b[1] for b in self._buckets[: i + 1])
            n1 = total_n - n0
            if n0 < 2 or n1 < 2:
                continue
            sum0 = sum(b[2] for b in self._buckets[: i + 1])
            sum1 = sum(b[2] for b in self._buckets[i + 1 :])
            mu0 = sum0 / n0
            mu1 = sum1 / n1
            diff = abs(mu0 - mu1)
            eps = np.sqrt(
                (1.0 / (n0 * 1.0) + 1.0 / (n1 * 1.0))
                * np.log(2.0 / self.delta)
                / (2.0 * (n0 + n1))
            )
            if diff > eps:
                while i >= 0:
                    removed = self._buckets.pop(0)
                    self._total -= removed[2]
                    self._n -= removed[1]
                    i -= 1
                return True
        return False

    def reset(self) -> None:
        self._buckets = []
        self._total = 0.0
        self._variance = 0.0
        self._n = 0
        self.drift_detected = False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "drift_count": self.drift_count,
            "window_size": self._n,
        }


class KSWIN:
    """
    Kolmogorov-Smirnov Windowing (KSWIN) for drift detection.

    Maintains two windows: a reference window (older data) and a recent
    window (newer data). Compares their distributions using the
    Kolmogorov-Smirnov statistic. If the statistic exceeds the threshold,
    drift is declared.

    Reference:
        Raab, C., Heusinger, M., & Schleif, F.-M. (2020). Reactive soft
        prototype computing for concept drift. ICPRAM.
    """

    def __init__(self, alpha: float = 0.005, window_size: int = 200):
        self.alpha = alpha
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=2 * window_size)
        self.drift_detected = False
        self.drift_count = 0
        self.total_samples = 0

    def update(self, value: float) -> bool:
        if isinstance(value, np.ndarray):
            value = float(np.mean(value))

        self.total_samples += 1
        self._buffer.append(value)

        self.drift_detected = False
        if len(self._buffer) == self._buffer.maxlen:
            ref = list(self._buffer)[: self.window_size]
            recent = list(self._buffer)[self.window_size :]
            ks_stat = self._kolmogorov_smirnov(ref, recent)
            threshold = np.sqrt(
                -np.log(self.alpha / 2.0) / (2.0 * self.window_size)
            )
            if ks_stat > threshold:
                self.drift_detected = True
                self.drift_count += 1
                self._buffer.clear()

        return self.drift_detected

    def _kolmogorov_smirnov(self, a: List[float], b: List[float]) -> float:
        combined = sorted(set(a + b))
        cdf_a = np.searchsorted(sorted(a), combined, side="right") / len(a)
        cdf_b = np.searchsorted(sorted(b), combined, side="right") / len(b)
        return float(np.max(np.abs(cdf_a - cdf_b)))

    def reset(self) -> None:
        self._buffer.clear()
        self.drift_detected = False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "drift_count": self.drift_count,
            "buffer_size": len(self._buffer),
        }


class DDM:
    """
    Drift Detection Method (DDM) based on error-rate monitoring.

    Tracks the running error rate p and its standard deviation s.
    When p + s exceeds the warning threshold, a warning is triggered.
    When p + 2*s exceeds the drift threshold, drift is declared.

    Reference:
        Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004).
        Learning with drift detection. SBIA.
    """

    def __init__(
        self,
        min_instances: int = 30,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
    ):
        self.min_instances = min_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self._n = 0
        self._correct_sum = 0.0
        self._min_p = float("inf")
        self._min_s = 0.0
        self.drift_detected = False
        self.drift_count = 0
        self.warning_zone = False
        self.total_samples = 0

    def update(self, value: float) -> bool:
        if isinstance(value, np.ndarray):
            value = float(np.mean(value))

        self.total_samples += 1
        self._n += 1
        error = 0.0 if value > 0.5 else 1.0
        self._correct_sum += error
        p = self._correct_sum / self._n
        s = np.sqrt(p * (1.0 - p) / self._n)

        self.drift_detected = False
        self.warning_zone = False

        if self._n < self.min_instances:
            return False

        if p + s * self.drift_level < self._min_p + self._min_s:
            self._min_p = p
            self._min_s = s

        if p + s * self.drift_level > self._min_p + self.drift_level * self._min_s:
            self.drift_detected = True
            self.drift_count += 1
            self._reset()
            return True

        if p + s * self.warning_level > self._min_p + self.warning_level * self._min_s:
            self.warning_zone = True

        return False

    def _reset(self) -> None:
        self._n = 0
        self._correct_sum = 0.0
        self._min_p = float("inf")
        self._min_s = 0.0

    def reset(self) -> None:
        self._reset()
        self.drift_detected = False
        self.warning_zone = False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "drift_count": self.drift_count,
            "warning_zone": self.warning_zone,
        }
