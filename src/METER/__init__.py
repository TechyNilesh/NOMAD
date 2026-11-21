"""
METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection

Zhu, Cai, Deng, Ooi, Zhang. PVLDB 17(4): 794-807, 2024.
Upstream code: https://github.com/zjiaqi725/METER (commit 9c0434b, 2024-07-24).
`meter.py` and `edl_pytorch/` are copied verbatim from upstream. The upstream
training/inference loop (main.py) is re-implemented in
benchmarks/scripts/meter_run.py; see that file for the fixes it needed.
"""

import os
import sys

# meter.py imports `edl_pytorch` as a top-level package, as in the upstream repo.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .meter import METER  # noqa: E402

__all__ = ['METER']
