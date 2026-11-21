"""
ARCUS: Adaptive Model Pooling for Online Deep Anomaly Detection from a Complex Evolving Data Stream

Yoon, Lee, Lee, Lee. KDD 2022.
Upstream code: https://github.com/kaist-dmlab/ARCUS (commit 0a6aa26, 2026-05-29).
The upstream .py files are fetched verbatim by tools/fetch_neural_baselines.sh. The upstream driver (main.py) is
re-implemented in benchmarks/scripts/arcus_run.py; see that file for the fixes it needed.
"""

import os
import sys

# Upstream modules import each other as top-level modules (`from CKA import ...`,
# `from model.model_utils import ...`), as in the upstream repo.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # upstream targets tf.keras 2 (TF 2.2)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .ARCUS import ARCUS  # noqa: E402

__all__ = ['ARCUS']
