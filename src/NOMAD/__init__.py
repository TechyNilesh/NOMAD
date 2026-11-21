"""
NOMAD: Neural Online Meta-learning for Automated Anomaly Detection

This package implements NOMAD, a neural meta-learning framework for
streaming anomaly detection that learns which algorithm-hyperparameter
configurations perform best under specific stream conditions.

Key Innovation: Algorithm-Aware Meta-Network (AAMN) architecture with
algorithm-specific prediction heads for improved configuration selection.
"""

from .nomad import (
    NOMAD,
    StreamFeatureExtractor,
    AlgorithmAwareMetaNetwork,
    PostProcessor,
    ConsensusRewardCalculator,
    PageHinkley,
)
from .drift_detectors import ADWIN, KSWIN, DDM

__all__ = [
    'NOMAD',
    'StreamFeatureExtractor',
    'AlgorithmAwareMetaNetwork',
    'PostProcessor',
    'ConsensusRewardCalculator',
    'PageHinkley',
    'ADWIN',
    'KSWIN',
    'DDM',
]

__version__ = '1.0.0'
__author__ = 'Nilesh Verma, Albert Bifet, Bernhard Pfahringer, Maroua Bahri'
