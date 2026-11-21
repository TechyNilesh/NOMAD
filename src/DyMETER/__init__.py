"""
DyMETER (IEEE TPAMI 2026): unofficial re-implementation built on the official METER code.
See dymeter.py and README.md for what is taken from the paper and what is assumed.
"""

from .dymeter import DyMETER, DynamicThreshold, focal_evidential_loss

__all__ = ['DyMETER', 'DynamicThreshold', 'focal_evidential_loss']
