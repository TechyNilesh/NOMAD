"""Compatibility shim for running upstream METER with the hypnettorch release it pins (0.0.4).

Upstream meter.py unpacks `output, weights = mlp.forward(...)`, but hypnettorch's MLP.forward returns
only the output. `patch_hypnettorch()` makes MLP.forward also return the weights actually used: the
passed-in weights, or the network's own parameters when none are passed. Upstream presumably ran a
locally modified hypnettorch.
"""

from hypnettorch.mnets import MLP

_orig_forward = MLP.forward


def _forward_with_weights(self, x, weights=None, distilled_params=None, condition=None):
    out = _orig_forward(self, x, weights=weights, distilled_params=distilled_params, condition=condition)
    return out, (weights if weights is not None else list(self.internal_params))


def patch_hypnettorch():
    if MLP.forward is not _forward_with_weights:
        MLP.forward = _forward_with_weights
