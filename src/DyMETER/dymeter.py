"""
DyMETER: unofficial re-implementation from the paper.

Zhu, Cai, Chen, Deng, Ooi, Zhang. "Catching Every Ripple: Enhanced Anomaly Awareness via Dynamic
Concept Adaptation". IEEE TPAMI 2026 (arXiv:2604.14726v2). The official code linked in the paper
(github.com/zjiaqi725/DyMETER) was not public when this was written (HTTP 404).

DyMETER extends METER (same authors; official code in ../METER). This module reuses the official
METER model unchanged and adds what the paper describes on top of it:

  * IEC trained with the focal evidential loss, Eq. (7), gamma = 2.
  * Routing per Algorithm 2: dynamic (hypernetwork-shifted) detector iff U_ctr(x) > mu_e.
  * Uncertainty-aware anomaly score, Eq. (15): A(x) = R_e * exp(lambda * U_ctr * (r_e - R_e)),
    lambda = 0.6, R_e = RMSE reconstruction error, r_e initialised to the max training R_e.
  * Dynamic Threshold Optimization, Eqs. (17)-(19): windows W_N and W_C of size 64, tau = 0.95,
    kappa = 0.8, delta = MAD(W_N), mu_t = max training U_ctr.

Assumptions where the paper is silent or ambiguous (see README.md):
  * r_e is updated by an EMA with decay 0.99 over samples the DTO classifies as normal.
  * The paper writes mu_a^0 = F_N^{-1}(1 - tau) while stating an FPR bound of tau with tau = 0.95;
    we use the tau-quantile of W_N (FPR bound 1 - tau = 5%), the reading consistent with the text.
  * Eq. (7) is used without METER's extra Dirichlet KL regulariser.
  * Eq. (15) is evaluated in log space (rank-equivalent, avoids overflow); the DTO runs on log scores.
"""

from collections import deque

import numpy as np
import torch

from METER import meter as _meter_module
from METER import METER


def focal_evidential_loss(alpha, y, lamb=0.0, gamma=2.0):
    """Eq. (7): (1 - p_c)^gamma * (log sum_k alpha_k - log alpha_c), averaged over the batch.
    `lamb` is accepted for signature compatibility with edl_pytorch.evidential_classification and ignored."""
    y = y.long().view(-1, 1)
    S = alpha.sum(-1)
    alpha_c = alpha.gather(1, y).squeeze(1)
    p_c = alpha_c / S
    return ((1.0 - p_c).pow(gamma) * (torch.log(S) - torch.log(alpha_c))).mean()


class DyMETER(METER):
    def __init__(self, *args, lam_score=0.6, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam_score = lam_score
        self.focal_gamma = focal_gamma

    def train_autoencoder(self, *args, **kwargs):
        """METER training, with the IEC loss replaced by the focal evidential loss (Eq. 7)."""
        orig = _meter_module.evidential_classification
        gamma = self.focal_gamma
        _meter_module.evidential_classification = lambda a, y, lamb=0.0: focal_evidential_loss(a, y, lamb, gamma)
        try:
            return super().train_autoencoder(*args, **kwargs)
        finally:
            _meter_module.evidential_classification = orig

    @torch.no_grad()
    def reconstruct(self, x, static_encoder_weight, static_decoder_weight, mu_e):
        """Algorithm 2, lines 2-8: returns (R_e, U_ctr, used_dynamic) for a single instance x (1 x d)."""
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        z, _ = self.encoder_static.forward(new)
        out_static, _ = self.decoder_static.forward(z)
        alpha = self.edl_model(out_static)
        S = alpha.sum(-1, keepdims=True)
        p = alpha / S
        data_unc = torch.sum(p * (torch.digamma(S + 1) - torch.digamma(alpha + 1)), dim=1)
        entropy = -torch.sum(p * torch.log(p + 1e-7), dim=1)
        u_ctr = float(entropy - data_unc)                      # Eq. (8): mutual information
        use_dynamic = u_ctr > mu_e
        if use_dynamic:
            enc_w = self.hyperen.forward(cond_input=new.mean(0).view(1, -1))
            dec_w = self.hyperde.forward(cond_input=new.mean(0).view(1, -1))
            enc_h = [enc_w[j] + list(static_encoder_weight)[j] for j in range(len(enc_w))]
            dec_h = [dec_w[j] + list(static_decoder_weight)[j] for j in range(len(dec_w))]
            z, _ = self.encoder.forward(new, weights=enc_h)
            out, _ = self.decoder.forward(z, weights=dec_h)
        else:
            out = out_static
        r_e = float(torch.sqrt(torch.mean((new - out) ** 2)))   # RMSE reconstruction error
        return r_e, u_ctr, bool(use_dynamic)

    def anomaly_score(self, r_e, u_ctr, pivot):
        """Eq. (15) in log space: log A = log R_e + lambda * U_ctr * (r_e - R_e).
        Rank-equivalent to A (log is monotone), so AUC is unchanged, but it cannot overflow when the
        pivot r_e is large (ADBench features are not min-max normalised as in the paper's datasets)."""
        return float(np.log(r_e + 1e-12) + self.lam_score * u_ctr * (pivot - r_e))


class DynamicThreshold:
    """Dynamic Threshold Optimization, Eqs. (17)-(19)."""

    def __init__(self, init_threshold, mu_t, tau=0.95, kappa=0.8, window=64, min_fill=8):
        self.init_threshold = init_threshold
        self.mu_t = mu_t
        self.tau = tau
        self.kappa = kappa
        self.window = window
        self.min_fill = min_fill
        self.reset()

    def reset(self):
        self.W_N = deque(maxlen=self.window)
        self.W_C = deque(maxlen=self.window)

    def threshold(self):
        if len(self.W_N) < self.min_fill:
            return self.init_threshold
        wn = np.asarray(self.W_N)
        mu0 = float(np.quantile(wn, self.tau))
        mu_r = self.kappa * (mu0 - float(np.median(self.W_C))) if self.W_C else 0.0
        return mu0 + mu_r

    def update(self, score, u_ctr):
        """Classify `score` with the current threshold, then update the windows. Returns True if anomalous."""
        thr = self.threshold()
        if len(self.W_N) >= self.min_fill:
            wn = np.asarray(self.W_N)
            mu0 = float(np.quantile(wn, self.tau))
            delta = float(np.median(np.abs(wn - np.median(wn))))
            if u_ctr > self.mu_t and abs(score - mu0) <= delta:
                self.W_C.append(score)
        is_anomaly = score > thr
        if not is_anomaly:
            self.W_N.append(score)
        return is_anomaly
