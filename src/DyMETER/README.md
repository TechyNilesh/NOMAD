# DyMETER (unofficial re-implementation)

Paper: Zhu, Cai, Chen, Deng, Ooi, Zhang, "Catching Every Ripple: Enhanced Anomaly Awareness via Dynamic
Concept Adaptation", IEEE TPAMI 2026 (arXiv:2604.14726v2, DOI 10.1109/TPAMI.2026.3682661).

**This is not the authors' code.** The repository linked in the paper (github.com/zjiaqi725/DyMETER)
returned HTTP 404 when this was written. Results must be reported as "DyMETER (our re-implementation)".

## What is reused vs. added

- Reused unchanged: the official METER model (`../METER/meter.py`), which DyMETER extends (SCD, IEC, DSD
  hypernetwork, offline updating).
- Added from the paper:
  - IEC focal evidential loss, Eq. (7), gamma = 2
  - Routing per Algorithm 2: dynamic detector iff U_ctr > mu_e
  - Uncertainty-aware anomaly score, Eq. (15), lambda = 0.6
  - Dynamic Threshold Optimization, Eqs. (17)-(19): |W_N| = |W_C| = 64, tau = 0.95, kappa = 0.8,
    delta = MAD(W_N), mu_t = max training U_ctr

## Assumptions (the paper is silent or ambiguous)

1. The pivot r_e starts at the maximum training reconstruction error and is updated by an EMA with
   decay 0.99 over samples the DTO classifies as normal (the paper only says "EMA").
2. The paper writes mu_a^0 = F_N^{-1}(1 - tau) but states an FPR upper bound of tau with tau = 0.95. We
   use the tau-quantile (FPR bound 5%), the only reading consistent with the stated behaviour.
3. Eq. (7) is used as written, without METER's extra Dirichlet KL regulariser.
4. R_e is the RMSE reconstruction error, as the paper states (METER's code scores with the L1 norm).
5. The DTO only changes binary decisions; the stream AUC depends on the score A(x), not the threshold.
   Decisions are saved so F1 / precision / recall can be reported too.
6. Eq. (15) is evaluated in log space, log A = log R_e + lambda U (r_e - R_e). This ranks samples
   exactly like A (so AUC is identical) but cannot overflow on unnormalised features; the DTO runs
   on these log scores.
7. Hyperparameters are fixed across datasets (the paper grid-searches mu_p and mu_e per dataset).
