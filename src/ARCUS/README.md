# ARCUS (third-party baseline)

Source: https://github.com/kaist-dmlab/ARCUS, commit `0a6aa263c2c93498836e2f26f1fc4ca3a412b5cd` (2026-05-29).
Paper: Yoon et al., "Adaptive Model Pooling for Online Deep Anomaly Detection from a Complex Evolving
Data Stream", KDD 2022.

- `ARCUS.py`, `CKA.py`, `utils.py`, `model/*.py` = upstream, unmodified.
- Upstream `main.py` / `datasets/data_utils.py` are not copied; `benchmarks/scripts/arcus_run.py`
  re-implements them.
- Requires TensorFlow with legacy Keras 2 (`tf-keras`, `TF_USE_LEGACY_KERAS=1`); upstream targets TF 2.2.

The upstream repository has no licence file, so its code is not included here.
Run `tools/fetch_neural_baselines.sh` from the repository root to download it at the commit above.

## Fixes needed (all in arcus_run.py)

1. `ARCUS._reduce_models_last` recurses with one argument too many
   (`self._reduce_models_last(self.model_pool, x_inp, epochs)`), raising a TypeError that `simulator`
   swallows, so a run aborts the first time a third model is merged. The runner replaces the method
   with the same logic and the correct call.
2. Upstream reads pre-normalised CSVs. ADBench features are raw, so the runner standardises each
   feature with the mean/std of the **first batch only** (no look-ahead).
3. `hidden_dim` follows the paper (number of principal components explaining >= 70% of variance),
   computed on the first batch only.

## Protocol

- Batch-incremental as upstream (batch 512, mini-batch 32, 5 initial / 1 interim epochs, RAPP backbone,
  3 layers, lr 1e-4, reliability 0.95, similarity 0.80, adaptive inference "ADP").
- Upstream reports the mean of per-batch AUCs. The runner reports the **stream-level AUC** over all
  scores (`auc`, comparable with the other methods) and also stores `auc_batch_mean`.
