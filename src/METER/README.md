# METER (third-party baseline)

Source: https://github.com/zjiaqi725/METER, commit `9c0434b7d7526a7eb6f7974b6cf9c3faa678826a` (2024-07-24).
Paper: Zhu et al., "METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection", PVLDB 17(4), 2024.

- `meter.py` = upstream `METER.py`, unmodified.
- `edl_pytorch/` = upstream vendored copy, unmodified.
- Upstream `main.py` is not copied; `benchmarks/scripts/meter_run.py` re-implements its loop.

The upstream repository has no licence file, so its code is not included here.
Run `tools/fetch_neural_baselines.sh` from the repository root to download it at the commit above.

## Fixes needed to run the upstream loop (all in meter_run.py)

1. `--mode hybrid+edl` reads `args.mu_e` and `args.beta_e`, which upstream `main.py` never defines.
   We set `mu_e=0.1` (from the paper's grid {0.001,...,0.4}) and `beta_e=0.3` (upstream commented-out value).
2. Upstream passes `batch_size=trans_batch_size` to `train_autoencoder`; that name is undefined and the
   method has no such parameter. We drop the argument.
3. After an online update upstream calls `model(data)` without the static weights, which fails in
   `hybrid+edl` mode. We pass the updated static weights.
4. `meter.py` unpacks `output, weights = mlp.forward(...)`, but hypnettorch 0.0.4 (the version upstream
   pins) returns only the output, so upstream fails on the first training step. meter_run.py wraps
   `hypnettorch.mnets.MLP.forward` to also return the weights actually used (the passed-in weights, or
   the network's own parameters). Upstream presumably ran a locally modified hypnettorch.

## Protocol

- `init_mode="official"`: like upstream, the first N *label-normal* points train the model and are
  removed from the test stream (uses labels).
- `init_mode="unsupervised"`: the first N points of the stream, no labels (comparable to the
  unsupervised streaming baselines).
- N = `train_rate` x stream length; the paper uses 0.2. Hyperparameters are fixed across datasets.
- AUC is computed on the scored (non-training) part of the stream only.
