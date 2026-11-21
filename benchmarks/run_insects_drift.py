#!/usr/bin/env python3
"""
Real concept-drift runner on the INSECTS controlled-drift streams (USP
repository, Souza et al.), converted to a binary anomaly-detection task.

Unlike the main benchmark, the stream is processed in its original temporal
order (no shuffle) so that known change points are honoured. For each method we
report:
  - overall stream AUC and Precision@k,
  - windowed AUC over the stream,
  - recovery: instances to return within 95% of the pre-drift AUC after each
    known change point (abrupt stream), or against a warm-up baseline for the
    incremental / gradual / reoccurring streams where drift is distributed.

Methods run here are the prequential detectors (NOMAD, NOMAD-P, Random,
AutoSAD, SEAD, and the fixed detectors). ARCUS / METER / DyMETER are
batch-incremental and are executed through their own runners
(arcus_run.py / meter_run.py / dymeter_run.py) with temporal order preserved;
see the note in run_all_experiments.py.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR / "src"))

from pysad.utils import ArrayStreamer
from scripts.utils import get_xy_from_npz, compute_precision_at_k, compute_precision_at_k_windowed


# USP file name -> (drift type, known change points for the abrupt stream).
# Drift points are taken from the USP repository stream metadata (Souza et al.
# 2020, "Challenges in Benchmarking Stream Learning Algorithms with Real-world
# Data"); verify the exact indices against the released metadata before citing.
STREAMS = {
    "INSECTS_Abr": {
        "usp_file": "abrupt_balanced",
        "drift_type": "abrupt",
        # Five known change points (0-indexed stream positions).
        "drift_points": [7800, 15200, 23600, 31800, 40100],
        "warm_up": 2000,
    },
    "INSECTS_Incr": {
        "usp_file": "incremental_balanced",
        "drift_type": "incremental",
        "drift_points": None,
        "warm_up": 2000,
    },
    "INSECTS_IncrGrd": {
        "usp_file": "gradual_balanced",
        "drift_type": "incremental+gradual",
        "drift_points": None,
        "warm_up": 2000,
    },
    "INSECTS_IncrRecr": {
        "usp_file": "incremental-reoccurring_balanced",
        "drift_type": "incremental+reoccurring",
        "drift_points": None,
        "warm_up": 2000,
    },
}

METHODS = [
    "NOMAD", "NOMAD-P", "Random", "AutoSAD", "SEAD",
    "HST", "IFASD", "LODA", "OIF", "RRCF", "RSHash", "SIF", "XStream",
]


def create_model(model_name, feature_mins, feature_maxes, seed):
    """Create a detector with the same settings as its benchmark runner."""
    if model_name in {"NOMAD", "NOMAD-P", "Random"}:
        from NOMAD.nomad import NOMAD

        settings = {
            "random_state": seed,
            "verbose": False,
            "feature_mins": feature_mins,
            "feature_maxes": feature_maxes,
            "n_pool_models": 10,
            "n_candidate_configs": 100,
            "evolution_interval": 1000,
            "shared_dim": 32,
            "head_hidden_dim": 16,
            "K": 20,
            "diversity_weight": 0.3,
            "feature_window_size": 100,
            "score_max_history": 1000,
            "reward_min_updates": 10,
            "meta_lr": 0.01,
            "meta_momentum": 0.9,
            "drift_delta": 0.005,
            "drift_threshold": 50.0,
            "drift_alpha": 0.9999,
        }
        if model_name == "NOMAD-P":
            settings["meta_arch"] = "perceptron"
        elif model_name == "Random":
            settings["selection_mode"] = "ensemble_avg"
        return NOMAD(**settings)

    if model_name == "HST":
        from pysad_rust import HalfSpaceTrees
        return HalfSpaceTrees(
            feature_mins=feature_mins, feature_maxes=feature_maxes,
            window_size=250, num_trees=32, max_depth=15, random_state=seed,
        )
    if model_name == "SIF":
        from pysad_rust import StreamingIsolationForest
        return StreamingIsolationForest(n_trees=100, window_size=256, height=None)
    if model_name == "RRCF":
        from pysad_rust import RobustRandomCutForest
        return RobustRandomCutForest(num_trees=32, shingle_size=1, tree_size=256, random_state=seed)
    if model_name == "OIF":
        from pysad_rust import OnlineIsolationForest
        return OnlineIsolationForest(
            branching_factor=2, split="axisparallel", num_trees=32,
            max_leaf_samples=32, window_size=2048, growth_criterion="adaptive",
            subsample=1.0, random_state=seed,
        )
    # The four detectors below use the same pysad_rust classes and settings as
    # their benchmark runners (scripts/{loda,xstream,rshash,ifasd}_run.py).
    if model_name == "LODA":
        from pysad_rust import LODA
        return LODA(num_bins=100, num_random_cuts=32, random_state=seed)
    if model_name == "XStream":
        from pysad_rust import XStream
        return XStream(n_components=8, n_chains=8, depth=8, random_state=seed)
    if model_name == "RSHash":
        from pysad_rust import RSHash
        return RSHash(decay=0.015, num_components=100, num_hash_fns=1, sampling_points=1000,
                      feature_mins=feature_mins, feature_maxes=feature_maxes)
    if model_name == "IFASD":
        from pysad_rust import IForestASD
        return IForestASD(window_size=2048, n_estimators=32, max_samples=256, contamination=0.1,
                          anomaly_rate_threshold=0.3, random_state=seed)
    if model_name == "AutoSAD":
        from AutoSAD import AutoSAD
        return AutoSAD(feature_mins=feature_mins, feature_maxes=feature_maxes, random_state=seed, verbose=False)
    if model_name == "SEAD":
        from SEAD.sead import SEAD
        return SEAD(feature_mins=feature_mins, feature_maxes=feature_maxes, eta=1.0, lam=1e-6, random_state=seed)
    raise ValueError(f"Unsupported model: {model_name}")


def windowed_auc(scores, labels, window):
    """Return cumulative AUC evaluated at each window boundary."""
    out = []
    for n in range(window, len(scores) + 1, window):
        yl = labels[:n]
        if 0 < yl.sum() < n:
            out.append(float(roc_auc_score(yl, scores[:n])))
        else:
            out.append(0.5)
    return out


def compute_recovery(scores, labels, stream_meta, window=1000):
    """Recovery = instances to return within 95% of the pre-drift AUC.

    For the abrupt stream we measure against each known change point. For the
    distributed-drift streams we measure against a warm-up baseline.
    """
    baseline_end = stream_meta["warm_up"]
    if baseline_end >= len(scores) or 0 < labels[:baseline_end].sum() < baseline_end:
        baseline_auc = float(roc_auc_score(labels[:baseline_end], scores[:baseline_end]))
    else:
        baseline_auc = 0.5

    thresholds = stream_meta["drift_points"] or [baseline_end]

    recoveries = []
    for start in thresholds:
        target = baseline_auc * 0.95
        recovered = None
        for end in range(start, len(scores) + 1, window):
            if 0 < labels[start:end].sum() < (end - start):
                auc = float(roc_auc_score(labels[start:end], scores[start:end]))
            else:
                auc = 0.5
            if auc >= target:
                recovered = end - start
                break
        recoveries.append(recovered if recovered is not None else len(scores) - start)

    return {
        "baseline_auc": round(baseline_auc, 4),
        "recovery_instances_mean": round(float(np.mean(recoveries)), 1),
        "recovery_instances_per_drift": recoveries,
    }


def run_stream(stream_name, model_name, seed, window, output_dir):
    """Run one model on one INSECTS stream and write its JSON result."""
    random.seed(seed)
    np.random.seed(seed)

    npz_path = CODE_DIR / "datasets" / f"{stream_name}.npz"
    X, y = get_xy_from_npz(str(npz_path))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    feature_mins = X.min(axis=0).astype(np.float64)
    feature_maxes = X.max(axis=0).astype(np.float64)

    detector = create_model(model_name, feature_mins, feature_maxes, seed)
    scores, labels = [], []
    start_time = time.time()

    for xi, yi in ArrayStreamer(shuffle=False).iter(X, y):
        xi = np.ascontiguousarray(xi, dtype=np.float64)
        scores.append(float(detector.fit_score_partial(xi)))
        labels.append(int(yi))

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    auc = float(roc_auc_score(labels, scores))
    precision_k = compute_precision_at_k(list(scores), list(labels))
    precision_k_scores = compute_precision_at_k_windowed(list(scores), list(labels), window)
    auc_scores = windowed_auc(scores, labels, window)
    recovery = compute_recovery(scores, labels, STREAMS[stream_name], window)

    result = {
        "dataset": stream_name,
        "usp_file": STREAMS[stream_name]["usp_file"],
        "drift_type": STREAMS[stream_name]["drift_type"],
        "n_instances": int(len(scores)),
        "n_anomalies": int(labels.sum()),
        "model": model_name,
        "auc": auc,
        "precision_at_k": precision_k,
        "recovery": recovery,
        "window_metric": {
            "window_size": window,
            "auc_scores": auc_scores,
            "precision_at_k_scores": precision_k_scores,
        },
        "run_count": 1,
        "seed": seed,
        "total_runtime": round(time.time() - start_time, 4),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}_{stream_name}_{seed}.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run real INSECTS concept-drift experiments")
    parser.add_argument("--streams", nargs="+", default=list(STREAMS))
    parser.add_argument("--models", nargs="+", default=METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "results" / "concept_drift_insects")
    args = parser.parse_args()

    for stream in args.streams:
        for model in args.models:
            result = run_stream(stream, model, args.seed, args.window, args.output_dir)
            print(f"{model} on {stream}: AUC {result['auc']:.4f}, "
                  f"recovery {result['recovery']['recovery_instances_mean']} inst.")


if __name__ == "__main__":
    main()
