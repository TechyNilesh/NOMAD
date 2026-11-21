"""Run a real prequential detection-delay experiment.

The script injects non-overlapping bursts made from observed anomaly instances,
then measures each model's first threshold crossing within each burst using
scores produced by real detector execution.
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


DATASETS = ("10_cover", "13_fraud")
MODELS = (
    "NOMAD", "HST", "LODA", "SIF", "XStream", "RRCF", "RSHash",
    "IFASD", "OIF", "AutoSAD", "SEAD", "RandomSearchEnsemble",
    "NOMAD-Perceptron",
)


def select_burst_positions(n_instances, burst_length, n_bursts, seed):
    """Choose reproducible, non-overlapping burst starts."""
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(n_instances - burst_length)
    positions = []
    for candidate in candidates:
        if all(abs(candidate - position) >= burst_length for position in positions):
            positions.append(int(candidate))
            if len(positions) == n_bursts:
                return sorted(positions)
    raise ValueError("Cannot place the requested number of non-overlapping bursts")


def create_model(model_name, feature_mins, feature_maxes, seed):
    """Create a detector with the same settings as its benchmark runner."""
    if model_name in {"NOMAD", "NOMAD-Perceptron", "RandomSearchEnsemble"}:
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
        if model_name == "NOMAD-Perceptron":
            settings["meta_arch"] = "perceptron"
        elif model_name == "RandomSearchEnsemble":
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
    if model_name == "LODA":
        from pysad.models import LODA
        return LODA()
    if model_name == "XStream":
        from pysad.models import xStream
        return xStream()
    if model_name == "RSHash":
        from pysad.models import RSHash
        return RSHash()
    if model_name == "IFASD":
        from pysad.models import IForestASD
        return IForestASD()
    if model_name == "AutoSAD":
        from AutoSAD import AutoSAD
        return AutoSAD(feature_mins=feature_mins, feature_maxes=feature_maxes, random_state=seed, verbose=False)
    if model_name == "SEAD":
        from SEAD.sead import SEAD
        return SEAD(feature_mins=feature_mins, feature_maxes=feature_maxes, eta=1.0, lam=1e-6, random_state=seed)
    raise ValueError(f"Unsupported model: {model_name}")


def run_detection_delay_experiment(dataset, model_name, seed, burst_length, n_bursts, fpr, output_dir):
    """Execute one model-dataset-seed experiment and write its JSON result."""
    random.seed(seed)
    np.random.seed(seed)

    with np.load(CODE_DIR / "datasets" / f"{dataset}.npz", allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=int)

    anomaly_indices = np.flatnonzero(y == 1)
    if not len(anomaly_indices):
        raise ValueError(f"{dataset} has no anomaly instances to use as burst templates")

    positions = select_burst_positions(len(X), burst_length, n_bursts, seed)
    rng = np.random.default_rng(seed)
    X_stream = X.copy()
    y_stream = y.copy()
    burst_ids = np.full(len(X), -1, dtype=int)
    for burst_id, start in enumerate(positions):
        template = X[rng.choice(anomaly_indices)]
        end = start + burst_length
        X_stream[start:end] = template
        y_stream[start:end] = 1
        burst_ids[start:end] = burst_id

    detector = create_model(model_name, X_stream.min(axis=0), X_stream.max(axis=0), seed)
    normal_scores = []
    scores = []
    labels = []
    burst_records = [None] * n_bursts
    start_time = time.time()

    for index, (xi, yi) in enumerate(ArrayStreamer(shuffle=False).iter(X_stream, y_stream)):
        burst_id = burst_ids[index]
        if burst_id >= 0 and burst_records[burst_id] is None:
            if not normal_scores:
                raise RuntimeError("A burst starts before any normal scores are available")
            burst_records[burst_id] = {
                "burst_id": int(burst_id),
                "start_position": int(index),
                "detected": False,
                "delay": None,
                "n_instances_scored": burst_length,
                "score_threshold": float(np.quantile(normal_scores, 1.0 - fpr)),
            }

        score = float(detector.fit_score_partial(np.ascontiguousarray(xi, dtype=np.float64)))
        scores.append(score)
        labels.append(int(yi))

        if burst_id >= 0:
            record = burst_records[burst_id]
            if not record["detected"] and score >= record["score_threshold"]:
                record["detected"] = True
                record["delay"] = int(index - record["start_position"])
        elif yi == 0:
            normal_scores.append(score)

    details = burst_records
    detected_delays = [record["delay"] for record in details if record["detected"]]
    all_delays = [record["delay"] if record["detected"] else burst_length for record in details]
    n_detected = len(detected_delays)
    result = {
        "dataset": dataset,
        "model": model_name,
        "analysis": "detection_delay",
        "auc": float(roc_auc_score(labels, scores)),
        "detection_delay": {
            "mean_delay": round(float(np.mean(detected_delays)), 4) if detected_delays else None,
            "median_delay": round(float(np.median(detected_delays)), 4) if detected_delays else None,
            "std_delay": round(float(np.std(detected_delays)), 4) if detected_delays else 0.0,
            "mean_delay_all": round(float(np.mean(all_delays)), 4),
            "detection_rate": round(n_detected / n_bursts, 4),
            "n_bursts": n_bursts,
            "n_detected": n_detected,
            "n_missed": n_bursts - n_detected,
            "burst_length": burst_length,
            "fpr": fpr,
            "threshold_method": "pre-burst normal-score quantile",
        },
        "burst_details": details,
        "Instances": len(X_stream),
        "seed": seed,
        "total_runtime": round(time.time() - start_time, 4),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{model_name}_{dataset}_{seed}.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run real prequential detection-delay experiments")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--burst-length", type=int, default=50)
    parser.add_argument("--n-bursts", type=int, default=20)
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "results" / "detection_delay_analysis")
    args = parser.parse_args()

    for dataset in args.datasets:
        for model_name in args.models:
            for seed in range(42, 42 + args.runs):
                result = run_detection_delay_experiment(
                    dataset, model_name, seed, args.burst_length, args.n_bursts, args.fpr, args.output_dir,
                )
                delay = result["detection_delay"]
                print(f"{model_name} on {dataset}, seed {seed}: "
                      f"detected {delay['n_detected']}/{delay['n_bursts']}, "
                      f"mean delay {delay['mean_delay']}")


if __name__ == "__main__":
    main()
