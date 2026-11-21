import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import random
import time
import json
import argparse
from sklearn.utils import shuffle
from pysad.evaluation import AUROCMetric
from pysad.utils import ArrayStreamer
from scripts.utils import get_xy_from_npz
from NOMAD.nomad import NOMAD

import warnings
warnings.filterwarnings("ignore")

DATASETS = ['1_ALOI', '48_chess', '34_smtp', '5_campaign', '3_backdoor']

# Diverse pool (default) vs monoculture pools (single family).
VARIANTS = {
    "diverse": None,
    "mono_hst": "HalfSpaceTrees",
    "mono_loda": "LODA",
    "mono_oif": "OnlineIsolationForest",
    "mono_sif": "StreamingIsolationForest",
}

CONFIG = {
    'n_pool_models': 10,
    'n_candidate_configs': 100,
    'evolution_interval': 1000,
    'shared_dim': 32,
    'head_hidden_dim': 16,
    'K': 20,
    'diversity_weight': 0.3,
    'feature_window_size': 100,
    'score_max_history': 1000,
    'reward_min_updates': 10,
    'meta_lr': 0.01,
    'meta_momentum': 0.9,
    'enable_drift_detection': True,
    'drift_delta': 0.005,
    'drift_threshold': 50.0,
    'drift_alpha': 0.9999,
}


def run_variant(dataset, pool_family, seed, out_dir, max_samples=None):
    random.seed(seed)
    np.random.seed(seed)

    X, y = get_xy_from_npz("../datasets/" + dataset + ".npz")
    X, y = shuffle(X, y, random_state=seed)
    X = np.asarray(X, dtype=np.float64)
    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    feature_mins = X.min(axis=0).astype(np.float64)
    feature_maxes = X.max(axis=0).astype(np.float64)

    params = dict(CONFIG)
    if pool_family is not None:
        params["pool_family"] = pool_family

    model = NOMAD(
        random_state=seed,
        verbose=False,
        feature_mins=feature_mins,
        feature_maxes=feature_maxes,
        **params
    )

    metric = AUROCMetric()
    iterator = ArrayStreamer(shuffle=False)
    n = 0
    for xi, yi in iterator.iter(X, y):
        xi = np.ascontiguousarray(np.asarray(xi, dtype=np.float64))
        score = model.fit_score_partial(xi)
        metric.update(yi, score)
        n += 1

    auc = metric.get()
    # final pool composition
    pool = [algo for algo, _, _ in model.model_configs]
    result = {
        "dataset": dataset,
        "variant": pool_family if pool_family else "diverse",
        "auc": float(auc),
        "final_pool_families": sorted(set(pool)),
        "n_families_in_pool": len(set(pool)),
        "seed": seed,
    }
    os.makedirs(out_dir, exist_ok=True)
    tag = pool_family or "diverse"
    with open(os.path.join(out_dir, f"collapse_{tag}_{dataset}_{seed}.json"), "w") as f:
        json.dump(result, f, indent=4)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_dir", default="../results/consensus_collapse")
    args = parser.parse_args()

    all_results = []
    for ds in args.datasets:
        for name in args.variants:
            family = VARIANTS[name]
            t0 = time.time()
            r = run_variant(ds, family, args.seed, args.out_dir, args.max_samples)
            r["variant"] = name
            r["runtime_s"] = round(time.time() - t0, 1)
            all_results.append(r)
            print(f"{ds:20s} {name:10s} AUC={r['auc']:.4f}  pool_families={r['n_families_in_pool']}  ({r['runtime_s']}s)")

    summary = {}
    for name in args.variants:
        aucs = [r["auc"] for r in all_results if r["variant"] == name]
        summary[name] = {"mean_auc": round(float(np.mean(aucs)), 4), "datasets": len(aucs)}
    with open(os.path.join(args.out_dir, "consensus_collapse_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print("\nSummary:")
    for name, s in summary.items():
        print(f"  {name:10s} mean AUC = {s['mean_auc']}")
