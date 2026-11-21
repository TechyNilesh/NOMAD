import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import random
import time
import psutil
import json
import argparse
from sklearn.utils import shuffle
from pysad.evaluation import AUROCMetric
from pysad.utils import ArrayStreamer
from scripts.utils import get_xy_from_npz, compute_precision_at_k, compute_precision_at_k_windowed
from NOMAD.nomad import NOMAD

import warnings
warnings.filterwarnings("ignore")

DATASET_LIST = [
    '8_celeba', '33_skin', '34_smtp', '11_donors', '5_campaign',
    '10_cover', '3_backdoor', '16_http', '32_shuttle', '13_fraud',
    '9_census', '1_ALOI', '48_chess', '49_kddcup99_prob',
    '50_bank', '51_kddcup99_u2r',
]

NOMAD_CONFIG = {
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

DETECTORS = [
    ("page_hinkley", "Page-Hinkley (NOMAD default)"),
    ("adwin",        "Adaptive Windowing"),
    ("kswin",        "Kolmogorov-Smirnov Windowing"),
    ("ddm",          "Drift Detection Method"),
    ("no_detection", "No drift detection"),
]


def run_detector(
    dataset_name: str,
    detector_type: str,
    seed: int = 42,
    output_dir: str = "Drift_Detector_Ablation",
    verbose: bool = True,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    config = dict(NOMAD_CONFIG)
    if detector_type == "no_detection":
        config["enable_drift_detection"] = False
    else:
        config["drift_detector_type"] = detector_type

    print(f"\n{'='*70}")
    print(f"Detector: {detector_type}  |  Dataset: {dataset_name}  |  Seed: {seed}")
    print(f"{'='*70}")

    try:
        X, y = get_xy_from_npz("../datasets/" + dataset_name + ".npz")
        X, y = shuffle(X, y, random_state=seed)
        X = np.asarray(X, dtype=np.float64)

        feature_mins = X.min(axis=0).astype(np.float64)
        feature_maxes = X.max(axis=0).astype(np.float64)

        if verbose:
            print(f"  Instances: {X.shape[0]}, Features: {X.shape[1]}, Anomaly ratio: {y.sum()/len(y):.4f}")

        model = NOMAD(
            random_state=seed,
            verbose=False,
            feature_mins=feature_mins,
            feature_maxes=feature_maxes,
            **config
        )

        processed = 0
        metric = AUROCMetric()
        iterator = ArrayStreamer(shuffle=False)

        all_scores = []
        all_labels = []
        auc_scores = []
        runtimes = []
        memory_usages = []

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        for xi, yi in iterator.iter(X, y):
            xi = np.asarray(xi, dtype=np.float64)
            xi = np.ascontiguousarray(xi)
            score = model.fit_score_partial(xi)
            metric.update(yi, score)
            all_scores.append(float(score))
            all_labels.append(float(yi))
            processed += 1

            if processed % 1000 == 0:
                try:
                    current_auc = metric.get()
                except Exception:
                    current_auc = 0.5
                auc_scores.append(current_auc)
                elapsed = time.time() - start_time
                runtimes.append(elapsed)
                mem = abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024)
                memory_usages.append(mem)
                if verbose:
                    print(f"  [{detector_type}] {processed} inst, AUROC: {current_auc:.4f}, Time: {elapsed:.2f}s")

        precision_k = compute_precision_at_k(all_scores, all_labels)
        precision_k_scores = compute_precision_at_k_windowed(all_scores, all_labels, 1000)
        final_auc = metric.get()
        total_time = time.time() - start_time
        total_mem = abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024)

        drift_events = model.drift_count if hasattr(model, 'drift_count') else 0

        results = {
            "dataset": dataset_name,
            "model": "NOMAD",
            "ablation_study": {
                "component": f"drift_detector_{detector_type}",
                "description": f"Drift detector: {detector_type}",
                "configuration": {
                    **config,
                    "feature_mins": None,
                    "feature_maxes": None,
                },
            },
            "drift_detector": {
                "type": detector_type,
                "name": DETECTORS[0][1] if detector_type == "page_hinkley"
                        else DETECTORS[1][1] if detector_type == "adwin"
                        else DETECTORS[2][1] if detector_type == "kswin"
                        else DETECTORS[3][1] if detector_type == "ddm"
                        else "No drift detection",
                "drift_events": drift_events,
            },
            "auc": final_auc,
            "total_runtime": total_time,
            "total_memory_usage": total_mem,
            "Instances": processed,
            "run_count": 1,
            "seed": seed,
            "window_metric": {
                "window_size": 1000,
                "auc_scores": auc_scores,
                "precision_at_k_scores": precision_k_scores,
                "runtimes": runtimes,
                "memory_usages": memory_usages,
            },
            "precision_at_k": precision_k,
        }

        fname = f"NOMAD_{detector_type}_{dataset_name}_1.json"
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"  Complete: AUROC={final_auc:.4f}, Time={total_time:.2f}s, Drifts={drift_events}")
        print(f"  Saved: {out_path}")
        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_ablation(
    output_dir: str = "Drift_Detector_Ablation",
    datasets: list = None,
    seed: int = 42,
    verbose: bool = True,
):
    datasets = datasets or DATASET_LIST
    print("=" * 70)
    print("DRIFT DETECTOR ABLATION STUDY")
    print("=" * 70)
    print(f"Detectors: {[d[0] for d in DETECTORS]}")
    print(f"Datasets: {len(datasets)}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    results = []
    failures = 0

    for det_type, det_desc in DETECTORS:
        for ds in datasets:
            r = run_detector(
                dataset_name=ds,
                detector_type=det_type,
                seed=seed,
                output_dir=output_dir,
                verbose=verbose,
            )
            if r:
                results.append(r)
            else:
                failures += 1

    summary = {
        "experiment": "NOMAD Drift Detector Ablation Study",
        "datasets": {ds: ds for ds in datasets},
        "detectors": {d[0]: {"name": d[1]} for d in DETECTORS},
        "n_runs": 1,
        "seed": seed,
        "summary_by_detector": {},
        "all_results": [],
    }

    for det_type, det_desc in DETECTORS:
        aucs = [r["auc"] for r in results if r["ablation_study"]["component"] == f"drift_detector_{det_type}"]
        runtimes = [r["total_runtime"] for r in results if r["ablation_study"]["component"] == f"drift_detector_{det_type}"]
        drifts = [r["drift_detector"]["drift_events"] for r in results if r["ablation_study"]["component"] == f"drift_detector_{det_type}"]

        summary["summary_by_detector"][det_type] = {
            "auc_mean": float(np.mean(aucs)) if aucs else 0,
            "auc_std": float(np.std(aucs)) if len(aucs) > 1 else 0.0,
            "runtime_mean": float(np.mean(runtimes)) if runtimes else 0,
            "drift_count_mean": float(np.mean(drifts)) if drifts else 0,
        }
        for r in results:
            if r["ablation_study"]["component"] == f"drift_detector_{det_type}":
                summary["all_results"].append({
                    "dataset": r["dataset"],
                    "detector": det_type,
                    "detector_name": det_desc,
                    "final_auc": r["auc"],
                    "runtime": r["total_runtime"],
                    "memory": r["total_memory_usage"],
                    "drift_events": r["drift_detector"]["drift_events"],
                })

    summary_path = os.path.join(output_dir, "drift_detector_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for det_type, _ in DETECTORS:
        s = summary["summary_by_detector"][det_type]
        print(f"  {det_type:20s}  AUC={s['auc_mean']:.4f}  Runtime={s['runtime_mean']:.1f}s  Drifts={s['drift_count_mean']:.1f}")
    print(f"\nProcessed: {len(results)}, Failed: {failures}")
    print(f"Summary: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOMAD Drift Detector Ablation Study")
    parser.add_argument("--output_dir", type=str, default="../results/drift_detector_ablation_full")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else None
    run_ablation(
        output_dir=args.output_dir,
        datasets=datasets,
        seed=args.seed,
        verbose=args.verbose,
    )
