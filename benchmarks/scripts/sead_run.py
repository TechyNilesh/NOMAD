import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import random
import time
import psutil
import json
import argparse
from sklearn.utils import shuffle
from pysad.evaluation import AUROCMetric
from pysad.utils import ArrayStreamer
from utils import get_xy_from_npz, compute_precision_at_k, compute_precision_at_k_windowed
from SEAD.sead import SEAD

import warnings
warnings.filterwarnings("ignore")


def run_sead(dataset_name, run_count=None, seed=42, progress_interval=1000,
             output_dir="benchmark_results", verbose=True,
             eta=1.0, lam=1e-6):
    """Run SEAD on a dataset."""

    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===Running SEAD on {dataset_name}===")

    try:
        X, y = get_xy_from_npz("../../datasets/" + dataset_name + ".npz")
        X, y = shuffle(X, y, random_state=seed)
        X = np.asarray(X, dtype=np.float64)

        feature_mins = X.min(axis=0).astype(np.float64)
        feature_maxes = X.max(axis=0).astype(np.float64)

        if verbose:
            print(f"Dataset: {X.shape[0]} instances, {X.shape[1]} features")
            print(f"Anomaly ratio: {y.sum() / len(y):.4f}")

        model = SEAD(
            feature_mins=feature_mins,
            feature_maxes=feature_maxes,
            eta=eta,
            lam=lam,
            random_state=seed,
        )

        processed_instances = 0
        metric = AUROCMetric()
        iterator = ArrayStreamer(shuffle=False)

        scores = []
        all_scores = []
        all_labels = []
        times = []
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

            processed_instances += 1
            if processed_instances % progress_interval == 0:
                try:
                    current_auroc = metric.get()
                    scores.append(current_auroc)
                except Exception:
                    current_auroc = 0.5
                    scores.append(current_auroc)

                isinstance_time = time.time() - start_time
                times.append(isinstance_time)
                isinstance_memory = psutil.Process().memory_info().rss - start_memory
                memory_usages.append(abs(isinstance_memory / 1024 / 1024))

                if verbose:
                    tag = f"(Run {run_count})" if run_count is not None else ""
                    print(f"SEAD on {dataset_name} {tag}: {processed_instances} instances, "
                          f"AUROC: {current_auroc:.4f}, Time: {isinstance_time:.2f}s", flush=True)

        precision_k = compute_precision_at_k(all_scores, all_labels)
        precision_k_scores = compute_precision_at_k_windowed(all_scores, all_labels, progress_interval)
        final_auroc = metric.get()
        total_runtime = time.time() - start_time
        memory_usage = abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024)

        results = {
            "dataset": dataset_name,
            "model": "SEAD",
            "auc": final_auroc,
            "precision_at_k": precision_k,
            "total_runtime": total_runtime,
            "total_memory_usage": memory_usage,
            "Instances": processed_instances,
            "parameters": {"eta": eta, "lam": lam, "n_detectors": model.N},
            "window_metric": {
                "window_size": progress_interval,
                "auc_scores": scores,
                "precision_at_k_scores": precision_k_scores,
                "runtimes": times,
                "memory_usages": memory_usages,
            }
        }

        if run_count is not None:
            results["run_count"] = run_count
            results["seed"] = seed
            file_name = f"SEAD_{dataset_name}_{run_count}.json"
        else:
            file_name = f"SEAD_{dataset_name}.json"

        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\nSEAD completed on {dataset_name}")
        print(f"Final AUROC: {final_auroc:.4f}")
        print(f"Runtime: {total_runtime:.2f}s, Memory: {memory_usage:.2f}MB")
        print(f"Results saved to {output_file}")

        return results

    except Exception as e:
        print(f"Error running SEAD on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SEAD on a dataset')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--run_count', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--progress_interval', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default="benchmark_results")
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=1e-6)

    args = parser.parse_args()

    run_sead(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=args.verbose,
        eta=args.eta,
        lam=args.lam,
    )
