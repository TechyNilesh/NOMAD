"""
Sensitivity Analysis for NOMAD Parameters (Real Execution)

This script performs sensitivity analysis on key NOMAD parameters using the real model:
- n_pool_models: Number of models in the pool
- meta_lr: Learning rate for meta-learner
- evolution_interval: Steps between evolution events
- diversity_weight: Weight for diversity penalty
- enable_drift_detection: Enable/disable drift detection

Results are saved in JSON format for analysis.

Usage:
    python sensitivity_analysis_nomad.py --output_dir "Sensitivity_Analysis_NOMAD"
    python sensitivity_analysis_nomad.py --dataset 48_chess --param n_pool_models
    python sensitivity_analysis_nomad.py --param all --runs 3
"""

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

# List of dataset names (same 16 datasets)
DATASET_LIST = [
    '8_celeba',
    '33_skin',
    '34_smtp',
    '11_donors',
    '5_campaign',
    '10_cover',
    '3_backdoor',
    '16_http',
    '32_shuttle',
    '13_fraud',
    '9_census',
    '1_ALOI',
    '48_chess',
    '49_kddcup99_prob',
    '50_bank',
    '51_kddcup99_u2r',
]

# Default NOMAD parameters
DEFAULT_PARAMS = {
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
    'efficiency_weight': 0.2,
    'meta_lr': 0.01,
    'meta_momentum': 0.9,
    'enable_drift_detection': True,
    'drift_delta': 0.005,
    'drift_threshold': 50.0,
    'drift_alpha': 0.9999,
}

# Parameter variations for sensitivity analysis
PARAMETER_VARIATIONS = {
    'n_pool_models': [5, 10, 15, 20, 25],
    'meta_lr': [0.001, 0.005, 0.01, 0.05, 0.1],
    'evolution_interval': [500, 1000, 2000, 5000],
    'diversity_weight': [0.0, 0.1, 0.3, 0.5],
    'enable_drift_detection': [True, False],
}


def run_sensitivity_experiment(
    dataset_name: str,
    param_name: str,
    param_value,
    run_count: int,
    seed: int,
    progress_interval: int = 1000,
    output_dir: str = "Sensitivity_Analysis_NOMAD",
    verbose: bool = True
) -> dict:
    """
    Run sensitivity experiment for a specific parameter value.

    Args:
        dataset_name: Name of the dataset
        param_name: Name of the parameter being varied
        param_value: Value of the parameter
        run_count: Run identifier
        seed: Random seed
        progress_interval: Interval for metrics
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dictionary containing results
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create config with modified parameter
    config = DEFAULT_PARAMS.copy()
    config[param_name] = param_value

    print(f"\n{'='*70}")
    print(f"Sensitivity Analysis: {param_name} = {param_value}")
    print(f"Dataset: {dataset_name}, Run: {run_count}, Seed: {seed}")
    print(f"{'='*70}")

    try:
        # Load dataset
        X, y = get_xy_from_npz("../datasets/" + dataset_name + ".npz")
        X, y = shuffle(X, y, random_state=seed)
        X = np.asarray(X, dtype=np.float64)

        # Compute feature bounds
        feature_mins = X.min(axis=0).astype(np.float64)
        feature_maxes = X.max(axis=0).astype(np.float64)

        if verbose:
            print(f"Dataset: {X.shape[0]} instances, {X.shape[1]} features")

        # Initialize NOMAD
        model = NOMAD(
            random_state=seed,
            verbose=False,
            feature_mins=feature_mins,
            feature_maxes=feature_maxes,
            **config
        )

        # Setup evaluation
        processed_instances = 0
        metric = AUROCMetric()
        iterator = ArrayStreamer(shuffle=False)

        # Track metrics
        auc_scores = []
        all_scores = []
        all_labels = []
        runtimes = []
        memory_usages = []

        # Start timing
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        # Process stream
        for xi, yi in iterator.iter(X, y):
            xi = np.asarray(xi, dtype=np.float64)
            xi = np.ascontiguousarray(xi)

            # Fit and score
            score = model.fit_score_partial(xi)
            metric.update(yi, score)
            all_scores.append(float(score))
            all_labels.append(float(yi))

            processed_instances += 1

            if processed_instances % progress_interval == 0:
                try:
                    current_auroc = metric.get()
                except:
                    current_auroc = 0.5
                auc_scores.append(current_auroc)

                elapsed_time = time.time() - start_time
                runtimes.append(elapsed_time)

                memory_used = abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024)
                memory_usages.append(memory_used)

                if verbose:
                    print(f"  [{param_name}={param_value}] {processed_instances} instances, "
                          f"AUROC: {current_auroc:.4f}, Time: {elapsed_time:.2f}s")

        # Final metrics
        precision_k = compute_precision_at_k(all_scores, all_labels)
        precision_k_scores = compute_precision_at_k_windowed(all_scores, all_labels, progress_interval)
        final_auroc = metric.get()
        total_runtime = time.time() - start_time
        total_memory = abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024)

        # Create results
        results = {
            "dataset": dataset_name,
            "model": "NOMAD",
            "sensitivity_analysis": {
                "parameter": param_name,
                "value": param_value,
                "default_value": DEFAULT_PARAMS[param_name]
            },
            "auc": final_auroc,
            "precision_at_k": precision_k,
            "total_runtime": total_runtime,
            "total_memory_usage": total_memory,
            "Instances": processed_instances,
            "run_count": run_count,
            "seed": seed,
            "window_metric": {
                "window_size": progress_interval,
                "auc_scores": auc_scores,
                "precision_at_k_scores": precision_k_scores,
                "runtimes": runtimes,
                "memory_usages": memory_usages
            }
        }

        # Save results
        param_str = str(param_value).replace('.', '_')
        file_name = f"NOMAD_{param_name}_{param_str}_{dataset_name}_{run_count}.json"
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\nCompleted: AUROC={final_auroc:.4f}, Runtime={total_runtime:.2f}s")
        print(f"Saved to: {output_file}")

        return results

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_sensitivity_analysis(
    param_name: str,
    output_dir: str = "Sensitivity_Analysis_NOMAD",
    progress_interval: int = 1000,
    datasets: list = None,
    runs_per_config: int = 3,
    verbose: bool = True
):
    """
    Run sensitivity analysis for a specific parameter.

    Args:
        param_name: Name of the parameter to analyze
        output_dir: Directory to save results
        progress_interval: Interval for metrics
        datasets: List of datasets (None = all)
        runs_per_config: Number of runs per configuration
        verbose: Print progress
    """
    if param_name not in PARAMETER_VARIATIONS:
        print(f"Error: Unknown parameter '{param_name}'")
        print(f"Available: {list(PARAMETER_VARIATIONS.keys())}")
        return

    param_values = PARAMETER_VARIATIONS[param_name]
    datasets_to_use = datasets if datasets else DATASET_LIST

    print("=" * 70)
    print(f"SENSITIVITY ANALYSIS: {param_name}")
    print("=" * 70)
    print(f"Values to test: {param_values}")
    print(f"Default value: {DEFAULT_PARAMS[param_name]}")
    print(f"Datasets: {len(datasets_to_use)}")
    print(f"Runs per config: {runs_per_config}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    total_generated = 0
    total_failed = 0

    for param_value in param_values:
        print(f"\n{'='*70}")
        print(f"Testing {param_name} = {param_value}")
        print(f"{'='*70}")

        for dataset_name in datasets_to_use:
            for run_count in range(1, runs_per_config + 1):
                seed = 42 + run_count

                result = run_sensitivity_experiment(
                    dataset_name=dataset_name,
                    param_name=param_name,
                    param_value=param_value,
                    run_count=run_count,
                    seed=seed,
                    progress_interval=progress_interval,
                    output_dir=output_dir,
                    verbose=verbose
                )

                if result:
                    total_generated += 1
                else:
                    total_failed += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Parameter: {param_name}")
    print(f"Values tested: {param_values}")
    print(f"Generated: {total_generated}, Failed: {total_failed}")
    print(f"Results: {output_dir}")
    print("=" * 70)


def run_all_sensitivity_analyses(
    output_dir: str = "Sensitivity_Analysis_NOMAD",
    progress_interval: int = 1000,
    datasets: list = None,
    runs_per_config: int = 3,
    verbose: bool = True
):
    """Run sensitivity analysis for all parameters."""
    print("=" * 70)
    print("COMPLETE SENSITIVITY ANALYSIS FOR ALL PARAMETERS")
    print("=" * 70)

    for param_name in PARAMETER_VARIATIONS.keys():
        run_sensitivity_analysis(
            param_name=param_name,
            output_dir=output_dir,
            progress_interval=progress_interval,
            datasets=datasets,
            runs_per_config=runs_per_config,
            verbose=verbose
        )
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sensitivity Analysis for NOMAD (Real Execution)')
    parser.add_argument('--output_dir', type=str, default="Sensitivity_Analysis_NOMAD",
                        help='Output directory')
    parser.add_argument('--progress_interval', type=int, default=1000,
                        help='Progress interval')
    parser.add_argument('--param', type=str, default=None,
                        choices=list(PARAMETER_VARIATIONS.keys()) + ['all'],
                        help='Parameter to analyze (default: all)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset (optional)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Runs per configuration')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')

    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else None

    if datasets and datasets[0] not in DATASET_LIST:
        print(f"Error: Dataset '{args.dataset}' not found")
        exit(1)

    if args.param is None or args.param == 'all':
        run_all_sensitivity_analyses(
            output_dir=args.output_dir,
            progress_interval=args.progress_interval,
            datasets=datasets,
            runs_per_config=args.runs,
            verbose=args.verbose
        )
    else:
        run_sensitivity_analysis(
            param_name=args.param,
            output_dir=args.output_dir,
            progress_interval=args.progress_interval,
            datasets=datasets,
            runs_per_config=args.runs,
            verbose=args.verbose
        )
