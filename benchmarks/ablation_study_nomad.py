"""
Ablation Study for NOMAD Components (Real Execution)

This script performs ablation study on key NOMAD components using the real model:
- Drift Detection: Tests impact of removing Page-Hinkley drift detection
- Diversity Weighting: Tests impact of removing diversity in model selection
- Meta-Learning: Tests impact of removing meta-learner (using simple averaging)
- Evolution Mechanism: Tests impact of disabling model evolution
- Feature Windowing: Tests impact of using full history vs windowed features

Results are saved in JSON format for analysis.

Usage:
    python ablation_study_nomad.py --output_dir "Ablation_Study_NOMAD"
    python ablation_study_nomad.py --dataset 48_chess --component no_drift_detection
    python ablation_study_nomad.py --component all --runs 3
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

# Default NOMAD configuration (full model). Must stay in sync with the settings
# reported in the paper's experimental setup, so that the full_model row is
# directly comparable with the main benchmark results.
FULL_CONFIG = {
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

# Ablation configurations - each removes one component
ABLATION_CONFIGS = {
    'full_model': {
        'description': 'Full NOMAD with all components',
        'config': FULL_CONFIG.copy()
    },
    'no_drift_detection': {
        'description': 'Remove drift detection (Page-Hinkley)',
        'config': {**FULL_CONFIG, 'enable_drift_detection': False}
    },
    'no_diversity': {
        'description': 'Remove diversity weighting (diversity_weight=0)',
        'config': {**FULL_CONFIG, 'diversity_weight': 0.0}
    },
    'no_meta_learning': {
        'description': 'Freeze the meta-learner at its initialisation (meta_lr=0), '
                       'so candidate scoring carries no learned signal',
        'config': {**FULL_CONFIG, 'meta_lr': 0.0}
    },
    'no_evolution': {
        'description': 'Disable model evolution (very long interval)',
        'config': {**FULL_CONFIG, 'evolution_interval': 999999}
    },
    'no_feature_windowing': {
        'description': 'Use full feature history instead of windowing',
        'config': {**FULL_CONFIG, 'feature_window_size': 999999}
    },
}


def run_ablation_experiment(
    dataset_name: str,
    component: str,
    run_count: int,
    seed: int,
    progress_interval: int = 1000,
    output_dir: str = "Ablation_Study_NOMAD",
    verbose: bool = True
) -> dict:
    """
    Run ablation experiment for a specific component on a dataset.

    Args:
        dataset_name: Name of the dataset
        component: Name of the component to ablate
        run_count: Run identifier (1-10)
        seed: Random seed for reproducibility
        progress_interval: Interval for window metrics
        output_dir: Directory to save results
        verbose: Print progress details

    Returns:
        Dictionary containing results
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    ablation_config = ABLATION_CONFIGS[component]
    config = ablation_config['config']

    print(f"\n{'='*70}")
    print(f"Ablation Study: {component}")
    print(f"Description: {ablation_config['description']}")
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
            print(f"Anomaly ratio: {y.sum() / len(y):.4f}")

        # Initialize NOMAD with ablation config
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
                    print(f"  [{component}] {processed_instances} instances, "
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
            "ablation_study": {
                "component": component,
                "description": ablation_config['description'],
                "configuration": config
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
        file_name = f"NOMAD_ablation_{component}_{dataset_name}_{run_count}.json"
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


def run_ablation_study(
    component: str,
    output_dir: str = "Ablation_Study_NOMAD",
    progress_interval: int = 1000,
    datasets: list = None,
    runs_per_config: int = 3,
    verbose: bool = True
):
    """
    Run ablation study for a specific component across datasets.

    Args:
        component: Name of the component to ablate
        output_dir: Directory to save results
        progress_interval: Interval for window metrics
        datasets: List of datasets (None = all)
        runs_per_config: Number of runs per configuration
        verbose: Print progress
    """
    if component not in ABLATION_CONFIGS:
        print(f"Error: Unknown component '{component}'")
        print(f"Available: {list(ABLATION_CONFIGS.keys())}")
        return

    datasets_to_use = datasets if datasets else DATASET_LIST

    print("=" * 70)
    print(f"ABLATION STUDY: {component}")
    print("=" * 70)
    print(f"Description: {ABLATION_CONFIGS[component]['description']}")
    print(f"Datasets: {len(datasets_to_use)}")
    print(f"Runs per dataset: {runs_per_config}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    total_generated = 0
    total_failed = 0

    for dataset_name in datasets_to_use:
        for run_count in range(1, runs_per_config + 1):
            seed = 42 + run_count

            result = run_ablation_experiment(
                dataset_name=dataset_name,
                component=component,
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
    print(f"Component: {component}")
    print(f"Generated: {total_generated}, Failed: {total_failed}")
    print(f"Results: {output_dir}")
    print("=" * 70)


def run_all_ablation_studies(
    output_dir: str = "Ablation_Study_NOMAD",
    progress_interval: int = 1000,
    datasets: list = None,
    runs_per_config: int = 3,
    verbose: bool = True
):
    """Run ablation study for all components."""
    print("=" * 70)
    print("COMPLETE ABLATION STUDY FOR ALL COMPONENTS")
    print("=" * 70)

    for component in ABLATION_CONFIGS.keys():
        run_ablation_study(
            component=component,
            output_dir=output_dir,
            progress_interval=progress_interval,
            datasets=datasets,
            runs_per_config=runs_per_config,
            verbose=verbose
        )
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ablation Study for NOMAD (Real Execution)')
    parser.add_argument('--output_dir', type=str, default="Ablation_Study_NOMAD",
                        help='Output directory')
    parser.add_argument('--progress_interval', type=int, default=1000,
                        help='Progress interval')
    parser.add_argument('--component', type=str, default=None,
                        choices=list(ABLATION_CONFIGS.keys()) + ['all'],
                        help='Component to ablate (default: all)')
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

    if args.component is None or args.component == 'all':
        run_all_ablation_studies(
            output_dir=args.output_dir,
            progress_interval=args.progress_interval,
            datasets=datasets,
            runs_per_config=args.runs,
            verbose=args.verbose
        )
    else:
        run_ablation_study(
            component=args.component,
            output_dir=args.output_dir,
            progress_interval=args.progress_interval,
            datasets=datasets,
            runs_per_config=args.runs,
            verbose=args.verbose
        )
