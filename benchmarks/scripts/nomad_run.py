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
from NOMAD.nomad import NOMAD

import warnings
warnings.filterwarnings("ignore")

def run_nomad(
    dataset_name,
    run_count=None,
    seed=42,
    progress_interval=1000,
    output_dir="benchmark_results",
    max_samples=None,
    verbose=True,
    n_pool_models=10,
    evolution_interval=1000,
    shared_dim=32,
    head_hidden_dim=16,
    K=20,
    diversity_weight=0.3,
    n_candidate_configs=100,
    feature_window_size=100,
    score_max_history=1000,
    reward_min_updates=10,
    meta_lr=0.01,
    meta_momentum=0.9,
    # Drift detection parameters (Page-Hinkley for abrupt drift; always on)
    drift_delta=0.005,
    drift_threshold=50.0,
    drift_alpha=0.9999,
    # Variant controls (defaults reproduce the standard NOMAD)
    model_label="NOMAD",
    meta_arch="aamn",
    selection_mode="meta",
    reward_mode="consensus"
):
    """
    Run NOMAD (Neural Online Meta-learning for Automated Anomaly Detection) on a dataset.

    NOMAD uses Algorithm-Aware Meta-Network (AAMN) with:
    - Shared encoder for stream representation
    - Algorithm-specific prediction heads
    """

    # set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===Running NOMAD on {dataset_name}===")

    try:
        # Load dataset
        X, y = get_xy_from_npz("../../datasets/" + dataset_name + ".npz")
        X, y = shuffle(X, y, random_state=seed)

        # Ensure X is float64
        X = np.asarray(X, dtype=np.float64)

        if max_samples is not None and len(X) > max_samples:
            X = X[:max_samples]
            y = y[:max_samples]
            print(f"Subsampled dataset to first {max_samples} samples for quick testing.")

        # Compute feature bounds for HalfSpaceTrees
        feature_mins = X.min(axis=0).astype(np.float64)
        feature_maxes = X.max(axis=0).astype(np.float64)

        if verbose:
            print(f"Dataset loaded: {X.shape[0]} instances, {X.shape[1]} features")
            print(f"Anomaly ratio: {y.sum() / len(y):.4f}")

        # Set NOMAD parameters
        nomad_params = {
            'n_pool_models': n_pool_models,
            'n_candidate_configs': n_candidate_configs,
            'random_state': seed,
            'verbose': False,  # Disable verbose output during benchmark
            'evolution_interval': evolution_interval,
            'shared_dim': shared_dim,
            'head_hidden_dim': head_hidden_dim,
            'K': K,
            'diversity_weight': diversity_weight,
            'feature_window_size': feature_window_size,
            'score_max_history': score_max_history,
            'reward_min_updates': reward_min_updates,
            'meta_lr': meta_lr,
            'meta_momentum': meta_momentum,
            'feature_mins': feature_mins,
            'feature_maxes': feature_maxes,
            # Drift detection (Page-Hinkley for abrupt drift; always enabled)
            'drift_delta': drift_delta,
            'drift_threshold': drift_threshold,
            'drift_alpha': drift_alpha,
            # Variant controls
            'meta_arch': meta_arch,
            'selection_mode': selection_mode,
            'reward_mode': reward_mode,
        }

        if verbose:
            print(f"\nNOMAD Parameters:")
            print(f"  Pool Management:")
            print(f"    n_pool_models: {n_pool_models}")
            print(f"    n_candidate_configs: {n_candidate_configs}")
            print(f"    evolution_interval: {evolution_interval}")
            print(f"  Algorithm-Aware Meta-Network (AAMN):")
            print(f"    shared_dim: {shared_dim}")
            print(f"    head_hidden_dim: {head_hidden_dim}")
            print(f"    K (Top-K sampling): {K}")
            print(f"    diversity_weight: {diversity_weight}")
            print(f"    learning_rate: {meta_lr}")
            print(f"    momentum: {meta_momentum}")
            print(f"  Feature Extraction:")
            print(f"    feature_window_size: {feature_window_size}")
            print(f"    score_max_history: {score_max_history}")
            print(f"  Reward Calculation:")
            print(f"    reward_min_updates: {reward_min_updates}")
            print(f"  Drift Detection (Page-Hinkley, always on):")
            print(f"    drift_delta: {drift_delta}")
            print(f"    drift_threshold: {drift_threshold}")
            print(f"    drift_alpha: {drift_alpha}")

        # Initialize model
        model = NOMAD(**nomad_params)

        # Print parameter count
        if verbose:
            param_info = model.meta_learner.get_parameter_count()
            print(f"\n  Meta-Network Parameters:")
            print(f"    Shared Encoder: {param_info['shared_encoder']}")
            print(f"    Total Head Parameters: {param_info['total_heads']}")
            print(f"    Total Parameters: {param_info['total']}")

        # Setup evaluation
        processed_instances = 0
        metric = AUROCMetric()
        iterator = ArrayStreamer(shuffle=False)

        # Track metrics
        scores = []
        all_scores = []
        all_labels = []
        times = []
        memory_usages = []
        pool_compositions = []
        best_models = []

        # Start timing
        start_time = time.time()

        # start memory usage
        start_memory = psutil.Process().memory_info().rss

        # Process the stream test then train (prequential)
        for xi, yi in iterator.iter(X, y):

            xi = np.asarray(xi, dtype=np.float64)  # Ensure correct dtype
            xi = np.ascontiguousarray(xi)

            # fit and score the model
            score = model.fit_score_partial(xi)
            metric.update(yi, score)
            all_scores.append(float(score))
            all_labels.append(float(yi))

            # record the instance
            processed_instances += 1
            if processed_instances % progress_interval == 0:

                # Calculate AUROC Incrementally
                try:
                    current_auroc = metric.get()
                    scores.append(current_auroc)
                except Exception as e:
                    if verbose:
                        print(f"Error calculating AUROC at instance {processed_instances}: {str(e)}", flush=True)
                    current_auroc = 0.5  # Default AUROC if error occurs
                    scores.append(current_auroc)  # Default AUROC if error occurs

                # Incremental Time and Memory Usage update
                isinstance_time = time.time() - start_time # seconds
                times.append(isinstance_time)
                isinstance_memory = psutil.Process().memory_info().rss - start_memory
                memory_usage = abs(isinstance_memory / 1024 / 1024)  # MB
                memory_usages.append(memory_usage)

                # Track model pool composition
                pool_info = model.get_model_pool_info()
                best_algo, best_hp, best_score = model.get_current_best_model()

                pool_composition = {
                    'step': model.step,
                    'algorithms': [m['algorithm'] for m in pool_info['models']],
                    'best_algorithm': best_algo,
                }
                pool_compositions.append(pool_composition)

                # Convert HP types for JSON serialization
                best_hp_serializable = {}
                for k, v in best_hp.items():
                    if isinstance(v, (np.integer, np.int64, np.int32)):
                        best_hp_serializable[k] = int(v)
                    elif isinstance(v, (np.floating, np.float64, np.float32)):
                        best_hp_serializable[k] = float(v)
                    else:
                        best_hp_serializable[k] = v

                best_models.append({
                    'algorithm': best_algo,
                    'hyperparameters': best_hp_serializable,
                    'mean_score': float(best_score)
                })

                if verbose:
                    if run_count is not None:
                        print(f"NOMAD on {dataset_name} (Run {run_count}): Processed {processed_instances} instances, "
                              f"AUROC: {current_auroc:.4f}, Time: {isinstance_time:.2f} seconds, "
                              f"Memory: {memory_usage:.2f} MB, Best: {best_algo}", flush=True)
                    else:
                        print(f"NOMAD on {dataset_name}: Processed {processed_instances} instances, "
                              f"AUROC: {current_auroc:.4f}, Time: {isinstance_time:.2f} seconds, "
                              f"Memory: {memory_usage:.2f} MB, Best: {best_algo}", flush=True)

        # Calculate final metrics
        precision_k = compute_precision_at_k(all_scores, all_labels)
        precision_k_scores = compute_precision_at_k_windowed(all_scores, all_labels, progress_interval)
        final_auroc = metric.get()
        total_runtime = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss - start_memory
        memory_usage = abs(memory_usage / 1024 / 1024)  # MB

        # Get final pool information
        final_pool_info = model.get_model_pool_info()
        final_best_algo, final_best_hp, final_best_score = model.get_current_best_model()

        # Helper function to convert numpy types to native Python types
        def convert_to_python_types(obj):
            """Recursively convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Store results
        results = {
            "dataset": dataset_name,
            "model": model_label,
            "auc": final_auroc,
            "precision_at_k": precision_k,
            "total_runtime": total_runtime,
            "total_memory_usage": memory_usage,
            "Instances": processed_instances,
            "parameters": {
                "n_pool_models": n_pool_models,
                "n_candidate_configs": n_candidate_configs,
                "evolution_interval": evolution_interval,
                "shared_dim": shared_dim,
                "head_hidden_dim": head_hidden_dim,
                "K": K,
                "diversity_weight": diversity_weight,
                "enable_drift_detection": True,
                "drift_delta": drift_delta,
                "drift_threshold": drift_threshold,
                "drift_alpha": drift_alpha,
                "meta_arch": meta_arch,
                "selection_mode": selection_mode,
                "reward_mode": reward_mode
            },
            "meta_network_params": convert_to_python_types(final_pool_info.get('meta_network_params', {})),
            "drift_events": model.drift_count if hasattr(model, 'drift_count') else 0,
            "window_metric": {
                "window_size": progress_interval,
                "auc_scores": scores,
                "precision_at_k_scores": precision_k_scores,
                "runtimes": times,
                "memory_usages": memory_usages,
                "pool_compositions": pool_compositions,
                "best_models": best_models
            },
            "final_pool": {
                "best_algorithm": final_best_algo,
                "best_hyperparameters": convert_to_python_types(final_best_hp),
                "best_mean_score": float(final_best_score),
                "pool_models": [
                    {
                        'algorithm': m['algorithm'],
                        'config_idx': int(m['config_idx']),
                        'hyperparameters': convert_to_python_types(m['hyperparameters'])
                    }
                    for m in final_pool_info['models']
                ],
                "meta_learner_suggestion": convert_to_python_types(final_pool_info['meta_learner_suggestion'])
            }
        }

        # Generate file name based on run count
        if run_count is not None:
            results["run_count"] = run_count
            results["seed"] = seed
            file_name = f"{model_label}_{dataset_name}_{run_count}.json"
        else:
            file_name = f"{model_label}_{dataset_name}.json"

        # Save results to file
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\nNOMAD completed on {dataset_name}")
        print(f"Final AUROC: {final_auroc:.4f}")
        print(f"Runtime: {total_runtime:.2f} seconds")
        print(f"Memory: {memory_usage:.2f} MB")
        print(f"Processed {processed_instances} instances")
        print(f"Final best algorithm: {final_best_algo}")
        if hasattr(model, 'drift_count'):
            print(f"Abrupt drift events detected: {model.drift_count}")
            if model.last_drift_step:
                print(f"  Last drift at step: {model.last_drift_step}")
        print(f"\nResults saved to {output_file}")

        return results

    except Exception as e:
        print(f"Error running NOMAD on {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NOMAD on a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='name of the dataset to run NOMAD on')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Interval for progress updates')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples for quick testing (None for full dataset)')
    parser.add_argument('--output_dir', type=str, default="benchmark_results", help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output', default=True)

    # NOMAD specific parameters
    parser.add_argument('--n_pool_models', type=int, default=10, help='Number of models in the pool (default: 10)')
    parser.add_argument('--n_candidate_configs', type=int, default=100, help='Number of random configs to generate (default: 100)')
    parser.add_argument('--evolution_interval', type=int, default=1000, help='Steps between evolution events (default: 1000)')

    # Algorithm-Aware Meta-Network parameters
    parser.add_argument('--shared_dim', type=int, default=32, help='Shared encoder output dimension (default: 32)')
    parser.add_argument('--head_hidden_dim', type=int, default=16, help='Hidden dimension in algorithm-specific heads (default: 16)')
    parser.add_argument('--K', type=int, default=20, help='Top-K configs to sample during selection (default: 20)')
    parser.add_argument('--diversity_weight', type=float, default=0.3, help='Weight for diversity penalty (default: 0.3)')

    # Stream feature extraction parameters
    parser.add_argument('--feature_window_size', type=int, default=100, help='Window size for feature extraction (default: 100)')
    parser.add_argument('--score_max_history', type=int, default=1000, help='Max history for score normalization and reward calculation (default: 1000)')

    # Reward calculation parameters
    parser.add_argument('--reward_min_updates', type=int, default=10, help='Min updates before correlation (default: 10)')

    # Meta-learner training parameters
    parser.add_argument('--meta_lr', type=float, default=0.01, help='Learning rate for meta-learner (default: 0.01)')
    parser.add_argument('--meta_momentum', type=float, default=0.9, help='Momentum for meta-learner (default: 0.9)')

    # Drift detection parameters (Page-Hinkley; always enabled as a core component)
    parser.add_argument('--drift_delta', type=float, default=0.005, help='Minimum magnitude of change to detect (default: 0.005)')
    parser.add_argument('--drift_threshold', type=float, default=50.0, help='Detection threshold - higher = only very abrupt changes (default: 50.0)')
    parser.add_argument('--drift_alpha', type=float, default=0.9999, help='Forgetting factor for gradual adaptation (default: 0.9999)')

    args = parser.parse_args()

    run_nomad(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        verbose=args.verbose,
        n_pool_models=args.n_pool_models,
        n_candidate_configs=args.n_candidate_configs,
        evolution_interval=args.evolution_interval,
        shared_dim=args.shared_dim,
        head_hidden_dim=args.head_hidden_dim,
        K=args.K,
        diversity_weight=args.diversity_weight,
        feature_window_size=args.feature_window_size,
        score_max_history=args.score_max_history,
        reward_min_updates=args.reward_min_updates,
        meta_lr=args.meta_lr,
        meta_momentum=args.meta_momentum,
        # Drift detection parameters
        drift_delta=args.drift_delta,
        drift_threshold=args.drift_threshold,
        drift_alpha=args.drift_alpha
    )
