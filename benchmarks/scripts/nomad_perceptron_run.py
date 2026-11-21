"""
NOMAD-Perceptron ablation runner.

Same streaming pipeline as NOMAD, but the Algorithm-Aware Meta-Network is
replaced by a linear perceptron meta-learner (single-layer linear heads and a
linear shared encoder, no hidden layer / ReLU). This isolates the benefit of
the nonlinear AAMN heads. Results are produced by the real streaming run
(real AUC, real precision@k), not by any generator.
"""

import argparse
from nomad_run import run_nomad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NOMAD-Perceptron on a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='name of the dataset to run on')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Interval for progress updates')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples for quick testing')
    parser.add_argument('--output_dir', type=str, default="benchmark_results", help='Output directory')
    args = parser.parse_args()

    run_nomad(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        verbose=False,
        model_label="NOMAD-Perceptron",
        meta_arch="perceptron",
        selection_mode="meta",
    )
