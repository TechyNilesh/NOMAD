"""
RandomSearchEnsemble baseline runner.

Same random pool of detector configurations as NOMAD, but with no meta-learning:
the anomaly score is the average of the pool's normalized scores, and the pool
is never evolved/selected. This isolates the benefit of NOMAD's meta-learning
from simply running several random configurations. Results are produced by the
real streaming run (real AUC, real precision@k), not by any generator.
"""

import argparse
from nomad_run import run_nomad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RandomSearchEnsemble on a dataset')
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
        model_label="RandomSearchEnsemble",
        meta_arch="aamn",
        selection_mode="ensemble_avg",
    )
