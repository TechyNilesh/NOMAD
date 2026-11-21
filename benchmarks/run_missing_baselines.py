"""
Re-run the real baseline models that are missing precision@k (AutoSAD, SIF)
across all 16 datasets x 10 runs with random seeds. Each job is a real streaming
run that writes a JSON with real AUC and real precision@k into the shared
benchmark_results directory.

Usage:
    python run_missing_baselines.py --output_dir /abs/path/to/benchmark_results [--workers 30]
"""

import concurrent.futures
import subprocess
import random
import argparse
import os
import sys

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

dataset_name_list = [
    '8_celeba', '33_skin', '34_smtp', '11_donors', '5_campaign', '10_cover',
    '3_backdoor', '16_http', '32_shuttle', '13_fraud', '9_census', '1_ALOI',
    '48_chess', '49_kddcup99_prob', '50_bank', '51_kddcup99_u2r',
]

run_counts = range(1, 11)

# (script filename, human label) for the real baselines missing precision@k
MODELS = [
    ("autosad_run.py", "AutoSAD"),
    ("sif_run.py", "SIF"),
]


def run_one(script, label, dataset, run_count, output_dir, progress_interval):
    seed = random.randint(1, 100)
    cmd = [
        sys.executable, os.path.join(SCRIPTS_DIR, script),
        "--dataset", dataset,
        "--run_count", str(run_count),
        "--seed", str(seed),
        "--progress_interval", str(progress_interval),
        "--output_dir", output_dir,
    ]
    print(f"START {label} {dataset} run {run_count} (seed {seed})", flush=True)
    proc = subprocess.run(cmd, cwd=SCRIPTS_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"ERROR {label} {dataset} run {run_count}:\n{proc.stderr[-2000:]}", flush=True)
    else:
        print(f"DONE  {label} {dataset} run {run_count}", flush=True)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Re-run real baselines missing precision@k.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Absolute directory to save results")
    parser.add_argument("--workers", type=int, default=30, help="Parallel workers")
    parser.add_argument("--progress_interval", type=int, default=1000)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated labels to run (e.g. 'SIF'); default all")
    args = parser.parse_args()

    models = MODELS
    if args.models:
        want = {m.strip().lower() for m in args.models.split(",")}
        models = [(s, l) for (s, l) in MODELS if l.lower() in want]

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tasks = [(script, label, ds, rc)
             for script, label in models
             for ds in dataset_name_list
             for rc in run_counts]
    print(f"Launching {len(tasks)} jobs with {args.workers} workers -> {output_dir}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_one, s, l, ds, rc, output_dir, args.progress_interval)
                   for (s, l, ds, rc) in tasks]
        concurrent.futures.wait(futures)

    print("ALL BASELINE RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
