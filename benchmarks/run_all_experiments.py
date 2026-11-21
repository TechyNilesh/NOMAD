#!/usr/bin/env python3
"""
Master runner for the full NOMAD evaluation suite.

This script runs every experiment reported in the paper using the runner
scripts in this folder.

Experiments:
  1. Main benchmark       -- 13 methods + ARCUS/METER/DyMETER, 16 datasets, 10 runs
                             (run_scripts.py, plus arcus/meter/dymeter runners)
  2. Drift-detector study -- Page-Hinkley vs ADWIN/KSWIN/DDM/None, 16 datasets
                             (drift_detector_ablation.py)
  3. Controlled drift     -- INSECTS streams with known change points
                             (run_insects_drift.py for prequential methods,
                              arcus/meter/dymeter runners in temporal mode)
  4. Ablation study       -- NOMAD component ablations (ablation_study_nomad.py)
  5. Detection delay      -- injected anomaly-event delay (detection_delay_analysis.py)
  6. Sensitivity          -- hyperparameter sensitivity (sensitivity_analysis_nomad.py)

Usage:
    python run_all_experiments.py --dry-run            # print commands only
    python run_all_experiments.py --suite main         # run a single suite
    python run_all_experiments.py --suite all          # run everything
"""

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
RESULTS = CODE_DIR.parent / "results"

DATASETS = [
    "8_celeba", "33_skin", "34_smtp", "11_donors", "5_campaign",
    "10_cover", "3_backdoor", "16_http", "32_shuttle", "13_fraud",
    "9_census", "1_ALOI", "48_chess", "49_kddcup99_prob", "50_bank",
    "51_kddcup99_u2r",
]

MAIN_MODELS = [
    "nomad", "nomad_perceptron", "random_search_ensemble",
    "autosad", "sead", "hst", "ifasd", "loda", "oif", "rrcf",
    "rshash", "sif", "xstream", "meter", "arcus", "dymeter",
]

INSECTS_STREAMS = ["INSECTS_Abr", "INSECTS_Incr", "INSECTS_IncrGrd", "INSECTS_IncrRecr"]

DRIFT_DETECTORS = ["page_hinkley", "adwin", "kswin", "ddm", "no_detection"]

ABLATION_COMPONENTS = [
    "full_model", "no_drift_detection", "no_diversity",
    "no_meta_learning", "no_evolution", "no_feature_windowing",
]


def build_commands(args):
    """Yield (description, argv) tuples for the requested suite(s)."""
    suites = args.suite if isinstance(args.suite, list) else [args.suite]

    for suite in suites:
        if suite == "main":
            for model in MAIN_MODELS:
                for ds in DATASETS:
                    cmd = [sys.executable, str(CODE_DIR / "scripts" / f"{model}_run.py"),
                           "--dataset", ds, "--output_dir", str(RESULTS / "benchmark_results")]
                    yield (f"main:{model}:{ds}", cmd)

        elif suite == "drift_detector":
            cmd = [sys.executable, str(CODE_DIR / "drift_detector_ablation.py"),
                   "--output_dir", str(RESULTS / "drift_detector_ablation_full")]
            yield ("drift_detector:all", cmd)

        elif suite == "controlled_drift":
            # Prequential methods (temporal order).
            cmd = [sys.executable, str(CODE_DIR / "run_insects_drift.py"),
                   "--output-dir", str(RESULTS / "concept_drift_insects")]
            yield ("controlled_drift:prequential", cmd)

            # Batch-incremental methods (ARCUS / METER / DyMETER) on the same
            # streams, preserving temporal order (their runners shuffle only for
            # the tabular benchmark; pass a temporal flag / dataset name).
            for model in ["arcus", "meter", "dymeter"]:
                for stream in INSECTS_STREAMS:
                    cmd = [sys.executable, str(CODE_DIR / "scripts" / f"{model}_run.py"),
                           "--dataset", stream, "--temporal",
                           "--output_dir", str(RESULTS / "concept_drift_insects")]
                    yield (f"controlled_drift:{model}:{stream}", cmd)

        elif suite == "ablation":
            cmd = [sys.executable, str(CODE_DIR / "ablation_study_nomad.py"),
                   "--component", "all",
                   "--output_dir", str(RESULTS / "ablation_study")]
            yield ("ablation:all", cmd)

        elif suite == "detection_delay":
            cmd = [sys.executable, str(CODE_DIR / "detection_delay_analysis.py"),
                   "--output-dir", str(RESULTS / "detection_delay_analysis")]
            yield ("detection_delay:all", cmd)

        elif suite == "sensitivity":
            cmd = [sys.executable, str(CODE_DIR / "sensitivity_analysis_nomad.py"),
                   "--output_dir", str(RESULTS / "sensitivity_analysis")]
            yield ("sensitivity:all", cmd)

        else:
            raise ValueError(f"Unknown suite: {suite}")


def main():
    parser = argparse.ArgumentParser(description="Master runner for the NOMAD evaluation suite")
    parser.add_argument("--suite", nargs="+", default=["main"],
                        choices=["main", "drift_detector", "controlled_drift",
                                 "ablation", "detection_delay", "sensitivity", "all"],
                        help="Experiment suite(s) to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without executing them")
    args = parser.parse_args()

    if args.suite == ["all"]:
        args.suite = ["main", "drift_detector", "controlled_drift",
                      "ablation", "detection_delay", "sensitivity"]

    n = 0
    for description, argv in build_commands(args):
        n += 1
        print(f"[{description}]")
        print("  " + " ".join(argv))
        if not args.dry_run:
            subprocess.run(argv, check=False)

    print(f"\nTotal commands: {n}")


if __name__ == "__main__":
    main()
