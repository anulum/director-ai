# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Ensemble Benchmark Runner (UpCloud GPU)

"""Run the full ensemble AggreFact benchmark on a GPU instance.

Designed for UpCloud L40S or equivalent. Loads base + 4-5 domain models
and evaluates on the full LLM-AggreFact test set with threshold sweep.

Prerequisites on the remote machine:
    pip install datasets scikit-learn nltk transformers torch
    export HF_TOKEN=hf_...

Usage:
    python run_ensemble_benchmark.py
    python run_ensemble_benchmark.py --sweep
    python run_ensemble_benchmark.py --max-samples 200  # quick test
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks._common import save_results
from benchmarks.aggrefact_ensemble import (
    _print_ensemble_results,
    run_ensemble_benchmark,
    sweep_ensemble_thresholds,
)


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("max_samples", nargs="?", type=int, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()

    if args.sweep:
        best_thresh, best_strat, r = sweep_ensemble_thresholds(
            max_samples=args.max_samples,
            models_dir=args.models_dir,
        )
        print(f"\nOptimal: threshold={best_thresh:.2f}, strategy={best_strat}")
    else:
        r = run_ensemble_benchmark(
            threshold=args.threshold,
            max_samples=args.max_samples,
            models_dir=args.models_dir,
        )

    _print_ensemble_results(r)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} min")

    # Save full results
    best_m = r.ensemble_mean if r.best_strategy == "mean" else r.ensemble_max
    save_data = {
        "benchmark": "LLM-AggreFact",
        "mode": "ensemble",
        "models": list(r.individual.keys()),
        "best_strategy": r.best_strategy,
        "best_accuracy": round(r.best_accuracy, 4),
        "base_accuracy": round(r.base_metrics.avg_balanced_acc, 4),
        "individual": {
            name: round(m.avg_balanced_acc, 4) for name, m in r.individual.items()
        },
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    if best_m:
        save_data["ensemble"] = best_m.to_dict()
    save_results(save_data, "aggrefact_ensemble.json")

    # Also save to working dir for easy SCP retrieval
    with open("ensemble_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("Results also saved to ./ensemble_results.json")


if __name__ == "__main__":
    main()
