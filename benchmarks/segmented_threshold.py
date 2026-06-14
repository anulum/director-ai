# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — per-segment adaptive threshold benchmark

"""Measure the value of per-segment adaptive thresholds over a single global one.

A synthetic multi-segment workload is built where each segment has a *different*
optimal halt threshold (a strict clinical-style segment, a lenient chat-style
segment, and a middle one). Human-labelled feedback is replayed, then we compare:

* **segmented** — each segment's own recommended threshold applied to its own
  held-out stream;
* **global** — one pooled recommended threshold applied to every segment.

We report the mean approval-accuracy of each policy and the lift from
segmentation. The Beta-posterior arithmetic underneath uses the Rust
``rust_beta_posterior_mean`` kernel (inherited from ``AdaptiveThresholdLearner``);
this layer adds only routing, so there is no separate backend comparison.

Output: ``benchmarks/results/segmented_threshold.json``. Reproduce with
``python -m benchmarks.segmented_threshold``.
"""

from __future__ import annotations

import json
import random

from benchmarks._common import RESULTS_DIR
from director_ai.core.calibration.adaptive_threshold import AdaptiveThresholdLearner
from director_ai.core.calibration.segmented_threshold import SegmentedThresholdLearner

_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
# segment -> approval cutoff (the latent "true" boundary a good threshold matches)
_SEGMENTS = {"clinical": 0.8, "middle": 0.6, "chat": 0.35}
_TRAIN_PER_SEGMENT = 120
_TEST_PER_SEGMENT = 400


def _label(score: float, cutoff: float) -> bool:
    return score >= cutoff


def _accuracy(threshold: float, cutoff: float, rng: random.Random, n: int) -> float:
    # Halt when score < threshold; approve when score >= threshold. Correct when
    # that decision matches the latent human label (approve iff score >= cutoff).
    hits = 0
    for _ in range(n):
        score = rng.random()
        approved = score >= threshold
        if approved == _label(score, cutoff):
            hits += 1
    return hits / n


def run() -> dict:
    segmented = SegmentedThresholdLearner(
        candidate_thresholds=_THRESHOLDS,
        current_threshold=0.5,
        min_samples=20,
        random_seed=7,
    )
    global_only = AdaptiveThresholdLearner(
        candidate_thresholds=_THRESHOLDS,
        current_threshold=0.5,
        min_samples=20,
        random_seed=7,
    )

    train_rng = random.Random(1)
    for segment, cutoff in _SEGMENTS.items():
        for _ in range(_TRAIN_PER_SEGMENT):
            score = train_rng.random()
            approved = _label(score, cutoff)
            segmented.observe(score, approved, segment=segment)
            global_only.observe(score, approved)

    global_thr = global_only.recommend().recommended_threshold or 0.5

    seg_accs = []
    glob_accs = []
    per_segment = {}
    test_rng = random.Random(2)
    for segment, cutoff in _SEGMENTS.items():
        rec = segmented.recommend(segment=segment).recommendation
        seg_thr = rec.recommended_threshold or 0.5
        seg_acc = _accuracy(seg_thr, cutoff, test_rng, _TEST_PER_SEGMENT)
        glob_acc = _accuracy(global_thr, cutoff, test_rng, _TEST_PER_SEGMENT)
        seg_accs.append(seg_acc)
        glob_accs.append(glob_acc)
        per_segment[segment] = {
            "cutoff": cutoff,
            "segment_threshold": seg_thr,
            "segment_accuracy": round(seg_acc, 4),
            "global_accuracy": round(glob_acc, 4),
        }

    mean_seg = sum(seg_accs) / len(seg_accs)
    mean_glob = sum(glob_accs) / len(glob_accs)
    return {
        "benchmark": "segmented_threshold",
        "n_segments": len(_SEGMENTS),
        "global_threshold": global_thr,
        "mean_segmented_accuracy": round(mean_seg, 4),
        "mean_global_accuracy": round(mean_glob, 4),
        "segmentation_lift": round(mean_seg - mean_glob, 4),
        "per_segment": per_segment,
        "backend": "inherits rust_beta_posterior_mean via AdaptiveThresholdLearner",
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "segmented_threshold.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"\nSegmented threshold (n_segments={result['n_segments']}):")
    print(
        f"  mean accuracy segmented={result['mean_segmented_accuracy']:.3f} "
        f"global={result['mean_global_accuracy']:.3f} "
        f"lift={result['segmentation_lift']:+.3f}"
    )
    for seg, d in result["per_segment"].items():
        print(
            f"    {seg:9} cutoff={d['cutoff']} thr={d['segment_threshold']} "
            f"seg_acc={d['segment_accuracy']:.3f} glob_acc={d['global_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
