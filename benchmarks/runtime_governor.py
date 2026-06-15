# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — runtime threshold governor benchmark

"""Measure the change-management overlay on live threshold updates.

The governor's logic is deterministic control flow, so the measurements are
behaviour and throughput:

* **Gating correctness** — labelled scenarios (global-source blocked, segment
  source with/without approval, auto-apply) produce the expected applied/held
  outcome; reported as the fraction correct.
* **Bounded convergence** — with a far recommendation and ``max_step=0.05`` the
  live threshold takes the expected number of steps to reach the target, never
  jumping; reported as the step count.
* **Throughput** — proposals per second.

Output: ``benchmarks/results/runtime_governor.json``. Reproduce with
``python -m benchmarks.runtime_governor``.
"""

from __future__ import annotations

import json
import time

from benchmarks._common import RESULTS_DIR
from director_ai.core.calibration.adaptive_threshold import (
    AdaptiveThresholdRecommendation,
)
from director_ai.core.calibration.runtime_governor import RuntimeThresholdGovernor
from director_ai.core.calibration.segmented_threshold import SegmentRecommendation


def _seg_rec(source, recommended, requires_human_approval):
    rec = AdaptiveThresholdRecommendation(
        current_threshold=0.5,
        recommended_threshold=recommended,
        expected_success_probability=0.9,
        current_success_probability=0.8,
        expected_lift=0.1,
        reason="benchmark",
        requires_human_approval=requires_human_approval,
        rollback_threshold=0.5,
    )
    return SegmentRecommendation(
        segment="seg", source=source, feedback_count=40, recommendation=rec
    )


class _StubLearner:
    def __init__(self, rec):
        self._rec = rec

    def observe(self, score, human_approved, *, segment):
        pass

    def recommend(self, *, segment):
        return self._rec


def _governor(rec, **kwargs):
    return RuntimeThresholdGovernor(
        learner=_StubLearner(rec), current_threshold=0.5, clock=lambda: 0.0, **kwargs
    )


def gating_correctness() -> dict:
    cases = [
        (_seg_rec("global", 0.7, False), False, {}),
        (_seg_rec("segment", 0.7, True), False, {}),  # held for approval
        (_seg_rec("segment", 0.7, False), True, {}),
        (_seg_rec("segment", 0.7, True), True, {"auto_apply": True}),
    ]
    passed = sum(
        1
        for rec, expect_applied, kw in cases
        if _governor(rec, **kw).propose("seg").applied is expect_applied
    )
    return {"n": len(cases), "accuracy": round(passed / len(cases), 4)}


def bounded_convergence() -> dict:
    gov = _governor(_seg_rec("segment", 0.9, False), max_step=0.05)
    steps = 0
    while gov.live_threshold("seg") < 0.9 - 1e-9 and steps < 100:
        gov.propose("seg")
        steps += 1
    # 0.5 -> 0.9 in 0.05 steps = 8
    return {"target": 0.9, "max_step": 0.05, "steps": steps}


def throughput(repeats: int) -> dict:
    gov = _governor(_seg_rec("segment", 0.52, False))
    t0 = time.perf_counter()
    for _ in range(repeats):
        gov.propose("seg")
    elapsed = time.perf_counter() - t0
    return {"proposals_per_sec": round(repeats / elapsed, 1) if elapsed else 0.0}


def run(*, repeats: int = 50000) -> dict:
    return {
        "benchmark": "runtime_governor",
        "gating": gating_correctness(),
        "convergence": bounded_convergence(),
        "throughput": throughput(repeats),
        "backend": "python-deterministic (change-management control flow)",
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "runtime_governor.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    g, c = result["gating"], result["convergence"]
    print("\nRuntime threshold governor:")
    print(f"  gating correctness={g['accuracy']:.2f} ({g['n']} scenarios)")
    print(
        f"  bounded convergence: {c['steps']} steps to {c['target']} @ {c['max_step']}"
    )
    print(f"  throughput {result['throughput']['proposals_per_sec']:.0f}/s")


if __name__ == "__main__":
    main()
