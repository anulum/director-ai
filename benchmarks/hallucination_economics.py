# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — hallucination economics benchmark

"""Measure the cost-risk guard-action selector.

The decision is exact arithmetic over a small action menu (no scoring model, no
polyglot kernel), so the measurements are decision correctness and throughput:

* **Decision accuracy** — labelled ``(risk, hallucination_cost, expected_action)``
  scenarios spanning low/high risk and low/high stakes are run through
  ``decide``; we report how often the chosen action matches the economically
  correct label.
* **Value realised** — the mean expected loss avoided versus doing nothing across
  the scenarios where guarding is worthwhile, in the abstract cost unit.
* **Throughput** — decisions per second.

Output: ``benchmarks/results/hallucination_economics.json``. Reproduce with
``python -m benchmarks.hallucination_economics``.
"""

from __future__ import annotations

import json
import time

from benchmarks._common import RESULTS_DIR
from director_ai.core.routing import HallucinationEconomics

# (risk, hallucination_cost, expected_action)
_SCENARIOS: list[tuple[float, float, str]] = [
    (0.01, 1.0, "skip"),  # negligible risk, low stakes
    (0.05, 1.0, "heuristic"),  # low risk, low stakes -> cheap guard
    (0.5, 10.0, "nli"),  # moderate risk, moderate stakes
    (0.9, 1.0, "nli"),  # high risk, low stakes -> still NLI
    (0.9, 100.0, "escalate"),  # high risk, high stakes -> escalate
    (0.95, 1000.0, "human_review"),  # critical stakes -> human review wins
]


def decision_accuracy() -> dict:
    econ = HallucinationEconomics()
    hits = sum(
        1
        for risk, hcost, expected in _SCENARIOS
        if econ.decide(risk, hallucination_cost=hcost).action == expected
    )
    return {"n": len(_SCENARIOS), "accuracy": round(hits / len(_SCENARIOS), 4)}


def value_realised() -> dict:
    econ = HallucinationEconomics()
    values = [
        d.value
        for risk, hcost, _ in _SCENARIOS
        if (d := econ.decide(risk, hallucination_cost=hcost)).worth_guarding
    ]
    mean = round(sum(values) / len(values), 4) if values else 0.0
    return {"guarded_scenarios": len(values), "mean_loss_avoided": mean}


def throughput(repeats: int) -> dict:
    econ = HallucinationEconomics()
    t0 = time.perf_counter()
    for _ in range(repeats):
        for risk, hcost, _ in _SCENARIOS:
            econ.decide(risk, hallucination_cost=hcost)
    elapsed = time.perf_counter() - t0
    rate = (len(_SCENARIOS) * repeats) / elapsed if elapsed else 0.0
    return {"decisions_per_sec": round(rate, 1)}


def run(*, repeats: int = 20000) -> dict:
    return {
        "benchmark": "hallucination_economics",
        "decision": decision_accuracy(),
        "value": value_realised(),
        "throughput": throughput(repeats),
        "backend": "python-deterministic (decision arithmetic; no kernel)",
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "hallucination_economics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    d = result["decision"]
    print(f"\nHallucination economics (n={d['n']}):")
    print(f"  decision accuracy={d['accuracy']:.2f}")
    print(f"  mean loss avoided={result['value']['mean_loss_avoided']}")
    print(f"  throughput {result['throughput']['decisions_per_sec']:.0f}/s")


if __name__ == "__main__":
    main()
