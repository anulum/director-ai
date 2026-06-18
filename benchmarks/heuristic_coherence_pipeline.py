# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Heuristic Coherence Pipeline Benchmark
"""Measure the heuristic coherence orchestration path.

The benchmark covers the hot no-model orchestration around
``CoherenceScorer._heuristic_coherence`` after the route-selection split. It
does not claim production latency: it records local, non-isolated regression
evidence so future refactors can compare route overhead without downloading NLI
models.

Reproduce with ``python -m benchmarks.heuristic_coherence_pipeline``.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import cast

from benchmarks._common import RESULTS_DIR
from director_ai.core import CoherenceScorer

_PROMPT = "The capital of France is Paris and the Seine crosses the city."
_ANSWER = "Paris is the capital of France and the Seine crosses the city."
_DIALOGUE = "User: What is the capital of France?\nAssistant: Paris.\nUser: Why?"
_SUMMARY_SOURCE = "Summarise this source.\n\nSource document: " + (
    "Paris is the capital of France. The Seine crosses Paris. " * 12
)


def _cpu_model() -> str:
    """Return the Linux CPU model string when available."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def _route_outputs() -> dict[str, dict[str, float | str | None]]:
    """Run each no-model route once and return component scores."""
    default = CoherenceScorer(use_nli=False)
    factual_only = CoherenceScorer(use_nli=False, w_logic=0.0, w_fact=1.0)
    dialogue = CoherenceScorer(use_nli=False)
    summary = CoherenceScorer(use_nli=False, w_logic=0.0, w_fact=1.0)
    summary._use_prompt_as_premise = True

    cases = {
        "parallel_components": (default, _PROMPT, _ANSWER),
        "factual_only": (factual_only, _PROMPT, _ANSWER),
        "dialogue": (dialogue, _DIALOGUE, "Because it is France's capital."),
        "prompt_premise": (summary, _SUMMARY_SOURCE, "Paris is the capital."),
    }
    outputs: dict[str, dict[str, float | str | None]] = {}
    for route, (scorer, prompt, answer) in cases.items():
        h_logic, h_fact, coherence, evidence = scorer._heuristic_coherence(
            prompt,
            answer,
        )
        selected_route = "parallel_components" if route == "dialogue" else route
        if route == "prompt_premise":
            selected_route = "factual_only"
        outputs[route] = {
            "route": selected_route,
            "h_logical": round(h_logic, 6),
            "h_factual": round(h_fact, 6),
            "coherence": round(coherence, 6),
            "evidence": "present" if evidence is not None else None,
        }
        scorer.close()
    return outputs


def _throughput(repeats: int) -> dict[str, float | int]:
    """Measure repeated no-model route execution throughput."""
    scorers = [
        CoherenceScorer(use_nli=False),
        CoherenceScorer(use_nli=False, w_logic=0.0, w_fact=1.0),
        CoherenceScorer(use_nli=False),
        CoherenceScorer(use_nli=False, w_logic=0.0, w_fact=1.0),
    ]
    scorers[3]._use_prompt_as_premise = True
    cases = [
        (_PROMPT, _ANSWER),
        (_PROMPT, _ANSWER),
        (_DIALOGUE, "Because it is France's capital."),
        (_SUMMARY_SOURCE, "Paris is the capital."),
    ]
    start = time.perf_counter()
    reviews = 0
    try:
        for _ in range(repeats):
            for scorer, (prompt, answer) in zip(scorers, cases, strict=True):
                scorer._heuristic_coherence(prompt, answer)
                reviews += 1
    finally:
        for scorer in scorers:
            scorer.close()
    elapsed = time.perf_counter() - start
    return {
        "reviews": reviews,
        "seconds": round(elapsed, 6),
        "reviews_per_sec": round(reviews / elapsed, 1) if elapsed else 0.0,
    }


def run(*, repeats: int = 5000) -> dict[str, object]:
    """Run the local-regression benchmark and return JSON-serialisable results.

    Parameters
    ----------
    repeats:
        Number of route batches. Each batch executes four route cases.

    Returns
    -------
    dict[str, object]
        Benchmark result with route outputs, throughput, host metadata, command,
        and evidence boundary.
    """
    load_before = os.getloadavg()
    route_outputs = _route_outputs()
    throughput = _throughput(repeats)
    load_after = os.getloadavg()
    return {
        "benchmark": "heuristic_coherence_pipeline",
        "evidence_grade": "non_isolated_local_regression",
        "claim_boundary": (
            "Measures local no-model orchestration overhead only; not a "
            "production latency claim."
        ),
        "command": "python -m benchmarks.heuristic_coherence_pipeline",
        "isolation": {
            "method": "none",
            "cpu_affinity": "not pinned",
            "host_load_before": load_before,
            "host_load_after": load_after,
            "other_heavy_jobs": "not asserted",
        },
        "host": {
            "cpu_model": _cpu_model(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "routes": {"cases": len(route_outputs), "names": sorted(route_outputs)},
        "route_outputs": route_outputs,
        "throughput": throughput,
    }


def main() -> None:
    """Write the benchmark result to ``benchmarks/results``."""
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "heuristic_coherence_pipeline.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    throughput = cast(dict[str, object], result["throughput"])
    routes = cast(dict[str, object], result["routes"])
    print("Heuristic coherence pipeline:")
    print(f"  evidence={result['evidence_grade']}")
    print(f"  routes={routes['cases']}")
    print(f"  throughput={throughput['reviews_per_sec']}/s")
    print(f"  result={output}")


if __name__ == "__main__":
    main()
