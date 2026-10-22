# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — swarm coherence benchmark

"""Measure the cross-agent cascade-coherence monitor.

The contradiction judgement itself comes from an injected NLI scorer; here we
measure the cascade machinery (claim accumulation, cross-agent comparison, halt)
with a deterministic NLI stub, plus the offline lexical backend:

* **Cascade-halt correctness** — labelled cascades (coherent vs poisoned with a
  planted cross-agent contradiction) are replayed through a fresh monitor; we
  report how often ``halted`` matches the label.
* **Rust/Python parity** — the lexical topical-overlap backend is bit-for-bit
  identical with the Rust kernel and the forced Python fallback.
* **Throughput** — ``observe`` calls per second.

Output: ``benchmarks/results/swarm_coherence.json``. Reproduce with
``python -m benchmarks.swarm_coherence``.
"""

from __future__ import annotations

import json
import time
from typing import TypedDict

from benchmarks._common import RESULTS_DIR
from director_ai.core import text_overlap as _overlap
from director_ai.core.swarm_coherence import SwarmCoherenceMonitor


class CascadeCorrectness(TypedDict):
    """Label agreement for the fixed deterministic cascade inputs."""

    n: int
    accuracy: float


class Parity(TypedDict):
    """Real kernel comparison; unavailable kernels leave parity unqualified."""

    pairs: int
    bit_exact: bool | None
    kernel_active: bool


class Throughput(TypedDict):
    """Measured public monitor observations per second."""

    observe_per_sec: float


class BenchmarkResult(TypedDict):
    """Complete serialisable swarm-control benchmark output."""

    benchmark: str
    cascade: CascadeCorrectness
    parity: Parity
    throughput: Throughput
    backend: str


class _StubNLI:
    """Return the explicit contradiction probability for each labelled pair."""

    threshold = 0.5

    def __init__(self, table: dict[tuple[str, str], float]) -> None:
        """Retain the deterministic pair scores used by the control benchmark."""
        self._table = table

    def contradiction(self, premise: str, hypothesis: str) -> float:
        """Look up the ordered pair, treating unlabelled pairs as coherent."""
        return self._table.get((premise, hypothesis), 0.0)


_PLAN = "The migration starts on Monday morning."
_OK = "The team will prepare the runbook beforehand."
_CONFLICT = "The migration starts on Friday evening."

# (cascade messages, nli_table, expected_halt)
_CASCADES: list[tuple[list[tuple[str, str]], dict[tuple[str, str], float], bool]] = [
    ([("planner", _PLAN), ("worker", _OK)], {}, False),
    (
        [("planner", _PLAN), ("editor", _CONFLICT)],
        {(_PLAN, _CONFLICT): 0.93},
        True,
    ),
    (
        [("a", _PLAN), ("b", _OK), ("c", _CONFLICT)],
        {(_PLAN, _CONFLICT): 0.9},
        True,
    ),
    ([("a", _PLAN), ("b", _OK), ("c", _OK)], {}, False),
]


def cascade_correctness() -> CascadeCorrectness:
    """Replay each labelled cascade through fresh public monitors."""
    hits = 0
    for messages, table, expected in _CASCADES:
        mon = SwarmCoherenceMonitor(nli=_StubNLI(table))
        halted = False
        for agent_id, text in messages:
            halted = mon.observe(agent_id, text).halted
        hits += int(halted == expected)
    return {"n": len(_CASCADES), "accuracy": round(hits / len(_CASCADES), 4)}


def parity() -> Parity:
    """Compare the current Rust dispatch and forced Python implementation.

    The shared public helper owns the backend now. Its threshold argument forces
    each implementation without changing global dispatch state. If the optional
    kernel is absent, report that no cross-language parity was established.
    """
    pairs = [(_PLAN, _CONFLICT), (_PLAN, _OK), (_OK, _CONFLICT)]
    kernel_active = _overlap._RUST_WORD_OVERLAP
    rust_vals = [_overlap.word_overlap(a, b, large_input_words=0) for a, b in pairs]
    py_vals = [_overlap.word_overlap(a, b, large_input_words=2**63) for a, b in pairs]
    exact = (
        all(r == p for r, p in zip(rust_vals, py_vals, strict=True))
        if kernel_active
        else None
    )
    return {"pairs": len(pairs), "bit_exact": exact, "kernel_active": kernel_active}


def throughput(repeats: int) -> Throughput:
    """Time public observe calls over the fixed coherent cascade."""
    table = {(_PLAN, _CONFLICT): 0.93}
    t0 = time.perf_counter()
    n = 0
    for _ in range(repeats):
        mon = SwarmCoherenceMonitor(nli=_StubNLI(table))
        for agent_id, text in (("planner", _PLAN), ("worker", _OK)):
            mon.observe(agent_id, text)
            n += 1
    elapsed = time.perf_counter() - t0
    return {"observe_per_sec": round(n / elapsed, 1) if elapsed else 0.0}


def run(*, repeats: int = 5000) -> BenchmarkResult:
    """Collect cascade correctness, actual backend parity and throughput."""
    return {
        "benchmark": "swarm_coherence",
        "cascade": cascade_correctness(),
        "parity": parity(),
        "throughput": throughput(repeats),
        "backend": "rust_word_overlap kernel with bit-exact Python fallback",
    }


def main() -> None:
    """Write and display the complete benchmark result."""
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "swarm_coherence.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    c = result["cascade"]
    print(f"\nSwarm coherence (n={c['n']}):")
    print(f"  cascade-halt correctness={c['accuracy']:.2f}")
    print(f"  parity bit_exact={result['parity']['bit_exact']}")
    print(f"  throughput {result['throughput']['observe_per_sec']:.0f}/s")


if __name__ == "__main__":
    main()
