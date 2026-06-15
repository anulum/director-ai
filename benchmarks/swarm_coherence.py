# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — swarm coherence benchmark

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

from benchmarks._common import RESULTS_DIR
from director_ai.core.swarm_coherence import SwarmCoherenceMonitor
from director_ai.core.swarm_coherence import cascade_monitor as _cm


class _StubNLI:
    threshold = 0.5

    def __init__(self, table):
        self._table = table

    def contradiction(self, premise: str, hypothesis: str) -> float:
        return self._table.get((premise, hypothesis), 0.0)


_PLAN = "The migration starts on Monday morning."
_OK = "The team will prepare the runbook beforehand."
_CONFLICT = "The migration starts on Friday evening."

# (cascade messages, nli_table, expected_halt)
_CASCADES = [
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


def cascade_correctness() -> dict:
    hits = 0
    for messages, table, expected in _CASCADES:
        mon = SwarmCoherenceMonitor(nli=_StubNLI(table))
        halted = False
        for agent_id, text in messages:
            halted = mon.observe(agent_id, text).halted
        hits += int(halted == expected)
    return {"n": len(_CASCADES), "accuracy": round(hits / len(_CASCADES), 4)}


def parity() -> dict:
    saved_flag, saved_fn = _cm._RUST_SWARM, _cm.rust_word_overlap
    pairs = [(_PLAN, _CONFLICT), (_PLAN, _OK), (_OK, _CONFLICT)]
    try:
        rust_vals = [_cm._lexical_overlap(a, b) for a, b in pairs]
        _cm._RUST_SWARM = False
        _cm.rust_word_overlap = None
        py_vals = [_cm._lexical_overlap(a, b) for a, b in pairs]
    finally:
        _cm._RUST_SWARM, _cm.rust_word_overlap = saved_flag, saved_fn
    exact = all(r == p for r, p in zip(rust_vals, py_vals, strict=True))
    return {"pairs": len(pairs), "bit_exact": exact, "kernel_active": saved_flag}


def throughput(repeats: int) -> dict:
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


def run(*, repeats: int = 5000) -> dict:
    return {
        "benchmark": "swarm_coherence",
        "cascade": cascade_correctness(),
        "parity": parity(),
        "throughput": throughput(repeats),
        "backend": "rust_word_overlap kernel with bit-exact Python fallback",
    }


def main() -> None:
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
