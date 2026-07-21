# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — end-to-end streaming false-halt benchmark (contradiction gate)

"""Token-level streaming false-halt benchmark driven by the contradiction gate.

The original ``streaming_false_halt_bench`` drives ``StreamingKernel`` from the
*coherence* score (``CoherenceScorer.review(...).score``), which folds
correct-but-unsupported text into "divergence" and false-halts >=99% of correct
streaming passages. This benchmark instead drives the halt from the working
mechanism: at each completed claim it scores ``P(contradiction)`` of the claim
against the retrieved grounding facts (:class:`ContradictionHalt`) and halts only
when a claim *contradicts* a governed fact — the same gate ``agent.stream`` uses.

It streams each passage token by token (so the halt index is a real token
position, not a post-hoc label), accumulates text exactly as
``StreamingKernel`` does (``"".join`` with leading-space tokens), detects claim
boundaries with :func:`ends_claim`, and runs the contradiction check on each
freshly completed claim. The metrics mirror the original bench so the two are
directly comparable:

* false-halt rate  — fraction of *correct* passages halted;
* recall           — fraction of *hallucinated* passages caught;
* token-of-halt accuracy / latency — how close the halt token is to the labelled
  contradiction token.

Run (loads the contradiction NLI model, so GPU is recommended)::

    python -m benchmarks.streaming_contradiction_halt_bench \
        --model training/output/contradiction-lora-merged --device 0 --tag finetuned
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from benchmarks.streaming_false_halt_bench import (
    BAD_PASSAGES,
    GOOD_PASSAGES,
    _expected_halt_index,
    _tokenize_simple,
)
from director_ai.core.runtime.contradiction_halt import ContradictionHaltDecision
from director_ai.core.runtime.streaming_gate import ends_claim

RESULTS_DIR = Path(__file__).parent / "results"

# A claim check maps a completed claim to a halt decision against its grounding.
ClaimCheck = Callable[[str], ContradictionHaltDecision]


@dataclass(frozen=True)
class StreamHaltOutcome:
    """Result of streaming one passage through the contradiction gate."""

    halted: bool
    halt_index: int
    contradiction: float
    fact: str
    token_count: int
    claims_checked: int


def stream_until_contradiction(
    tokens: Iterable[str],
    claim_check: ClaimCheck,
    *,
    min_words: int = 3,
) -> StreamHaltOutcome:
    """Stream *tokens*, halting on the first claim that contradicts its grounding.

    Tokens accumulate exactly as ``StreamingKernel.stream_tokens`` builds them
    (``"".join``). When the accumulated text completes a claim (sentence
    punctuation) and the claim carries at least *min_words* words, the claim
    since the previous boundary is passed to *claim_check*; a halt decision stops
    the stream and records the current token index. Sub-``min_words`` fragments
    never trigger a check, mirroring the production claim-boundary gate.
    """
    accumulated = ""
    claim_start = 0  # char offset in ``accumulated`` where the live claim begins
    index = -1
    claims_checked = 0
    for index, token in enumerate(tokens):
        accumulated += token
        if not ends_claim(accumulated):
            continue
        claim = accumulated[claim_start:].strip()
        if len(claim.split()) < min_words:
            claim_start = len(accumulated)
            continue
        claims_checked += 1
        decision = claim_check(claim)
        if decision.halt:
            return StreamHaltOutcome(
                halted=True,
                halt_index=index,
                contradiction=decision.contradiction,
                fact=decision.fact,
                token_count=index + 1,
                claims_checked=claims_checked,
            )
        claim_start = len(accumulated)
    return StreamHaltOutcome(
        halted=False,
        halt_index=-1,
        contradiction=0.0,
        fact="",
        token_count=index + 1,
        claims_checked=claims_checked,
    )


def _aggregate(
    good: list[StreamHaltOutcome],
    bad: list[tuple[StreamHaltOutcome, int]],
    *,
    token_tolerance: int = 8,
) -> dict:
    """Confusion + token-timing metrics, matching the original bench's schema."""
    false_positives = sum(1 for o in good if o.halted)
    true_positives = sum(1 for o, _exp in bad if o.halted)
    false_halt_rate = false_positives / len(good) if good else 0.0
    recall = true_positives / len(bad) if bad else 0.0
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    latencies: list[int] = []
    on_time = 0
    for outcome, expected in bad:
        if not outcome.halted:
            continue
        latency = outcome.halt_index - expected
        latencies.append(latency)
        if abs(latency) <= token_tolerance:
            on_time += 1
    token_accuracy = on_time / true_positives if true_positives else 0.0
    median_latency = statistics.median(latencies) if latencies else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": len(good) - false_positives,
        "false_negatives": len(bad) - true_positives,
        "halt_precision": round(precision, 4),
        "halt_recall": round(recall, 4),
        "false_halt_rate": round(false_halt_rate, 4),
        "token_of_halt_accuracy": round(token_accuracy, 4),
        "median_halt_latency_tokens": round(float(median_latency), 4),
        "token_tolerance": token_tolerance,
    }


def _build_halt(model_id: str, threshold: float, device: int):
    """Wire a :class:`ContradictionHalt` factory bound to a per-passage store."""
    from director_ai.core import GroundTruthStore
    from director_ai.core.runtime.contradiction_halt import ContradictionHalt
    from director_ai.core.scoring.contradiction import ContradictionScorer

    scorer = ContradictionScorer.from_pretrained(model_id, device=device)

    def make_check(facts: dict[str, str]) -> ClaimCheck:
        store = GroundTruthStore()
        for key, value in facts.items():
            store.add(key, value)
        halt = ContradictionHalt(scorer, store.retrieve_context, threshold=threshold)
        return halt.should_halt

    return make_check, scorer.threshold


def _isolation_verdict(load_avg: tuple[float, ...]) -> str:
    """Classify host isolation from the 1-minute load and core count, so the
    artefact honestly labels an isolated (dedicated GPU) refresh versus a
    contended shared-workstation run rather than hard-coding one label."""
    if not load_avg:
        return "unknown"
    cores = os.cpu_count() or 1
    ratio = load_avg[0] / cores
    if ratio < 0.5:
        return "isolated_quiet"
    if ratio < 1.0:
        return "moderate_load"
    return "contended_shared_host"


def _runtime_metadata() -> dict[str, object]:
    """Return host context (isolation classified from live load)."""

    try:
        load_avg = tuple(round(value, 4) for value in os.getloadavg())
    except OSError:
        load_avg = ()
    return {
        "command": [
            sys.executable,
            "-m",
            "benchmarks.streaming_contradiction_halt_bench",
            *sys.argv[1:],
        ],
        "host": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "load_average": load_avg,
        "isolation": _isolation_verdict(load_avg),
    }


def _is_numeric_fragment(fragment: str) -> bool:
    """A corrupted fragment is *numeric* when it is a bare number — the class of
    contradiction where NLI-based detection is known to be weakest, tracked
    separately so the semantic recall (the headline capability) is not masked."""
    return fragment.replace(",", "").replace(".", "").isdigit()


def _recall_by_kind(per_bad: list[dict]) -> dict:
    """Split halt recall by contradiction kind (semantic vs numeric)."""
    kinds: dict[str, list[bool]] = {"semantic": [], "numeric": []}
    for row in per_bad:
        kinds[row["kind"]].append(bool(row["halted"]))
    out: dict[str, dict[str, float | int]] = {}
    for kind, halted in kinds.items():
        n = len(halted)
        out[kind] = {
            "n": n,
            "caught": sum(halted),
            "recall": round(sum(halted) / n, 4) if n else 0.0,
        }
    return out


def run_benchmark(
    model_id: str,
    *,
    threshold: float = 0.5,
    device: int = -1,
) -> dict:
    """Stream every labelled passage through the contradiction gate and score it."""
    make_check, resolved_threshold = _build_halt(model_id, threshold, device)

    good_outcomes: list[StreamHaltOutcome] = []
    per_good: list[dict] = []
    t0 = time.perf_counter()
    for pid, facts, passage in GOOD_PASSAGES:
        outcome = stream_until_contradiction(
            _tokenize_simple(passage), make_check(facts)
        )
        good_outcomes.append(outcome)
        per_good.append(
            {
                "id": pid,
                "halted": outcome.halted,
                "contradiction": outcome.contradiction,
            }
        )

    bad_outcomes: list[tuple[StreamHaltOutcome, int]] = []
    per_bad: list[dict] = []
    for pid, facts, passage, expected_fragment in BAD_PASSAGES:
        expected_index = _expected_halt_index(passage, expected_fragment)
        outcome = stream_until_contradiction(
            _tokenize_simple(passage), make_check(facts)
        )
        bad_outcomes.append((outcome, expected_index))
        per_bad.append(
            {
                "id": pid,
                "kind": "numeric"
                if _is_numeric_fragment(expected_fragment)
                else "semantic",
                "halted": outcome.halted,
                "halt_index": outcome.halt_index,
                "expected_halt_index": expected_index,
                "contradiction": outcome.contradiction,
            }
        )
    elapsed = (time.perf_counter() - t0) * 1000

    from benchmarks.contradiction_aggrefact import _device_label

    return {
        "benchmark": "streaming_contradiction_halt",
        "benchmark_context": _runtime_metadata(),
        "gate": "contradiction",
        "model": model_id,
        "device": _device_label(device),
        "threshold": round(float(resolved_threshold), 4),
        "n_good": len(GOOD_PASSAGES),
        "n_bad": len(BAD_PASSAGES),
        "halt_quality": _aggregate(good_outcomes, bad_outcomes),
        "halt_recall_by_kind": _recall_by_kind(per_bad),
        "wall_ms": round(elapsed, 2),
        "per_good": per_good,
        "per_bad": per_bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="training/output/contradiction-lora-merged",
        help="HuggingFace id or local merged-model directory.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--tag", default="finetuned")
    args = parser.parse_args()

    result = run_benchmark(args.model, threshold=args.threshold, device=args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"streaming_contradiction_halt_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    q = result["halt_quality"]
    print(f"\nStreaming contradiction-halt ({args.tag}, {result['device']}):")
    print(f"  model: {result['model']}  threshold={result['threshold']}")
    print(
        f"  false-halt rate (correct passages): {q['false_halt_rate']:.4f} "
        f"({q['false_positives']}/{result['n_good']})"
    )
    print(
        f"  recall (hallucinated passages):     {q['halt_recall']:.4f} "
        f"({q['true_positives']}/{result['n_bad']})"
    )
    bk = result["halt_recall_by_kind"]
    print(
        f"    semantic recall: {bk['semantic']['recall']:.4f} "
        f"({bk['semantic']['caught']}/{bk['semantic']['n']})   "
        f"numeric recall: {bk['numeric']['recall']:.4f} "
        f"({bk['numeric']['caught']}/{bk['numeric']['n']})"
    )
    print(f"  halt precision:                     {q['halt_precision']:.4f}")
    print(f"  token-of-halt accuracy (±8):        {q['token_of_halt_accuracy']:.4f}")
    print(f"  median halt latency (tokens):       {q['median_halt_latency_tokens']}")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
