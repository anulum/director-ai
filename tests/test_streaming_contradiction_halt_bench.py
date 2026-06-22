# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — streaming contradiction-halt benchmark harness tests

"""Offline tests for the contradiction-gate streaming benchmark.

These exercise the pure streaming loop and the aggregation without loading any
NLI model: a fake claim check halts on a trigger word, so the claim-level halt
index, claim-boundary gating, and confusion/latency metrics are verified
deterministically.
"""

from __future__ import annotations

from benchmarks.streaming_contradiction_halt_bench import (
    StreamHaltOutcome,
    _aggregate,
    _runtime_metadata,
    stream_until_contradiction,
)
from benchmarks.streaming_false_halt_bench import _tokenize_simple
from director_ai.core.runtime.contradiction_halt import ContradictionHaltDecision


def _check_on(trigger: str):
    """A claim check that halts when *trigger* appears in the claim."""

    def check(claim: str) -> ContradictionHaltDecision:
        if trigger in claim.lower():
            return ContradictionHaltDecision(halt=True, contradiction=0.91, fact="f")
        return ContradictionHaltDecision(halt=False, contradiction=0.02)

    return check


# --------------------------------------------------------------------------- #
# stream_until_contradiction                                                  #
# --------------------------------------------------------------------------- #


def test_no_contradiction_streams_to_end_and_counts_claims() -> None:
    text = "Water boils at one hundred degrees. The sky is clear today. All is fine."
    tokens = _tokenize_simple(text)
    outcome = stream_until_contradiction(tokens, _check_on("zzz-never"))
    assert outcome.halted is False
    assert outcome.halt_index == -1
    assert outcome.token_count == len(tokens)
    assert outcome.claims_checked == 3  # three complete, >=3-word claims


def test_halts_on_first_contradicting_claim() -> None:
    text = "Water boils at one hundred degrees. Water boils at fifty wrong degrees. Tail follows here."
    tokens = _tokenize_simple(text)
    outcome = stream_until_contradiction(tokens, _check_on("wrong"))
    assert outcome.halted is True
    assert outcome.contradiction == 0.91
    assert outcome.fact == "f"
    # Halt token is the one that completes the second claim (its full stop).
    expected = next(
        i for i, _ in enumerate(tokens) if "".join(tokens[: i + 1]).count(".") == 2
    )
    assert outcome.halt_index == expected
    assert outcome.token_count == expected + 1
    assert outcome.claims_checked == 2  # first clean claim, then the halting one


def test_sub_min_words_fragment_is_not_checked() -> None:
    # "Yes." is a one-word claim and must never reach the check, even though it
    # contains the trigger; the real claim afterwards is what counts.
    text = "Yes. The reactor output is wrong here."
    tokens = _tokenize_simple(text)
    outcome = stream_until_contradiction(tokens, _check_on("yes"), min_words=3)
    assert outcome.halted is False  # "yes" only appears in the skipped fragment
    assert outcome.claims_checked == 1  # only the >=3-word claim was checked


def test_empty_token_stream_is_safe() -> None:
    outcome = stream_until_contradiction([], _check_on("x"))
    assert outcome.halted is False
    assert outcome.token_count == 0
    assert outcome.claims_checked == 0


def test_claim_without_terminator_is_never_checked() -> None:
    # No sentence-ending punctuation -> no claim boundary -> no check.
    tokens = _tokenize_simple("this passage never ends with punctuation so no halt")
    outcome = stream_until_contradiction(tokens, _check_on("halt"))
    assert outcome.halted is False
    assert outcome.claims_checked == 0


# --------------------------------------------------------------------------- #
# _aggregate                                                                  #
# --------------------------------------------------------------------------- #


def _good(halted: bool) -> StreamHaltOutcome:
    return StreamHaltOutcome(halted, -1, 0.0, "", 10, 1)


def _bad(halted: bool, halt_index: int) -> StreamHaltOutcome:
    return StreamHaltOutcome(halted, halt_index, 0.9, "f", halt_index + 1, 1)


def test_aggregate_perfect_separation() -> None:
    good = [_good(False), _good(False), _good(False)]
    bad = [(_bad(True, 12), 10), (_bad(True, 9), 10)]
    m = _aggregate(good, bad)
    assert m["false_halt_rate"] == 0.0
    assert m["halt_recall"] == 1.0
    assert m["halt_precision"] == 1.0
    assert m["true_negatives"] == 3
    assert m["false_negatives"] == 0
    assert m["token_of_halt_accuracy"] == 1.0  # both within ±8 tokens
    assert m["median_halt_latency_tokens"] == 0.5  # latencies +2 and -1 -> median 0.5


def test_aggregate_false_halts_and_misses() -> None:
    good = [_good(True), _good(False), _good(False), _good(False)]  # 1/4 false-halt
    bad = [(_bad(True, 10), 10), (_bad(False, -1), 10)]  # 1/2 recall
    m = _aggregate(good, bad)
    assert m["false_halt_rate"] == 0.25
    assert m["halt_recall"] == 0.5
    assert m["false_positives"] == 1
    assert m["true_positives"] == 1
    assert m["halt_precision"] == 0.5  # 1 TP / (1 TP + 1 FP)
    assert m["false_negatives"] == 1


def test_aggregate_late_halt_outside_tolerance_lowers_accuracy() -> None:
    bad = [(_bad(True, 100), 10)]  # latency +90, outside ±8
    m = _aggregate([_good(False)], bad)
    assert m["halt_recall"] == 1.0
    assert m["token_of_halt_accuracy"] == 0.0


def test_aggregate_empty_sets_are_safe() -> None:
    m = _aggregate([], [])
    assert m["false_halt_rate"] == 0.0
    assert m["halt_recall"] == 0.0
    assert m["halt_precision"] == 0.0


def test_runtime_metadata_labels_non_isolated_evidence() -> None:
    meta = _runtime_metadata()
    assert meta["isolation"] == "non_isolated_local_regression"
    assert isinstance(meta["command"], list)
    assert meta["python"]
    assert meta["platform"]
