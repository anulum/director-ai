# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cross-model consensus tests

from __future__ import annotations

import pytest

from director_ai.core.consensus import (
    ConsensusResult,
    CrossModelConsensus,
    Divergence,
    ModelResponse,
)
from director_ai.core.consensus.cross_model_consensus import (
    _lexical_overlap,
    _split_claims,
)


class _StubNLI:
    """Directional NLI stub: contradiction looked up by (premise, hypothesis).

    Missing pairs score ``0.0`` (no contradiction). The lookup is intentionally
    directional so the engine's ``max`` over both directions is exercised.
    """

    threshold = 0.5

    def __init__(self, table: dict[tuple[str, str], float] | None = None):
        self._table = table or {}

    def contradiction(self, premise: str, hypothesis: str) -> float:
        return self._table.get((premise, hypothesis), 0.0)


def _responses(*pairs: tuple[str, str]) -> list[ModelResponse]:
    return [ModelResponse(model_id=m, text=t) for m, t in pairs]


# ── _split_claims ──────────────────────────────────────────────────────────


def test_split_claims_keeps_sentences_drops_fragments():
    text = "Paris is the capital of France. Yes. The Seine flows through it."
    claims = _split_claims(text)
    assert claims == [
        "Paris is the capital of France.",
        "The Seine flows through it.",
    ]


def test_split_claims_caps_count():
    text = " ".join(f"Sentence number {i} has content." for i in range(40))
    assert len(_split_claims(text, cap=5)) == 5


def test_split_claims_empty():
    assert _split_claims("   ") == []


# ── _lexical_overlap (Rust/Python parity) ──────────────────────────────────


def test_lexical_overlap_identical_and_disjoint():
    assert _lexical_overlap("alpha beta gamma", "alpha beta gamma") == 1.0
    assert _lexical_overlap("alpha beta", "delta epsilon") == 0.0


def test_lexical_overlap_empty_inputs():
    assert _lexical_overlap("", "") == 0.0
    assert _lexical_overlap("word", "") == 0.0


_PARITY_CASES = [
    ("the cat sat on the mat", "the dog sat on the mat"),
    ("one two three four", "three four five six"),
    ("Disjoint Tokens Here", "completely different words"),
    ("repeated repeated word", "word repeated"),
]


@pytest.mark.parametrize("a,b", _PARITY_CASES)
def test_lexical_overlap_rust_python_parity(a, b):
    """The dispatched value matches the pure-Python Jaccard exactly."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    union = words_a | words_b
    expected = len(words_a & words_b) / len(union) if union else 0.0
    assert _lexical_overlap(a, b) == pytest.approx(expected)


def test_lexical_overlap_empty_returns_zero():
    # _lexical_overlap delegates to the shared text_overlap helper; dispatch and
    # Python/Rust parity are covered by test_text_overlap.
    assert _lexical_overlap("", "") == 0.0
    assert _lexical_overlap("only one side", "") == 0.0


# ── agreement ──────────────────────────────────────────────────────────────


def test_agreement_semantic_uses_directional_max():
    nli = _StubNLI({("A", "B"): 0.9, ("B", "A"): 0.1})
    engine = CrossModelConsensus(nli=nli)
    # max(0.9, 0.1) = 0.9 contradiction -> 0.1 agreement
    assert engine.agreement("A", "B") == pytest.approx(0.1)


def test_agreement_lexical_without_nli():
    engine = CrossModelConsensus()
    assert engine.agreement("alpha beta", "alpha beta") == 1.0
    assert engine.agreement("alpha beta", "gamma delta") == 0.0


def test_agreement_is_clamped():
    nli = _StubNLI({("A", "B"): 1.5})  # out-of-range contradiction
    engine = CrossModelConsensus(nli=nli)
    assert engine.agreement("A", "B") == 0.0


# ── consensus: high agreement ──────────────────────────────────────────────


def test_consensus_identical_answers_accept():
    engine = CrossModelConsensus(nli=_StubNLI())
    res = engine.consensus(
        _responses(
            ("gpt", "The capital of France is Paris."),
            ("claude", "The capital of France is Paris."),
            ("gemini", "The capital of France is Paris."),
        )
    )
    assert isinstance(res, ConsensusResult)
    assert res.consensus == 1.0
    assert res.recommendation == "accept"
    assert res.divergences == ()
    assert res.n_models == 3
    assert "models broadly agree" in res.rationale


# ── consensus: contradiction -> escalate with evidence ─────────────────────


def test_consensus_contradiction_escalates_with_evidence():
    claim_a = "The treaty was signed in 1920."
    claim_b = "The treaty was signed in 1919."
    nli = _StubNLI({(claim_a, claim_b): 0.95})
    engine = CrossModelConsensus(nli=nli)
    res = engine.consensus(_responses(("gpt", claim_a), ("claude", claim_b)))
    assert res.recommendation == "escalate"
    assert res.consensus == pytest.approx(0.05)
    assert len(res.divergences) == 1
    div = res.divergences[0]
    assert isinstance(div, Divergence)
    assert div.model_a == "gpt" and div.model_b == "claude"
    assert div.claim_a == claim_a and div.claim_b == claim_b
    assert div.contradiction == pytest.approx(0.95)
    assert "1 contradicting claim pair(s)" in res.rationale


def test_consensus_partial_agreement_review():
    nli = _StubNLI({("X", "Y"): 0.4})  # below flag threshold 0.5
    engine = CrossModelConsensus(nli=nli, accept_threshold=0.8, escalate_threshold=0.5)
    res = engine.consensus(_responses(("gpt", "X"), ("claude", "Y")))
    # agreement 0.6 -> between escalate (0.5) and accept (0.8)
    assert res.recommendation == "review"
    assert res.divergences == ()  # 0.4 < 0.5 flag threshold
    assert "partial agreement" in res.rationale


# ── matrix shape ───────────────────────────────────────────────────────────


def test_agreement_matrix_is_symmetric_with_unit_diagonal():
    engine = CrossModelConsensus(nli=_StubNLI({("A", "B"): 0.8}))
    res = engine.consensus(_responses(("m1", "A"), ("m2", "B"), ("m3", "C")))
    m = res.agreement_matrix
    assert len(m) == 3 and all(len(row) == 3 for row in m)
    for i in range(3):
        assert m[i][i] == 1.0
        for j in range(3):
            assert m[i][j] == m[j][i]


# ── divergence cap and ordering ────────────────────────────────────────────


def test_divergences_capped_and_sorted_strongest_first():
    a = "Claim one here. Claim two here. Claim three here."
    b = "Counter one here. Counter two here. Counter three here."
    table = {
        ("Claim one here.", "Counter one here."): 0.6,
        ("Claim two here.", "Counter two here."): 0.95,
        ("Claim three here.", "Counter three here."): 0.7,
    }
    engine = CrossModelConsensus(nli=_StubNLI(table), max_divergences=2)
    res = engine.consensus(_responses(("gpt", a), ("claude", b)))
    assert len(res.divergences) == 2
    assert res.divergences[0].contradiction == pytest.approx(0.95)
    assert res.divergences[1].contradiction == pytest.approx(0.7)


# ── lexical-only divergence fallback ───────────────────────────────────────


def test_lexical_only_surfaces_weakest_pair_as_divergence():
    engine = CrossModelConsensus()  # no NLI
    res = engine.consensus(
        _responses(
            ("gpt", "alpha beta gamma delta"),
            ("claude", "alpha beta gamma delta"),
            ("llama", "completely unrelated tokens entirely"),
        )
    )
    assert len(res.divergences) == 1
    div = res.divergences[0]
    assert {div.model_a, div.model_b} == {"gpt", "llama"} or {
        div.model_a,
        div.model_b,
    } == {"claude", "llama"}
    assert "lexical agreement only (no NLI scorer supplied)" in res.rationale


# ── divergence_threshold default falls back to nli.threshold ───────────────


def test_flag_threshold_defaults_to_scorer_threshold():
    engine = CrossModelConsensus(nli=_StubNLI())
    assert engine._flag_threshold == 0.5  # the stub's threshold


def test_flag_threshold_explicit_override():
    engine = CrossModelConsensus(nli=_StubNLI(), divergence_threshold=0.3)
    assert engine._flag_threshold == 0.3


def test_flag_threshold_lexical_default():
    assert CrossModelConsensus()._flag_threshold == 0.5


# ── validation ─────────────────────────────────────────────────────────────


def test_consensus_requires_two_responses():
    engine = CrossModelConsensus()
    with pytest.raises(ValueError, match="at least two"):
        engine.consensus(_responses(("solo", "only one answer here")))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"accept_threshold": 0.3, "escalate_threshold": 0.5},  # escalate > accept
        {"escalate_threshold": -0.1},
        {"accept_threshold": 1.5},
        {"divergence_threshold": 1.2},
        {"max_divergences": -1},
    ],
)
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        CrossModelConsensus(**kwargs)


def test_valid_boundary_construction():
    engine = CrossModelConsensus(
        accept_threshold=0.5,
        escalate_threshold=0.5,
        divergence_threshold=0.0,
        max_divergences=0,
    )
    assert engine.accept_threshold == 0.5


# ── ProductionGuard wiring ──────────────────────────────────────────────────


def _lexical_guard():
    from director_ai.core.config import DirectorConfig
    from director_ai.guard import ProductionGuard

    return ProductionGuard(config=DirectorConfig(use_nli=False))


def test_guard_cross_model_consensus_lexical():
    guard = _lexical_guard()
    res = guard.cross_model_consensus(
        _responses(
            ("gpt", "The capital of France is Paris."),
            ("claude", "The capital of France is Paris."),
        )
    )
    assert isinstance(res, ConsensusResult)
    assert res.recommendation == "accept"
    assert res.consensus == 1.0


def test_guard_cross_model_consensus_semantic_with_injected_nli():
    guard = _lexical_guard()
    a, b = "The launch was in 2021.", "The launch was in 1999."
    nli = _StubNLI({(a, b): 0.92})
    res = guard.cross_model_consensus(_responses(("gpt", a), ("claude", b)), nli=nli)
    assert res.recommendation == "escalate"
    assert len(res.divergences) == 1
    assert res.divergences[0].contradiction == pytest.approx(0.92)


def test_guard_cross_model_consensus_engine_persists():
    guard = _lexical_guard()
    guard.cross_model_consensus(_responses(("a", "x y z"), ("b", "x y z")))
    first = guard._cross_model
    guard.cross_model_consensus(_responses(("a", "x y z"), ("b", "x y z")))
    assert guard._cross_model is first  # reused when no new nli supplied
