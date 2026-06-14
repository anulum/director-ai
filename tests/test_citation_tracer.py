# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — citation tracer tests

"""Claim/citation mapping, coverage and reasoning-integration coverage for the
citation tracer."""

from __future__ import annotations

import pytest

from director_ai.core.verification import citation_tracer as ct
from director_ai.core.verification.citation_tracer import (
    ClaimCitation,
    TraceResult,
    trace_citations,
)
from director_ai.core.verification.reasoning_verifier import verify_reasoning_chain

_TEXT = (
    "The transformer was introduced in 2017 [1]. "
    "It relies on self-attention to model dependencies. "
    "Diffusion models came later (Ho et al., 2020).\n\n"
    "References:\n"
    "[1] https://arxiv.org/abs/1706.03762\n"
)


def test_sentence_spans_offsets_and_filtering():
    body = "First claim here. Second one too.   \n  "
    spans = ct._sentence_spans(body)
    assert [t for _, _, t in spans] == ["First claim here.", "Second one too."]
    for start, end, text in spans:
        assert body[start:end] == text


def test_sentence_spans_trailing_sentence_without_terminator():
    spans = ct._sentence_spans("One sentence. A trailing clause with no period")
    assert spans[-1][2] == "A trailing clause with no period"


def test_trace_attaches_citations_and_excludes_references():
    result = trace_citations(_TEXT)
    assert isinstance(result, TraceResult)
    assert len(result.claims) == 3
    assert result.claims[0].cited is True
    assert result.claims[0].citations[0].identifier == "1706.03762"
    assert result.claims[1].cited is False
    assert result.claims[2].citations[0].identifier == "Ho et al. 2020"


def test_coverage_and_uncited():
    result = trace_citations(_TEXT)
    assert result.coverage == pytest.approx(2 / 3)
    assert [c.index for c in result.uncited] == [1]
    assert [c.index for c in result.cited] == [0, 2]


def test_empty_text_has_zero_coverage():
    result = trace_citations("")
    assert result.claims == []
    assert result.coverage == 0.0


def test_claim_citation_dataclass_default():
    claim = ClaimCitation(index=0, claim="x", citations=())
    assert claim.cited is False


def test_reasoning_chain_citation_trace_opt_in():
    enabled = verify_reasoning_chain(_TEXT, check_citations=True)
    assert enabled.citation_trace is not None
    assert enabled.citation_trace.coverage == pytest.approx(2 / 3)
    assert verify_reasoning_chain(_TEXT).citation_trace is None


def test_reasoning_chain_citation_trace_does_not_affect_validity():
    with_trace = verify_reasoning_chain(_TEXT, check_citations=True)
    without = verify_reasoning_chain(_TEXT, check_citations=False)
    assert with_trace.chain_valid == without.chain_valid
    assert with_trace.issues_found == without.issues_found
