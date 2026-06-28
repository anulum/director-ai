# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - streaming runtime real-surface tests
"""Real streaming runtime tests for halt evidence emitted by the token kernel."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import pytest

import director_ai.core.runtime.streaming as streaming_mod
from director_ai.core.observability.callbacks import TokenTraceCallback, TokenTraceEvent
from director_ai.core.runtime.streaming import StreamingKernel
from director_ai.core.types import (
    CoherenceScore,
    EvidenceChunk,
    HaltEvidence,
    HaltTraceAttribution,
    ScoringEvidence,
)

if TYPE_CHECKING:
    from director_ai.core.scoring.scorer import CoherenceScorer


class _ChunkScoreScorer:
    """Production-shaped scorer returning per-chunk NLI scores."""

    def review(self, prompt: str, response: str) -> tuple[bool, CoherenceScore]:
        """Return a score whose chunks intentionally arrive out of distance order."""
        del prompt, response
        evidence = ScoringEvidence(
            chunks=[
                EvidenceChunk(text="far retrieved fact", distance=0.7, source="far"),
                EvidenceChunk(text="near retrieved fact", distance=0.1, source="near"),
                EvidenceChunk(
                    text="middle retrieved fact", distance=0.4, source="middle"
                ),
            ],
            nli_premise="near retrieved fact\nmiddle retrieved fact\nfar retrieved fact",
            nli_hypothesis="unsafe streamed claim",
            nli_score=0.2,
            chunk_scores=[0.7, 0.1, 0.4],
        )
        return False, CoherenceScore(
            score=0.2,
            approved=False,
            h_logical=0.8,
            h_factual=0.7,
            evidence=evidence,
        )


class _RecordingCallback(TokenTraceCallback):
    """Trace callback that records token events from the streaming runtime."""

    def __init__(self) -> None:
        """Initialise the callback with an empty event buffer."""
        self.events: list[TokenTraceEvent] = []

    def on_token(self, event: TokenTraceEvent) -> None:
        """Store each emitted token trace event."""
        self.events.append(event)


class _AttributeSpan:
    """Minimal telemetry span that records assigned attributes."""

    def __init__(self) -> None:
        """Initialise the span attribute buffer."""
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        """Record one telemetry attribute assignment."""
        self.attributes[key] = value


def test_streaming_halt_evidence_keeps_chunk_scores_aligned_to_top_chunks() -> None:
    """Structured halt evidence should keep per-chunk scores with ranked chunks."""
    kernel = StreamingKernel(hard_limit=0.99)

    session = kernel.stream_tokens(
        ["unsafe"],
        lambda _text: 0.2,
        scorer=cast("CoherenceScorer", _ChunkScoreScorer()),
        top_k=2,
        prompt="operator prompt",
    )

    evidence = session.halt_evidence_structured
    assert evidence is not None
    assert [chunk.source for chunk in evidence.evidence_chunks] == ["near", "middle"]
    assert evidence.nli_scores == [0.1, 0.4]


def test_streaming_token_trace_emits_when_span_has_no_attribute_setter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token callbacks should still run when the telemetry span is attribute-only."""

    @contextmanager
    def no_attribute_trace_token(
        index: int,
        *,
        token: str,
        tenant_id: str = "",
        request_id: str = "",
    ) -> Iterator[object]:
        del index, token, tenant_id, request_id
        yield object()

    monkeypatch.setattr(streaming_mod, "trace_token", no_attribute_trace_token)
    callback = _RecordingCallback()

    session = StreamingKernel(hard_limit=0.1).stream_tokens(
        ["safe"],
        lambda _text: 0.8,
        trace_callbacks=[callback],
        tenant_id="tenant-a",
        request_id="request-a",
    )

    assert session.halted is False
    assert [event.token for event in callback.events] == ["safe"]


def test_halt_telemetry_omits_missing_threshold_without_dropping_margin() -> None:
    """Halt evidence serialization should tolerate threshold-less trace causes."""
    span = _AttributeSpan()
    evidence = HaltEvidence(
        reason="manual_stop",
        last_score=0.2,
        evidence_chunks=[],
        trace_attribution=HaltTraceAttribution(
            fact_source="manual",
            retrieval_path="manual.review",
            scorer_path="manual.scorer",
            token_offset=3,
            threshold=None,
            causal_contribution=0.4,
        ),
    )

    StreamingKernel._set_halt_otel_attributes(span, evidence)

    assert "stream.halt_cause.threshold" not in span.attributes
    assert span.attributes["stream.halt_cause.margin"] == 0.4
    assert span.attributes["stream.counterfactual.available"] is False
