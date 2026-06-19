# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the OpenTelemetry eval trace standard.

Covers the span-attribute schema, the span emitter (no-op and tracer-present
paths), the GuardResult-to-record builder, and the ProductionGuard integration.
"""

from __future__ import annotations

from contextlib import contextmanager

from director_ai.core import eval_trace
from director_ai.core.eval_trace import (
    EVAL_SCHEMA_VERSION,
    eval_record_from_guard,
    guard_decision_attributes,
    record_guard_decision,
)
from director_ai.core.types import (
    ClaimAttribution,
    CoherenceScore,
    EvidenceChunk,
    ScoringEvidence,
)
from director_ai.guard import GuardResult

# ── attribute schema ────────────────────────────────────────────────────


class TestAttributes:
    def test_core_fields(self):
        attrs = guard_decision_attributes(
            decision="halt", approved=False, score=0.4, threshold=0.6
        )
        assert attrs["director.eval.schema_version"] == EVAL_SCHEMA_VERSION
        assert attrs["director.eval.decision"] == "halt"
        assert attrs["director.eval.approved"] is False
        assert attrs["director.eval.score"] == 0.4
        assert attrs["director.eval.threshold"] == 0.6

    def test_gen_ai_conventions(self):
        attrs = guard_decision_attributes(
            decision="allow", approved=True, score=0.9, threshold=0.6, model="gpt-4o"
        )
        assert attrs["gen_ai.system"] == "director-ai"
        assert attrs["gen_ai.operation.name"] == "guard"
        assert attrs["gen_ai.request.model"] == "gpt-4o"

    def test_optional_fields_omitted_when_empty(self):
        attrs = guard_decision_attributes(
            decision="allow", approved=True, score=0.9, threshold=0.6
        )
        assert "gen_ai.request.model" not in attrs
        assert "director.eval.tenant_id" not in attrs
        assert "director.eval.answer_id" not in attrs

    def test_optional_fields_included_when_set(self):
        attrs = guard_decision_attributes(
            decision="allow",
            approved=True,
            score=0.9,
            threshold=0.6,
            scorer="rust",
            tenant_id="acme",
            domain="finance",
            answer_id="abom_1",
        )
        assert attrs["director.eval.scorer"] == "rust"
        assert attrs["director.eval.tenant_id"] == "acme"
        assert attrs["director.eval.domain"] == "finance"
        assert attrs["director.eval.answer_id"] == "abom_1"

    def test_values_are_primitives(self):
        attrs = guard_decision_attributes(
            decision="halt",
            approved=False,
            score=0.4,
            threshold=0.6,
            evidence_count=3,
            unsupported_claims=2,
        )
        assert isinstance(attrs["director.eval.evidence_count"], int)
        assert isinstance(attrs["director.eval.score"], float)
        for value in attrs.values():
            assert isinstance(value, str | int | float | bool)


# ── span emitter ────────────────────────────────────────────────────────


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _RecordingTracer:
    def __init__(self, span: _RecordingSpan) -> None:
        self._span = span

    @contextmanager
    def start_as_current_span(self, name: str):
        self.span_name = name
        yield self._span


class TestSpanEmitter:
    def test_noop_path_yields_span(self, monkeypatch):
        monkeypatch.setattr(eval_trace.tracer, "_get_tracer", lambda: None)
        attrs = guard_decision_attributes(
            decision="allow", approved=True, score=0.9, threshold=0.6
        )
        with record_guard_decision(attrs) as span:
            # Without the OTel SDK a no-op sink is yielded; it accepts attributes.
            span.set_attribute("extra", "x")
        assert span is not None

    def test_tracer_path_sets_attributes(self, monkeypatch):
        recording = _RecordingSpan()
        tracer = _RecordingTracer(recording)
        monkeypatch.setattr(eval_trace.tracer, "_get_tracer", lambda: tracer)
        attrs = guard_decision_attributes(
            decision="halt", approved=False, score=0.4, threshold=0.6
        )
        with record_guard_decision(attrs):
            pass
        assert recording.attributes["director.eval.decision"] == "halt"
        assert tracer.span_name == "director_ai.eval.guard_decision"


# ── builder from GuardResult ────────────────────────────────────────────


def _guard_result(*, approved: bool, score: float) -> GuardResult:
    evidence = ScoringEvidence(
        chunks=[EvidenceChunk(text="t", distance=0.1, source="vector:d1")],
        nli_premise="",
        nli_hypothesis="",
        nli_score=0.0,
        attributions=[
            ClaimAttribution(
                claim="c",
                claim_index=0,
                source_sentence="s",
                source_index=0,
                divergence=0.9,
                supported=False,
            )
        ],
        claims=["c"],
    )
    coherence = CoherenceScore(
        score=score,
        approved=approved,
        h_logical=0.1,
        h_factual=0.1,
        evidence=evidence,
    )
    return GuardResult(
        approved=approved,
        score=score,
        coherence=coherence,
        calibrated_threshold=0.55,
    )


class TestBuilderFromGuard:
    def test_record_from_halted_result(self):
        record = eval_record_from_guard(
            _guard_result(approved=False, score=0.4),
            model="gpt-4o",
            tenant_id="acme",
        )
        assert record["director.eval.decision"] == "halt"
        assert record["director.eval.approved"] is False
        assert record["director.eval.threshold"] == 0.55
        assert record["director.eval.evidence_count"] == 1
        assert record["director.eval.unsupported_claims"] == 1

    def test_record_from_approved_result(self):
        record = eval_record_from_guard(_guard_result(approved=True, score=0.9))
        assert record["director.eval.decision"] == "allow"
        assert record["director.eval.approved"] is True

    def test_threshold_defaults_to_zero_without_calibration(self):
        result = GuardResult(
            approved=True,
            score=0.9,
            coherence=CoherenceScore(
                score=0.9, approved=True, h_logical=0.0, h_factual=0.0
            ),
        )
        record = eval_record_from_guard(result)
        assert record["director.eval.threshold"] == 0.0
        assert record["director.eval.evidence_count"] == 0


# ── guard integration ───────────────────────────────────────────────────


class TestGuardIntegration:
    def test_eval_trace_returns_record(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        result = guard.check("q", "a")
        record = guard.eval_trace(result, model="mock", tenant_id="acme")
        assert record["gen_ai.system"] == "director-ai"
        assert record["director.eval.scorer"] == guard.config.scorer_backend
        assert record["director.eval.tenant_id"] == "acme"

    def test_eval_trace_without_span(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        result = guard.check("q", "a")
        record = guard.eval_trace(result, emit_span=False)
        assert record["director.eval.schema_version"] == EVAL_SCHEMA_VERSION
