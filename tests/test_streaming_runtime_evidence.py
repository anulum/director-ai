# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real stream evidence, exporters and structured recovery
"""Exercise streaming evidence plumbing without claiming heuristic NLI accuracy."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from director_ai.core.observability.callbacks import TokenTraceCallback, TokenTraceEvent
from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.runtime.streaming import StreamingKernel
from director_ai.core.runtime.structured_recovery import StructuredRecoveryConfig
from director_ai.core.scoring.scorer import CoherenceScorer


class _FileTraceCallback(TokenTraceCallback):
    """Consume the actual public callback protocol by writing a real JSONL file."""

    def __init__(self, path: Path) -> None:
        """Bind the caller-owned output path.

        Parameters
        ----------
        path : Path
            File receiving token hashes and stream summaries.
        """
        self.path = path

    def on_token(self, event: TokenTraceEvent) -> None:
        """Persist one delivered event without retaining its raw token.

        Parameters
        ----------
        event : TokenTraceEvent
            Actual event delivered by the streaming kernel.
        """
        with self.path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "type": "token",
                        "index": event.index,
                        "token_hash": event.token_hash(),
                        "halted": event.halted,
                        "reason": event.halt_reason,
                        "tenant_id": event.tenant_id,
                        "request_id": event.request_id,
                    }
                )
                + "\n"
            )

    def on_stream_end(
        self, *, tenant_id: str, request_id: str, summary: dict[str, Any]
    ) -> None:
        """Persist the actual terminal summary received from the emitter.

        Parameters
        ----------
        tenant_id : str
            Tenant supplied to the stream entry point.
        request_id : str
            Request supplied to the stream entry point.
        summary : dict of str to Any
            Unmodified callback-protocol summary.
        """
        with self.path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "type": "end",
                        "tenant_id": tenant_id,
                        "request_id": request_id,
                        "summary": summary,
                    }
                )
                + "\n"
            )


@pytest.fixture(scope="module")
def span_exporter() -> Iterator[InMemorySpanExporter]:
    """Configure the real SDK through its public provider API for this process."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    assert trace.get_tracer_provider() is provider
    try:
        yield exporter
    finally:
        processor.shutdown()


@pytest.mark.parametrize("with_fact", [False, True])
@pytest.mark.parametrize("blank_prompt", [False, True])
@pytest.mark.parametrize(
    ("scores", "reason", "threshold", "margin"),
    [
        ([0.1], "hard_limit", 0.5, 0.4),
        ([0.54, 0.54, 0.54], "window_avg", 0.55, 0.01),
        ([0.95, 0.85, 0.55], "downward_trend", 0.2, 0.2),
    ],
)
def test_real_scorer_store_and_sdk_preserve_halt_attribution_without_raw_text(
    span_exporter: InMemorySpanExporter,
    with_fact: bool,
    blank_prompt: bool,
    scores: list[float],
    reason: str,
    threshold: float,
    margin: float,
) -> None:
    """Verify evidence/trace transport with real heuristic scoring, not NLI proof."""
    store = GroundTruthStore()
    if with_fact:
        store.add("reference", "PRIVATE stored reference fact", tenant_id="tenant-one")
    scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
    values = iter(scores)
    request = f"stream-{reason}-{with_fact}-{blank_prompt}"
    offset = len(span_exporter.get_finished_spans())
    kernel = StreamingKernel(window_size=3, trend_window=3, trend_threshold=0.2)
    session = kernel.stream_tokens(
        ["PRIVATE reference token"] * len(scores),
        lambda _text: next(values),
        scorer=scorer,
        prompt="" if blank_prompt else "reference",
        tenant_id="tenant-one",
        request_id=request,
    )
    evidence = session.halt_evidence_structured
    assert session.halted
    assert evidence is not None
    assert evidence.reason.startswith(reason)
    assert evidence.nli_scores is None
    assert len(evidence.evidence_chunks) == int(with_fact)
    assert evidence.trace_attribution is not None
    assert evidence.trace_attribution.fact_source == ("keyword" if with_fact else "")
    assert evidence.trace_attribution.retrieval_path == (
        "scorer.review.evidence.chunks" if with_fact else "scorer.review.no_chunks"
    )
    assert evidence.trace_attribution.scorer_path == (
        "director_ai.core.scoring.scorer.CoherenceScorer.review"
    )
    assert evidence.trace_attribution.token_offset == len(scores) - 1
    assert evidence.trace_attribution.threshold == threshold
    assert evidence.trace_attribution.causal_contribution == pytest.approx(margin)
    assert session.events[-1].halt_evidence is evidence
    assert len(session.safety_events) == 1
    assert session.events[-1].safety_event is session.safety_events[0]

    spans = span_exporter.get_finished_spans()[offset:]
    token_spans = [span for span in spans if span.name == "director_ai.stream.token"]
    summary_spans = [span for span in spans if span.name == "director_ai.stream"]
    assert len(token_spans) == len(scores)
    assert len(summary_spans) == 1
    attributes = summary_spans[0].attributes
    assert attributes is not None
    assert attributes["stream.halt_cause.threshold"] == threshold
    assert attributes["stream.halt_cause.margin"] == pytest.approx(margin)
    assert attributes["stream.counterfactual.available"] is True
    assert attributes["stream.counterfactual.candidate_count"] == int(with_fact)
    for span in token_spans:
        assert span.attributes is not None
        assert span.attributes["tenant.id"] == "tenant-one"
        assert span.attributes["request.id"] == request
    encoded = json.dumps([dict(span.attributes or {}) for span in spans])
    assert "PRIVATE reference token" not in encoded
    assert "PRIVATE stored reference fact" not in encoded


@pytest.mark.parametrize(
    ("window_size", "scores", "reason", "threshold", "margin"),
    [
        (2, [0.5, 0.5, 0.9, 0.9], "window_avg", 0.6, 0.1),
        (10, [0.95, 0.85, 0.55, 0.95, 0.95], "downward_trend", 0.2, 0.2),
    ],
)
def test_soft_halt_evidence_describes_the_original_decision(
    span_exporter: InMemorySpanExporter,
    window_size: int,
    scores: list[float],
    reason: str,
    threshold: float,
    margin: float,
) -> None:
    """Evidence, diagnostics and telemetry share the same trigger prefix."""
    store = GroundTruthStore()
    store.add("reference", "reference fact")
    store.add("afterward", "afterward fact")
    scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
    trigger_tokens = ["reference "] + ["word "] * (len(scores) - 3)
    tokens = [*trigger_tokens, "afterward ", "."]
    values = iter(scores)
    offset = len(span_exporter.get_finished_spans())
    session = StreamingKernel(
        hard_limit=0.3,
        window_size=window_size,
        window_threshold=0.6,
        trend_window=3,
        trend_threshold=0.2,
        halt_mode="soft",
    ).stream_tokens(
        tokens, lambda _text: next(values), evidence_callback=str, scorer=scorer
    )

    assert session.halted and session.soft_halted
    assert session.output == "".join(tokens)
    assert session.halt_index == len(trigger_tokens) - 1
    assert session.events[-1].coherence == scores[-1]
    evidence = session.halt_evidence_structured
    assert evidence is not None
    assert evidence.reason.startswith(reason)
    attribution = evidence.trace_attribution
    assert attribution is not None
    assert attribution.token_offset == session.halt_index
    assert attribution.threshold == threshold
    assert attribution.causal_contribution == pytest.approx(margin)
    assert session.halt_evidence == "".join(trigger_tokens)
    assert [chunk.text for chunk in evidence.evidence_chunks] == ["reference fact"]
    diagnostic = evidence.counterfactual_diagnostic
    assert diagnostic is not None
    assert diagnostic.observed_score == scores[session.halt_index]

    [summary] = [
        span
        for span in span_exporter.get_finished_spans()[offset:]
        if span.name == "director_ai.stream"
    ]
    assert summary.attributes is not None
    assert summary.attributes["stream.halt_cause.margin"] == pytest.approx(margin)
    assert summary.attributes["stream.halt_cause.token_offset"] == session.halt_index


def test_file_callback_records_tokens_halt_and_summary_after_real_evidence_read(
    tmp_path: Path,
) -> None:
    """Exercise a concrete filesystem consumer and host evidence callback."""
    note = tmp_path / "evidence.txt"
    note.write_text("Actual host evidence note", encoding="utf-8")

    def read_evidence(_text: str) -> str:
        """Read the host-owned evidence file only when the stream actually halts."""
        return note.read_text(encoding="utf-8")

    path = tmp_path / "tokens.jsonl"
    values = iter([0.9, 0.1])
    session = StreamingKernel().stream_tokens(
        ["good", "bad", "unused"],
        lambda _text: next(values),
        evidence_callback=read_evidence,
        trace_callbacks=[_FileTraceCallback(path)],
        tenant_id="tenant-one",
        request_id="request-one",
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert session.halt_evidence == session.events[-1].evidence == note.read_text()
    assert [record["type"] for record in records] == ["token", "token", "end"]
    assert records[1]["halted"] is True
    assert records[1]["reason"].startswith("hard_limit")
    assert records[2]["summary"]["halted"] is True
    assert records[2]["summary"]["token_count"] == 2
    assert all(record["request_id"] == "request-one" for record in records)
    assert all(record["tenant_id"] == "tenant-one" for record in records)
    assert "token" not in records[0]
    assert len(records[0]["token_hash"]) == 16


def test_empty_stream_still_delivers_one_real_terminal_callback(tmp_path: Path) -> None:
    """An empty token source must produce a zero-score summary without tokens."""
    path = tmp_path / "empty.jsonl"
    session = StreamingKernel().stream_tokens(
        [], float, trace_callbacks=[_FileTraceCallback(path)]
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert not session.halted
    assert len(records) == 1
    assert records[0]["type"] == "end"
    assert records[0]["summary"]["avg_coherence"] == 0.0


def test_actual_trace_file_failure_does_not_abort_the_scored_stream(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Opening a directory as a file fails through real IO in both callback hooks."""
    session = StreamingKernel().stream_tokens(
        ["safe"], lambda _text: 0.9, trace_callbacks=[_FileTraceCallback(tmp_path)]
    )
    assert not session.halted
    assert session.output == "safe"
    assert "token trace callback _FileTraceCallback raised" in caplog.text
    assert "token trace callback _FileTraceCallback end raised" in caplog.text


@pytest.mark.parametrize("policy", ["last_valid", "redacted", "raw_partial"])
def test_real_json_recovery_retains_only_the_requested_output_policy(
    policy: str,
) -> None:
    """Recover an actual parsed prefix after the next token breaks JSON and halts."""
    values = iter([0.9, 0.1])
    session = StreamingKernel().stream_tokens(
        ['{"answer":1}', " trailing"],
        lambda _text: next(values),
        structured_recovery=StructuredRecoveryConfig(kind="json", policy=policy),
    )
    recovery = session.structured_recovery
    assert session.halted
    assert recovery is not None
    assert recovery.halted_at == session.halt_index == 1
    assert recovery.valid is (policy == "last_valid")
    assert recovery.last_valid_output == (
        '{"answer":1}' if policy == "last_valid" else ""
    )
    assert recovery.raw_partial == (
        '{"answer":1} trailing' if policy == "raw_partial" else ""
    )
    assert recovery.metadata == {"json_root": "object", "field_errors": 0}
    assert recovery.errors[0].startswith("json_parse:")
    assert session.output == '{"answer":1}'


@pytest.mark.parametrize("tenant", ["tenant-one", "tenant-two", "tenant-empty"])
def test_halt_evidence_retrieval_retains_the_streams_actual_tenant(tenant: str) -> None:
    """The halt packet must not merge another tenant's matching private facts."""
    facts = {
        "tenant-one": "TENANT ONE PRIVATE reference fact",
        "tenant-two": "TENANT TWO PRIVATE reference fact",
    }
    store = GroundTruthStore()
    for identity, fact in facts.items():
        store.add("reference", fact, tenant_id=identity)
    scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
    session = StreamingKernel().stream_tokens(
        ["reference"],
        lambda _text: 0.1,
        scorer=scorer,
        prompt="reference",
        tenant_id=tenant,
    )
    evidence = session.halt_evidence_structured
    assert evidence is not None
    assert [chunk.text for chunk in evidence.evidence_chunks] == (
        [facts[tenant]] if tenant in facts else []
    )


@pytest.mark.parametrize(
    "order", [("tenant-one", "tenant-two"), ("tenant-two", "tenant-one")]
)
@pytest.mark.parametrize("warm", [False, True])
def test_shared_scorer_cache_keeps_tenant_evidence_separate(
    order: tuple[str, str],
    warm: bool,
) -> None:
    """Identical prompt/output pairs cannot share evidence across tenant scopes."""
    store = GroundTruthStore()
    for tenant in order:
        store.add("reference", f"{tenant} reference fact", tenant_id=tenant)
    scorer = CoherenceScorer(use_nli=False, ground_truth_store=store, cache_size=8)
    cache = scorer.cache
    assert cache is not None
    if warm:
        for tenant in order:
            scorer.review("reference", "reference", tenant_id=tenant)
            scorer.review("reference", "reference", tenant_id=tenant)
        assert cache.hits == len(order)
    hits_before = cache.hits
    for tenant in order:
        session = StreamingKernel().stream_tokens(
            ["reference"],
            lambda _text: 0.1,
            scorer=scorer,
            prompt="reference",
            tenant_id=tenant,
        )
        evidence = session.halt_evidence_structured
        assert evidence is not None
        assert [chunk.text for chunk in evidence.evidence_chunks] == [
            f"{tenant} reference fact"
        ]
    assert cache.hits == hits_before
    if warm:
        for tenant in order:
            scorer.review("reference", "reference", tenant_id=tenant)
        assert cache.hits == hits_before + len(order)


def test_unscoped_real_scorer_retains_the_legacy_two_argument_path() -> None:
    """Explicitly unscoped callers still retrieve their existing global facts."""
    store = GroundTruthStore()
    store.add("reference", "global reference fact")
    session = StreamingKernel().stream_tokens(
        ["reference"],
        lambda _text: 0.1,
        scorer=CoherenceScorer(use_nli=False, ground_truth_store=store),
        prompt="reference",
    )
    evidence = session.halt_evidence_structured
    assert evidence is not None
    assert [chunk.text for chunk in evidence.evidence_chunks] == [
        "global reference fact"
    ]


@pytest.mark.parametrize("backend", ["deberta", "rust", "backfire"])
def test_factual_scores_ignore_other_tenants_knowledge_changes(
    span_exporter: InMemorySpanExporter, backend: str
) -> None:
    """Unrelated tenant facts cannot change a tenant's factual divergence."""
    offset = len(span_exporter.get_finished_spans())
    store = GroundTruthStore()
    scorer = CoherenceScorer(
        use_nli=False, scorer_backend=backend, ground_truth_store=store
    )
    _, before = scorer.review(
        "reference", "reference alpha fact", tenant_id="tenant-with-no-facts"
    )
    store.add("reference", "reference alpha fact", tenant_id="other-tenant")
    _, after = scorer.review(
        "reference", "reference alpha fact", tenant_id="tenant-with-no-facts"
    )
    if backend in ("rust", "backfire"):
        spans = span_exporter.get_finished_spans()[offset:]
        assert any(
            (span.attributes or {}).get("coherence.backend") == "rust" for span in spans
        )
    assert after.h_factual == pytest.approx(before.h_factual)


@pytest.mark.parametrize("backend", ["deberta", "rust", "backfire"])
def test_model_required_policy_refuses_both_real_heuristic_backends(
    backend: str,
) -> None:
    """A native implementation without model weights cannot satisfy model policy."""
    scorer = CoherenceScorer(
        use_nli=False,
        scorer_backend=backend,
        require_model_backed_nli=True,
    )
    with pytest.raises(RuntimeError, match="requires model-backed NLI"):
        scorer.review("reference", "reference alpha fact")
