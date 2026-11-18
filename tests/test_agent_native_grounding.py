# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — HTTP generation and agent native-grounding integration
"""Exercise real HTTP generation, agent entry points and native grounding."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import aclosing
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import AsyncGeneratorType
from typing import Self

import pytest

from director_ai.core.actor import LLMGenerator
from director_ai.core.agent import AgentStreamEvent, CoherenceAgent
from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.runtime.streaming import StreamingKernel
from director_ai.core.scoring.scorer import CoherenceScorer
from director_ai.core.types import ReviewResult

PROMPT = "reference"
ACTION = "alpha reference."
STREAM_TOKENS = ("The approved ", "reference is alpha.")
CONTINUATION_PROMPT = "reference continuation"
CONTINUATION_TOKENS = (*STREAM_TOKENS, " More")
FRAGMENT_PROMPT = "assay mass"
FRAGMENT_TOKENS = ("The approved result is 1", "25 milligrams.")
WHOLE_PROMPT = "whole assay mass"
RETRY_PROMPT = "retry assay mass"
WHOLE_TEXT = "\u2003The approved result is\t125  milligrams.\n"


@pytest.fixture
def upstream() -> Iterator[tuple[str, list[dict[str, object]]]]:
    """Serve controlled completion and SSE responses on a real loopback socket."""
    observed: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        """Implement the completion protocol consumed by the real generator."""

        timeout = 2.0

        def do_POST(self) -> None:
            """Record the parsed request and return controlled generated text."""
            body: dict[str, object] = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            observed.append(body)
            prompt = body.get("prompt")
            streaming = body.get("stream", False)
            if streaming and prompt != WHOLE_PROMPT:
                tokens: tuple[str, ...] = STREAM_TOKENS
                if prompt == CONTINUATION_PROMPT:
                    tokens = CONTINUATION_TOKENS
                elif prompt == FRAGMENT_PROMPT:
                    tokens = FRAGMENT_TOKENS
                elif prompt == RETRY_PROMPT:
                    tokens = ()
                payload = "".join(
                    "data: " + json.dumps({"content": token}) + "\n\n"
                    for token in tokens
                ).encode()
                content_type = "text/event-stream"
            else:
                response = (
                    WHOLE_TEXT if prompt in (WHOLE_PROMPT, RETRY_PROMPT) else ACTION
                )
                payload = json.dumps({"content": response}).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            """Keep the controlled upstream from emitting routine access logs."""

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/completion", observed
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)
        assert not worker.is_alive()


def _review(agent: CoherenceAgent, entry: str, tenant: str) -> ReviewResult:
    """Generate and score through the selected real agent entry point."""
    if entry == "async":
        return asyncio.run(agent.aprocess(PROMPT, tenant_id=tenant))
    return agent.process(PROMPT, tenant_id=tenant)


@pytest.mark.parametrize("entry", ["sync", "async"])
def test_http_agent_rejection_retains_evidence_with_warm_cache(
    upstream: tuple[str, list[dict[str, object]]], entry: str
) -> None:
    """Cached verdicts must still carry the grounding for a returned halt."""
    url, observed = upstream
    tenant = "audit-team"
    fact = "The approved reference is beta."
    store = GroundTruthStore()
    store.add(PROMPT, fact, tenant_id=tenant)
    store.add(PROMPT, ACTION, tenant_id="other-team")
    scorer = CoherenceScorer(
        threshold=1.0, use_nli=False, ground_truth_store=store, cache_size=8
    )
    agent = CoherenceAgent(
        llm_api_url=url,
        max_candidates=1,
        production_mode=True,
        _store=store,
        _scorer=scorer,
    )
    cold = _review(agent, entry, tenant)
    warm = _review(agent, entry, tenant)
    assert cold.halted and warm.halted
    assert cold.output == warm.output == "[HALT]: All candidates rejected."
    assert cold.halt_evidence is not None and warm.halt_evidence is not None
    assert warm.halt_evidence.last_score == cold.halt_evidence.last_score
    assert [chunk.text for chunk in cold.halt_evidence.evidence_chunks] == [fact]
    assert [chunk.text for chunk in warm.halt_evidence.evidence_chunks] == [fact]
    assert warm.coherence is not None and warm.coherence.evidence is not None
    assert scorer.cache is not None and scorer.cache.hits == 1
    assert len(observed) == 2


@pytest.mark.parametrize("entry", ["sync", "async"])
def test_http_agent_keeps_foreign_facts_out_of_tenant_scoring(
    upstream: tuple[str, list[dict[str, object]]], entry: str
) -> None:
    """Foreign KB writes cannot change an empty tenant's generated-answer score."""
    url, observed = upstream
    agent = CoherenceAgent(
        llm_api_url=url, use_nli=False, production_mode=True, max_candidates=1
    )
    initial = _review(agent, entry, "empty")
    assert initial.coherence is not None
    assert initial.coherence.h_factual == pytest.approx(0.5)
    agent.store.add(PROMPT, ACTION, tenant_id="owner")
    after = _review(agent, entry, "empty")
    assert after.coherence is not None
    assert after.coherence.h_factual == pytest.approx(0.5)
    assert after.coherence.score == pytest.approx(initial.coherence.score)
    assert after.halted == initial.halted
    owner = _review(agent, entry, "owner")
    assert owner.coherence is not None
    assert owner.coherence.h_factual == pytest.approx(0.0)
    assert isinstance(agent.scorer, CoherenceScorer)
    assert agent.scorer.scorer_backend == "rust"
    assert len(observed) == 3
    assert all(body["prompt"] == PROMPT for body in observed)


def test_http_streaming_agent_keeps_its_tenant_when_a_foreign_fact_is_added(
    upstream: tuple[str, list[dict[str, object]]],
) -> None:
    """The actual SSE consumer retains native grounding through claim scoring."""
    url, observed = upstream
    agent = CoherenceAgent(llm_api_url=url, use_nli=False, production_mode=True)

    async def collect(tenant: str) -> list[tuple[str, float]]:
        """Consume the public async stream until its real upstream closes."""
        stream: AsyncIterator[tuple[str, float]] = agent.stream(
            PROMPT, tenant_id=tenant
        )
        return [event async for event in stream]

    initial = asyncio.run(collect("empty"))
    agent.store.add(PROMPT, "".join(STREAM_TOKENS), tenant_id="owner")
    after = asyncio.run(collect("empty"))
    owner = asyncio.run(collect("owner"))
    assert [token for token, _score in initial] == list(STREAM_TOKENS)
    assert initial[0][1] == 1.0
    assert initial[-1][1] < owner[-1][1]
    assert after == initial
    assert len(observed) == 3
    assert all(body["stream"] is True for body in observed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "wire_tokens", "scored_text", "requests"),
    [
        (FRAGMENT_PROMPT, FRAGMENT_TOKENS, "".join(FRAGMENT_TOKENS), 1),
        (WHOLE_PROMPT, tuple(WHOLE_TEXT.split()), WHOLE_TEXT, 1),
        (RETRY_PROMPT, tuple(WHOLE_TEXT.split()), WHOLE_TEXT, 2),
    ],
)
async def test_http_agent_scores_source_text_with_unchanged_stream_tokens(
    upstream: tuple[str, list[dict[str, object]]],
    prompt: str,
    wire_tokens: tuple[str, ...],
    scored_text: str,
    requests: int,
) -> None:
    """The native scorer sees the provider text and clients keep their tokens."""
    url, observed = upstream
    store = GroundTruthStore()
    store.add(prompt, scored_text, tenant_id="assay")
    scorer = CoherenceScorer(
        threshold=0.0, use_nli=False, scorer_backend="rust", ground_truth_store=store
    )
    agent = CoherenceAgent(
        llm_api_url=url, production_mode=True, _store=store, _scorer=scorer
    )
    agent.streaming_kernel = StreamingKernel(
        hard_limit=0.0, window_threshold=0.0, trend_threshold=1.0
    )
    events = [event async for event in agent.stream_events(prompt, "assay")]
    assert tuple(event.token for event in events) == wire_tokens
    assert not any(event.halt_reason for event in events)
    assert scorer.history == [scored_text]
    assert len(observed) == requests


@pytest.mark.asyncio
async def test_concurrent_http_agent_streams_keep_independent_halt_windows(
    upstream: tuple[str, list[dict[str, object]]],
) -> None:
    """Interleaving real SSE streams must preserve both isolated decisions."""
    url, observed = upstream
    agent = CoherenceAgent(llm_api_url=url, use_nli=False, production_mode=True)
    agent.store.add(CONTINUATION_PROMPT, "".join(STREAM_TOKENS), tenant_id="owner")
    threshold = 0.77
    agent.streaming_kernel = StreamingKernel(
        window_size=2, window_threshold=threshold, trend_window=10
    )
    owner = [event async for event in agent.stream(CONTINUATION_PROMPT, "owner")]
    other = [event async for event in agent.stream(CONTINUATION_PROMPT, "other")]
    assert [token for token, _ in owner] == list(CONTINUATION_TOKENS)
    assert [token for token, _ in other] == list(CONTINUATION_TOKENS)
    assert (owner[1][1] + other[1][1]) / 2 < threshold

    owner_stream = agent.stream(CONTINUATION_PROMPT, "owner")
    other_stream = agent.stream(CONTINUATION_PROMPT, "other")
    assert await anext(owner_stream) == owner[0]
    assert await anext(other_stream) == other[0]
    assert await anext(owner_stream) == owner[1]
    assert await anext(other_stream) == other[1]
    assert [event async for event in owner_stream] == owner[2:]
    assert [event async for event in other_stream] == other[2:]
    assert len(observed) == 4
    assert all(body["stream"] is True for body in observed)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("halt", "closable"), [(False, True), (True, True), (False, False)]
)
async def test_external_token_generator_and_tokenless_stream_verdict(
    upstream: tuple[str, list[dict[str, object]]], halt: bool, closable: bool
) -> None:
    """Keep real provider lifetime and tokenless decisions consistent across APIs."""
    url, observed = upstream

    class IteratorOnly:
        """Provide an async iterator without a separate closing method."""

        def __init__(self, stream: AsyncIterator[str]) -> None:
            """Remember the public provider iterator."""
            self.stream = stream

        def __aiter__(self) -> Self:
            """Return this iterator for async iteration."""
            return self

        async def __anext__(self) -> str:
            """Read the next actual provider token."""
            return await anext(self.stream)

    class ExternalGenerator:
        """Expose a real HTTP generator through the token-only API."""

        def __init__(self, endpoint: str) -> None:
            """Connect to the endpoint and track its opened streams."""
            self.generator = LLMGenerator(endpoint)
            self.opened: list[AsyncIterator[str]] = []

        def stream_tokens(self, prompt: str) -> AsyncIterator[str]:
            """Retain streams so their finalization can be observed explicitly."""
            stream = self.generator.stream_tokens(prompt)
            self.opened.append(stream)
            return stream if closable else IteratorOnly(stream)

    class DecisionOnlyAgent(CoherenceAgent):
        """Publish decisions while withholding the token that caused a halt."""

        async def stream_events(
            self, prompt: str, tenant_id: str = ""
        ) -> AsyncGenerator[AgentStreamEvent, None]:
            """Retain the real verdict and redact its rejected token."""
            async with aclosing(super().stream_events(prompt, tenant_id)) as events:
                async for event in events:
                    yield replace(event, token=None) if event.halt_reason else event

    agent = DecisionOnlyAgent(llm_api_url=url, use_nli=False, production_mode=True)
    provider = ExternalGenerator(url)
    agent.generator = provider
    agent.streaming_kernel = StreamingKernel(hard_limit=float(halt))
    events = [event async for event in agent.stream_events(PROMPT, "audit-team")]
    legacy = [token async for token, _ in agent.stream(PROMPT, "audit-team")]

    if halt:
        assert [event.token for event in events] == [STREAM_TOKENS[0], None]
        assert events[-1].halt_reason == "coherence_halt"
        assert legacy == [STREAM_TOKENS[0]]
    else:
        assert [event.token for event in events] == list(STREAM_TOKENS)
        assert all(not event.halt_reason for event in events)
        assert legacy == list(STREAM_TOKENS)
    if closable:
        async with aclosing(agent.stream_events(PROMPT, "audit-team")) as stopped:
            assert (await anext(stopped)).token == STREAM_TOKENS[0]
    expected_streams = 3 if closable else 2
    assert len(observed) == expected_streams
    assert all(body["stream"] is True for body in observed)
    assert len(provider.opened) == expected_streams
    for stream in provider.opened:
        assert isinstance(stream, AsyncGeneratorType) and stream.ag_frame is None


@pytest.mark.parametrize("entry", ["sync", "async", "stream"])
def test_http_agent_does_not_retry_a_scoped_retrieval_error_without_tenant(
    upstream: tuple[str, list[dict[str, object]]], entry: str
) -> None:
    """A retrieval failure cannot cause a second review in a broader scope."""
    url, observed = upstream
    tenant = "audit-team"
    scopes: list[str] = []

    class FailingStore(GroundTruthStore):
        """Report the requested scope and refuse retrieval."""

        def retrieve_context(
            self, query: str, top_k: int = 0, tenant_id: str = ""
        ) -> str | None:
            """Raise a controlled adapter error after recording the scope."""
            scopes.append(tenant_id)
            raise TypeError("test retrieval adapter failed")

    store = FailingStore()
    scorer = CoherenceScorer(threshold=0.6, use_nli=False, ground_truth_store=store)
    agent = CoherenceAgent(
        llm_api_url=url,
        max_candidates=1,
        production_mode=True,
        _store=store,
        _scorer=scorer,
    )

    async def collect() -> None:
        """Consume the actual upstream SSE response through the agent."""
        async for _ in agent.stream(PROMPT, tenant_id=tenant):
            pass

    with pytest.raises(TypeError, match="test retrieval adapter failed"):
        if entry == "stream":
            asyncio.run(collect())
        else:
            _review(agent, entry, tenant)
    assert scopes == [tenant]
    assert len(observed) == 1


def test_explicit_model_request_is_not_replaced_with_native_heuristics(
    upstream: tuple[str, list[dict[str, object]]],
) -> None:
    """NLI-enabled construction configures the model path without loading weights."""
    url, observed = upstream
    agent = CoherenceAgent(llm_api_url=url, use_nli=True, production_mode=True)
    assert isinstance(agent.scorer, CoherenceScorer)
    assert agent.scorer.use_nli
    assert agent.scorer.scorer_backend == "deberta"
    assert observed == []
