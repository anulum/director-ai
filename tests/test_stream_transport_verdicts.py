# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — streaming verdicts through the server transports

"""Exercise pre-egress halt verdicts with a real HTTP generation source."""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from math import log
from pathlib import Path
from threading import Event, Thread
from threading import enumerate as running_threads
from typing import Any

import pytest
from starlette.testclient import TestClient

from director_ai.core.agent import AgentStreamEvent, CoherenceAgent
from director_ai.core.config import DirectorConfig
from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.scoring.scorer import CoherenceScorer
from director_ai.grpc_server import create_grpc_server
from director_ai.integrations.providers import OpenAIProvider
from director_ai.server import create_app

PROMPT = "reference continuation"
TOKENS = ("The approved ", "reference is alpha.", " More")


@pytest.fixture
def completion_url() -> Iterator[str]:
    """Run the completion protocol consumed by the configured local provider."""

    class Completion(BaseHTTPRequestHandler):
        """Stream known token boundaries over a real loopback connection."""

        def do_POST(self) -> None:
            """Record a valid request and send its incremental response."""
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/completion/unavailable":
                assert request["prompt"] == PROMPT
                self.send_error(503)
                return
            if self.path == "/chat/completions":
                assert request["messages"][-1]["content"] == PROMPT
                body = json.dumps(
                    {"choices": [{"message": {"content": "".join(TOKENS)}}]}
                ).encode()
                content_type = "application/json"
            else:
                assert request["prompt"] == PROMPT
                assert request["stream"] is True
                tokens = (
                    TOKENS[:1] if self.path == "/completion/interrupted" else TOKENS
                )
                body = "".join(
                    "data: " + json.dumps({"content": token}) + "\n\n"
                    for token in tokens
                ).encode()
                content_type = "text/event-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            length = len(body)
            if self.path == "/completion/interrupted":
                length += len(b"data: [DONE]\n\n")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Completion)
    worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/completion"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)
        assert not worker.is_alive()


@dataclass(frozen=True)
class _Verdict:
    """Client-visible tokens and the server's terminal decision."""

    tokens: tuple[str, ...]
    halted: bool
    reason: str


def _http_stream(config: DirectorConfig, transport: str) -> _Verdict:
    """Consume the published incremental HTTP or WebSocket protocol."""
    request = {"prompt": PROMPT, "streaming_oversight": True}
    frames: list[dict[str, Any]] = []
    with TestClient(create_app(config=config)) as client:
        if transport == "sse":
            response = client.post("/v1/stream/sse", json=request)
            assert response.status_code == 200
            for block in response.text.strip().split("\n\n"):
                event, data = block.splitlines()
                frames.append(
                    {
                        "type": event.removeprefix("event: "),
                        **json.loads(data.removeprefix("data: ")),
                    }
                )
        else:
            with client.websocket_connect("/v1/stream") as connection:
                connection.send_json({**request, "session_id": "verdict"})
                for _ in range(len(TOKENS) + 1):
                    frame = connection.receive_json()
                    frames.append(frame)
                    if frame.get("type") != "token":
                        break
    tokens = tuple(frame["token"] for frame in frames[:-1] if frame["type"] == "token")
    terminal = frames[-1]
    assert len(tokens) == len(frames) - 1 == terminal["tokens_delivered"]
    assert terminal["type"] == ("halt" if terminal["halted"] else "complete")
    return _Verdict(tokens, terminal["halted"], terminal.get("reason", ""))


@contextmanager
def _grpc_stub(
    config: DirectorConfig, max_workers: int = 4
) -> Iterator[tuple[Any, Any]]:
    """Connect the generated stub to the configured server over loopback."""
    grpc = import_module("grpc")
    stubs = import_module("director_ai.director_pb2_grpc")
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = int(port_socket.getsockname()[1])
    server = create_grpc_server(config=config, port=port, max_workers=max_workers)
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            yield server, stubs.DirectorServiceStub(channel)
    finally:
        assert server.stop(0).wait(timeout=5.0)


def _grpc_stream(config: DirectorConfig) -> _Verdict:
    """Consume the generated protobuf events over a real gRPC channel."""
    pytest.importorskip("grpc")
    messages = pytest.importorskip("director_ai.director_pb2")
    pytest.importorskip("director_ai.director_pb2_grpc")
    with _grpc_stub(config) as (_, stub):
        frames = list(
            stub.StreamTokens(messages.StreamRequest(prompt=PROMPT), timeout=15.0)
        )
    assert [frame.index for frame in frames] == list(range(len(frames)))
    terminal = frames[-1]
    assert all(not frame.halted for frame in frames[:-1])
    if terminal.halted:
        assert terminal.token == ""
    return _Verdict(
        tuple(frame.token for frame in frames if not frame.halted),
        terminal.halted,
        terminal.halt_reason,
    )


def _grpc_workers() -> set[Thread]:
    """Identify the threads that bridge streaming to the gRPC server."""
    return {t for t in running_threads() if t.name.startswith("grpc-async")}


@pytest.mark.parametrize("prompt", ["", " \t\n"])
def test_grpc_stream_rejects_blank_input(completion_url: str, prompt: str) -> None:
    """Return a caller error for malformed requests and keep valid streams usable."""
    grpc = import_module("grpc")
    messages = import_module("director_ai.director_pb2")
    config = DirectorConfig(
        mode="general",
        llm_provider="local",
        llm_api_url=completion_url,
        use_nli=False,
        hard_limit=0.0,
        rate_limit_rpm=0,
    )
    with _grpc_stub(config) as (_, stub):
        stream = stub.StreamTokens(messages.StreamRequest(prompt=prompt), timeout=5.0)
        with pytest.raises(grpc.RpcError) as refused:
            next(stream)
        assert refused.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert refused.value.details() == "prompt must be a non-empty string"
        frames = stub.StreamTokens(messages.StreamRequest(prompt=PROMPT), timeout=5.0)
        assert tuple(frame.token for frame in frames) == TOKENS


@pytest.mark.parametrize(
    ("failure", "prefix"), [("unavailable", ()), ("interrupted", TOKENS[:1])]
)
def test_grpc_stream_reports_upstream_failure_safely(
    completion_url: str, failure: str, prefix: tuple[str, ...]
) -> None:
    """Keep delivered tokens and return one safe error without a halt verdict."""
    grpc = import_module("grpc")
    messages = import_module("director_ai.director_pb2")
    config = DirectorConfig(
        mode="general",
        llm_provider="local",
        llm_api_url=f"{completion_url}/{failure}",
        use_nli=False,
        hard_limit=0.0,
        rate_limit_rpm=0,
    )
    with _grpc_stub(config) as (_, stub):
        stream = stub.StreamTokens(messages.StreamRequest(prompt=PROMPT), timeout=8.0)
        delivered = []
        with pytest.raises(grpc.RpcError) as failed:
            for frame in stream:
                assert not frame.halted
                delivered.append(frame.token)
        assert tuple(delivered) == prefix
        assert failed.value.code() == grpc.StatusCode.INTERNAL
        assert failed.value.details() == "streaming failed"


@pytest.mark.parametrize("streaming", [False, True])
def test_grpc_stream_threads_follow_server_lifetime(
    completion_url: str, streaming: bool
) -> None:
    """Leave no streaming threads after repeated public server lifecycles."""
    config = DirectorConfig(
        mode="general",
        llm_provider="local",
        llm_api_url=completion_url,
        use_nli=False,
        hard_limit=0.0,
        rate_limit_rpm=0,
    )
    messages = import_module("director_ai.director_pb2")

    before = _grpc_workers()
    for _ in range(2):
        with _grpc_stub(config) as (_, stub):
            if streaming:
                frames = list(
                    stub.StreamTokens(messages.StreamRequest(prompt=PROMPT), timeout=10)
                )
                assert tuple(frame.token for frame in frames) == TOKENS
                [bridge] = _grpc_workers() - before
                frames = list(
                    stub.StreamTokens(messages.StreamRequest(prompt=PROMPT), timeout=10)
                )
                assert tuple(frame.token for frame in frames) == TOKENS
                assert _grpc_workers() - before == {bridge}
            else:
                result = stub.Review(
                    messages.ReviewRequest(prompt=PROMPT, response="alpha."),
                    timeout=5,
                )
                assert 0.0 <= result.coherence <= 1.0
            spawned = _grpc_workers() - before
            assert bool(spawned) is streaming
        for thread in spawned:
            thread.join(timeout=2)
        assert not spawned.intersection(_grpc_workers())


@pytest.mark.parametrize("deliver_prefix", [False, True])
@pytest.mark.parametrize("stop_server", [False, True])
def test_grpc_termination_releases_a_held_stream(
    deliver_prefix: bool,
    stop_server: bool,
) -> None:
    """Client cancellation or server stop releases a held generation."""
    entered, release = Event(), Event()
    prefix = b'data: {"content":"The approved "}\n\n'
    end = b"data: [DONE]\n\n"

    class HeldCompletion(BaseHTTPRequestHandler):
        """Wait at the selected response boundary on the real provider socket."""

        def do_POST(self) -> None:
            """Hold generation before headers or immediately after one token."""
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["prompt"] == PROMPT and request["stream"] is True
            if not deliver_prefix:
                entered.set()
                release.wait(timeout=10.0)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(prefix) + len(end)))
                self.end_headers()
                self.wfile.write(prefix)
                self.wfile.flush()
                if deliver_prefix:
                    entered.set()
                    release.wait(timeout=10.0)
                self.wfile.write(end)
            except (BrokenPipeError, ConnectionResetError):
                # Deliberate termination can close the upstream connection.
                if not release.is_set():
                    raise

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), HeldCompletion)
    worker = Thread(target=upstream.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        grpc = import_module("grpc")
        messages = import_module("director_ai.director_pb2")
        config = DirectorConfig(
            mode="general",
            use_nli=False,
            llm_provider="local",
            llm_api_url=f"http://127.0.0.1:{upstream.server_port}/completion",
            rate_limit_rpm=0,
            hard_limit=0.0,
        )
        before = _grpc_workers()
        with _grpc_stub(config, max_workers=1) as (server, stub):
            try:
                stream = stub.StreamTokens(
                    messages.StreamRequest(prompt=PROMPT), timeout=12.0
                )
                assert entered.wait(timeout=5.0)
                if deliver_prefix:
                    assert next(stream).token == "The approved "
                if stop_server:
                    spawned = _grpc_workers() - before
                    assert spawned
                    assert server.stop(0).wait(timeout=2.0)
                    with pytest.raises(grpc.RpcError) as aborted:
                        next(stream)
                    assert aborted.value.code() == grpc.StatusCode.UNAVAILABLE
                    for thread in spawned:
                        thread.join(timeout=2)
                    assert not spawned.intersection(_grpc_workers())
                else:
                    assert stream.cancel()
                    review = stub.Review(
                        messages.ReviewRequest(
                            prompt=PROMPT, response="alpha reference."
                        ),
                        timeout=2.0,
                    )
                    assert 0.0 <= review.coherence <= 1.0
                assert not release.is_set()
            finally:
                release.set()
    finally:
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=2.0)
        assert not worker.is_alive()


@pytest.mark.parametrize("transport", ["sse", "websocket", "grpc"])
@pytest.mark.parametrize("halt", [False, True])
def test_streaming_transport_preserves_the_pre_egress_verdict(
    completion_url: str, transport: str, halt: bool
) -> None:
    """Deliver ordinary tokens and suppress the trigger on all three routes."""
    config = DirectorConfig(
        mode="general",
        use_nli=False,
        llm_provider="local",
        llm_api_url=completion_url,
        hard_limit=0.8 if halt else 0.0,
        soft_limit=0.8,
        rate_limit_rpm=0,
    )
    if transport == "grpc":
        result = _grpc_stream(config)
        reason = "hard_limit"
    else:
        result = _http_stream(config, transport)
        reason = "coherence_halt"
    assert result == (
        _Verdict(TOKENS[:1], True, reason) if halt else _Verdict(TOKENS, False, "")
    )


@pytest.mark.parametrize("threshold", [0.60, 0.50])
@pytest.mark.parametrize("coherence_requires_nli", [True, False])
def test_sse_contradiction_threshold_governs_egress(
    completion_url: str, tmp_path: Path, threshold: float, coherence_requires_nli: bool
) -> None:
    """Use the required NLI model at startup and the contradiction cut at egress."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocabulary = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "reference")
    transformers.BertTokenizer(
        vocab={word: index for index, word in enumerate(vocabulary)}
    ).save_pretrained(tmp_path)
    model_config = transformers.BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    )
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = transformers.BertForSequenceClassification(model_config)
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([log(0.225), log(0.225), log(0.55)]))
    model.save_pretrained(tmp_path)
    api_key = "local-test"
    config = DirectorConfig(
        mode="grounded",
        use_nli=True,
        coherence_require_model_backed_nli=coherence_requires_nli,
        injection_detection_enabled=not coherence_requires_nli,
        injection_require_model_backed_nli=not coherence_requires_nli,
        nli_model=str(tmp_path),
        nli_model_revision="",
        reranker_enabled=False,
        llm_provider="local",
        llm_api_url=completion_url,
        streaming_contradiction_halt=True,
        streaming_contradiction_model=str(tmp_path),
        streaming_contradiction_threshold=threshold,
        hard_limit=0.50,
        rate_limit_rpm=0,
        api_keys=[api_key],
        api_key_tenant_map=json.dumps({api_key: "reference-test"}),
    )
    with TestClient(
        create_app(config=config), headers={"X-API-Key": api_key}
    ) as client:
        fact = client.post(
            "/v1/knowledge/ingest", json={"text": "The approved reference is beta."}
        )
        assert fact.status_code == 201
        response = client.post(
            "/v1/stream/sse", json={"prompt": PROMPT, "streaming_oversight": True}
        )
    assert response.status_code == 200
    frames = [
        (event.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
        for event, data in (
            block.splitlines() for block in response.text.strip().split("\n\n")
        )
    ]
    terminal_event, terminal = frames[-1]
    assert terminal["coherence"] == pytest.approx(0.45)
    tokens = tuple(frame["token"] for event, frame in frames if event == "token")
    # With contradiction halt enabled, the configured contradiction threshold
    # decides these events. See docs-site/guide/streaming.md.
    if threshold > 0.55:
        assert (terminal_event, terminal["halted"], tokens) == (
            "complete",
            False,
            TOKENS,
        )
    else:
        assert (terminal_event, terminal["halted"], tokens) == (
            "halt",
            True,
            TOKENS[:1],
        )
        assert terminal["reason"] == "contradiction"


@pytest.mark.asyncio
@pytest.mark.parametrize("halt", [False, True])
async def test_agent_reports_whole_response_only_provider_verdict(
    completion_url: str, halt: bool
) -> None:
    """A legacy tuple stream retains its output while events report rejection."""
    store = GroundTruthStore()
    if not halt:
        store.add(PROMPT, "".join(TOKENS), tenant_id="demo")
    scorer = CoherenceScorer(
        threshold=1.0 if halt else 0.0,
        use_nli=False,
        scorer_backend="rust",
        ground_truth_store=store,
    )
    agent = CoherenceAgent(
        provider="openai",
        api_key="local-test",
        max_candidates=1,
        production_mode=True,
        _store=store,
        _scorer=scorer,
    )
    assert isinstance(agent.generator, OpenAIProvider)
    agent.generator.base_url = completion_url.removesuffix("/completion")
    events = [event async for event in agent.stream_events(PROMPT, "demo")]
    legacy = [token async for token, _score in agent.stream(PROMPT, "demo")]
    if halt:
        assert len(events) == 1
        assert events[0].token is None
        assert events[0].halt_reason == "all_candidates_rejected"
        assert legacy == ["[HALT]:", "All", "candidates", "rejected."]
    else:
        expected = "".join(TOKENS).split()
        assert [event.token for event in events] == legacy
        assert legacy[-len(expected) :] == expected
        assert all(not event.halt_reason for event in events)


@pytest.mark.asyncio
async def test_whole_response_monitor_verdict_survives_unary_and_stream_calls(
    completion_url: str,
) -> None:
    """A permissive scorer cannot turn a monitor refusal into a success event."""
    store = GroundTruthStore()
    scorer = CoherenceScorer(
        threshold=0.0,
        use_nli=False,
        scorer_backend="deberta",
        ground_truth_store=store,
    )
    agent = CoherenceAgent(
        provider="openai",
        api_key="local-test",
        max_candidates=1,
        production_mode=True,
        _store=store,
        _scorer=scorer,
    )
    assert isinstance(agent.generator, OpenAIProvider)
    agent.generator.base_url = completion_url.removesuffix("/completion")

    first = agent.process(PROMPT, "demo")
    assert first.halted and first.output.endswith("[HALT: COHERENCE BELOW THRESHOLD]")
    assert (
        first.halt_evidence is not None and first.halt_evidence.reason == "hard_limit"
    )
    score = first.halt_evidence.last_score
    assert 0.0 < score < agent.kernel.hard_limit
    assert not agent.kernel.is_active

    inactive = await agent.aprocess(PROMPT, "demo")
    assert inactive.halted and inactive.output.endswith("[HALT: KERNEL INACTIVE]")
    assert inactive.halt_evidence is not None
    assert inactive.halt_evidence.reason == "kernel_inactive"

    agent.kernel.reactivate()
    first_events = [event async for event in agent.stream_events(PROMPT, "demo")]
    inactive_events = [event async for event in agent.stream_events(PROMPT, "demo")]
    assert first_events == [AgentStreamEvent(None, score, "hard_limit")]
    assert inactive_events == [AgentStreamEvent(None, score, "kernel_inactive")]
    legacy = [token async for token, _ in agent.stream(PROMPT, "demo")]
    assert legacy[-3:] == ["[HALT:", "KERNEL", "INACTIVE]"]


def test_sse_reports_whole_response_monitor_refusal_without_delivering_marker(
    completion_url: str,
) -> None:
    """Expose the monitor verdict on both SSE shapes with a permissive egress floor."""
    config = DirectorConfig(
        mode="general",
        llm_provider="openai",
        llm_api_key="local-test",
        use_nli=False,
        scorer_backend="rust",
        coherence_threshold=0.0,
        adaptive_threshold_enabled=False,
        hard_limit=0.0,
        max_candidates=1,
        rate_limit_rpm=0,
    )
    app = create_app(config=config)
    with TestClient(app) as client:
        agent = app.state._state["agent"]
        generator = agent.generator
        assert isinstance(generator, OpenAIProvider)
        generator.base_url = completion_url.removesuffix("/completion")
        agent.kernel.hard_limit = 0.8
        result = client.post("/v1/stream/sse", json={"prompt": PROMPT})
        assert result.status_code == 200
        name, data = result.text.strip().splitlines()
        assert name == "event: result"
        payload = json.loads(data.removeprefix("data: "))
        assert payload["halted"] is True
        assert payload["halt_evidence"]["reason"] == "hard_limit"

        stream = client.post(
            "/v1/stream/sse", json={"prompt": PROMPT, "streaming_oversight": True}
        )
        assert stream.status_code == 200
        name, data = stream.text.strip().splitlines()
        assert name == "event: halt"
        assert json.loads(data.removeprefix("data: ")) == {
            "halted": True,
            "reason": "kernel_inactive",
            "tokens_delivered": 0,
            "coherence": round(payload["coherence"], 4),
        }
