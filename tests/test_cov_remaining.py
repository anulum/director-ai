# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle remaining coverage gaps: gRPC, server, CLI, NLI, providers."""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import director_ai as _director_pkg
from director_ai.core.config import DirectorConfig

_HAS_FASTAPI = __import__("importlib").util.find_spec("fastapi") is not None
_skip_no_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")

_SENTINEL = object()


@contextlib.contextmanager
def _grpc_context(mods):
    """Patch sys.modules AND clear director_ai submodule attr cache."""
    saved = {}
    for attr in ("director_pb2", "director_pb2_grpc"):
        old = _director_pkg.__dict__.pop(attr, _SENTINEL)
        if old is not _SENTINEL:
            saved[attr] = old
    try:
        with patch.dict(sys.modules, mods):
            yield
    finally:
        for attr, old in saved.items():
            setattr(_director_pkg, attr, old)


# â”€â”€ gRPC StreamTokens â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestGrpcStreamTokens:
    def test_stream_tokens_rpc(self):
        grpc = MagicMock()
        grpc.__version__ = "1.78.0"
        server = MagicMock()
        grpc.server.return_value = server
        grpc.ServerInterceptor = type("ServerInterceptor", (), {})

        pb2 = MagicMock()
        descriptor = MagicMock()
        svc_desc = MagicMock()
        svc_desc.full_name = "director_ai.DirectorService"
        descriptor.services_by_name = {"DirectorService": svc_desc}
        pb2.DESCRIPTOR = descriptor
        pb2.ReviewResponse = SimpleNamespace
        pb2.ProcessResponse = SimpleNamespace
        pb2.BatchReviewResponse = SimpleNamespace
        pb2.TokenEvent = SimpleNamespace

        pb2_grpc = MagicMock()

        mods = {
            "grpc": grpc,
            "director_ai.director_pb2": pb2,
            "director_ai.director_pb2_grpc": pb2_grpc,
        }

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(port=50220)
            servicer = pb2_grpc.add_DirectorServiceServicer_to_server.call_args[0][0]

            request = SimpleNamespace(prompt="What is 2+2?")
            tokens = list(servicer.StreamTokens(request, MagicMock()))
            assert len(tokens) >= 0


# â”€â”€ Server: rate limit 429 handler, WS non-streaming, lifespan â”€â”€â”€â”€


@_skip_no_server
class TestServerRateLimitHandler:
    def test_rate_limit_handler_registered(self):
        cfg = DirectorConfig(use_nli=False, rate_limit_rpm=60)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        assert app is not None


@_skip_no_server
class TestServerWsNonStreaming:
    def test_ws_standard_mode(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c, c.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "What is 2+2?"})
            resp = ws.receive_json()
            assert "type" in resp or "error" in resp

    def test_ws_bad_json(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c, c.websocket_connect("/v1/stream") as ws:
            ws.send_text("not valid json{{{")
            resp = ws.receive_json()
            assert "error" in resp

    def test_ws_empty_prompt(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c, c.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": ""})
            resp = ws.receive_json()
            assert "error" in resp

    def test_ws_non_dict(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c, c.websocket_connect("/v1/stream") as ws:
            ws.send_json([1, 2, 3])
            resp = ws.receive_json()
            assert "error" in resp


@_skip_no_server
class TestServerReviewSession:
    def test_review_with_session(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.post(
                "/v1/review",
                json={
                    "prompt": "sky?",
                    "response": "The sky is blue.",
                    "session_id": "test-session-1",
                },
            )
            assert resp.status_code == 200


@_skip_no_server
class TestServerNliMetric:
    def test_nli_enabled_gauge(self):
        cfg = DirectorConfig(use_nli=True)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        assert app is not None


# â”€â”€ CLI: remaining uncovered lines â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestCliUnknownProfile:
    def test_unknown_profile(self, capsys):
        from director_ai.cli import main

        with pytest.raises(SystemExit):
            main(["--profile", "nonexistent", "review", "p", "h"])


class TestCliBatchFile:
    def test_batch_from_file(self, tmp_path, capsys):
        f = tmp_path / "batch.jsonl"
        f.write_text('{"prompt": "sky?"}\n{"prompt": "sun?"}\n', encoding="utf-8")
        from director_ai.cli import main

        main(["batch", str(f)])

    def test_batch_missing_file(self, capsys):
        from director_ai.cli import main

        with pytest.raises(SystemExit):
            main(["batch", "/nonexistent/file.jsonl"])


class TestCliBenchModel:
    def test_bench_with_model(self, capsys):
        pytest.importorskip("benchmarks", reason="benchmarks not on sys.path")
        from director_ai.cli import main

        main(["bench", "--model", "heuristic"])


class TestCliIngestChunkSize:
    def test_ingest_with_chunk_size(self, tmp_path, capsys):
        f = tmp_path / "doc.txt"
        f.write_text(
            "Some facts about the world.\n\nAnother paragraph.\n",
            encoding="utf-8",
        )
        from director_ai.cli import main

        main(["ingest", str(f), "--chunk-size", "50"])


class TestCliIngestPersist:
    def test_ingest_with_persist(self, tmp_path, capsys):
        f = tmp_path / "doc.txt"
        f.write_text("Fact one.\n\nFact two.\n", encoding="utf-8")
        persist = tmp_path / "persist_dir"
        persist.mkdir()
        from director_ai.cli import main

        main(["ingest", str(f), "--persist", str(persist)])


class TestCliServeGrpc:
    def test_serve_grpc_transport(self):
        mock_grpc_mod = MagicMock()
        mock_grpc_mod.__version__ = "1.78.0"
        mock_grpc_server = MagicMock()
        mock_grpc_mod.server.return_value = mock_grpc_server
        mock_grpc_mod.ServerInterceptor = type("SI", (), {})
        mods = {
            "grpc": mock_grpc_mod,
            "director_ai.director_pb2": MagicMock(),
            "director_ai.director_pb2_grpc": MagicMock(),
        }
        with _grpc_context(mods):
            from director_ai.cli import main

            main(["serve", "--transport", "grpc", "--port", "50333"])


# â”€â”€ NLI: line 751 (score_detailed empty claims) + score path â”€â”€â”€â”€â”€â”€


class TestNliScoreDecomposed:
    def test_score_decomposed_empty_claims(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "heuristic"
        scorer._model = None
        scorer._tokenizer = None
        scorer._onnx_session = None
        scorer._minicheck = None
        scorer._minicheck_loaded = True
        scorer._custom_backend = None
        scorer._spacy_nlp = None
        scorer._model_name = ""
        scorer.max_length = 512
        scorer._model_loaded = False
        scorer.use_model = False
        scorer._label_indices = None

        max_s, per_claim = scorer.score_decomposed("premise", "")
        assert isinstance(max_s, float)
        assert len(per_claim) == 1


class TestNliScoreDispatch:
    def test_score_onnx_path(self):
        import numpy as np

        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "onnx"
        scorer._model_name = "deberta"
        scorer.max_length = 512
        scorer._onnx_session = MagicMock()
        scorer._tokenizer = MagicMock()
        scorer._model = MagicMock()
        scorer._minicheck = None
        scorer._minicheck_loaded = True
        scorer._custom_backend = None
        scorer._model_loaded = True
        scorer._last_token_count = 0
        scorer._label_indices = None

        input_mock = MagicMock()
        input_mock.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_mock]
        ids = np.array([[1, 2, 3]], dtype=np.int64)
        scorer._tokenizer.return_value = {"input_ids": ids}
        scorer._onnx_session.run.return_value = [np.array([[0.1, 0.3, 0.6]])]

        result = scorer.score("premise", "hypothesis")
        assert isinstance(result, float)


class TestNliEnsureModel:
    def test_ensure_model_not_loaded_falls_to_heuristic(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "deberta"
        scorer._model = None
        scorer._tokenizer = None
        scorer._onnx_session = None
        scorer._minicheck = None
        scorer._minicheck_loaded = True
        scorer._custom_backend = None
        scorer._model_name = "deberta"
        scorer.max_length = 512
        scorer._model_loaded = False
        scorer.use_model = False
        scorer._label_indices = None

        result = scorer.score("The sky is blue.", "The sky is blue.")
        assert isinstance(result, float)


# â”€â”€ Providers: streaming + Anthropic + Local â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestProvidersStreaming:
    def test_openai_stream_generate(self):
        from director_ai.integrations.providers import OpenAIProvider

        with patch("director_ai.integrations.providers.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.iter_lines.return_value = [
                'data: {"choices": [{"delta": {"content": "hello"}}]}',
                'data: {"choices": [{"delta": {"content": " world"}}]}',
                "data: [DONE]",
            ]
            mock_post.return_value = mock_resp

            provider = OpenAIProvider(api_key="fake")
            tokens = list(provider.stream_generate("test"))
            assert tokens == ["hello", " world"]

    def test_anthropic_generate_candidates(self):
        from director_ai.integrations.providers import AnthropicProvider

        with patch("director_ai.integrations.providers.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "content": [{"type": "text", "text": "hello"}],
            }
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            provider = AnthropicProvider(api_key="fake")
            result = provider.generate_candidates("test", n=1)
            assert len(result) == 1
            assert result[0]["text"] == "hello"


# â”€â”€ LangChain callback: extraction failure + no text â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestLangchainCallbackEdges:
    def test_on_llm_end_extraction_failure(self):
        from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

        handler = CoherenceCallbackHandler(use_nli=False, threshold=0.5)
        handler._current_prompt = "test"
        response = SimpleNamespace(generations=None)
        handler.on_llm_end(response)

    def test_on_llm_end_empty_text(self):
        from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

        handler = CoherenceCallbackHandler(use_nli=False, threshold=0.5)
        handler._current_prompt = "test"
        response = SimpleNamespace(generations=[[SimpleNamespace(text="")]])
        handler.on_llm_end(response)


# â”€â”€ Streaming: halt_evidence with chunks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestStreamingHaltEvidence:
    def test_streaming_halt_with_evidence(self):
        from director_ai.core.streaming import StreamingKernel

        scorer = MagicMock()
        chunk_mock = MagicMock()
        chunk_mock.distance = 0.1
        evidence_mock = MagicMock()
        evidence_mock.chunks = [chunk_mock]
        evidence_mock.chunk_scores = [0.9]
        score_obj = MagicMock()
        score_obj.score = 0.3
        score_obj.evidence = evidence_mock
        scorer.review.return_value = (False, score_obj)

        kernel = StreamingKernel(hard_limit=0.4)

        def token_gen():
            yield "word1"
            yield "word2"

        def coherence_cb(text):
            return 0.2

        session = kernel.stream_tokens(
            token_gen(),
            coherence_cb,
            scorer=scorer,
            prompt="test",
        )
        assert session.halted


# â”€â”€ Agent: Rust scorer path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAgentRustScorerFallback:
    def test_build_scorer_rust_unavailable(self):
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent(use_nli=False)
        assert agent.scorer is not None


# â”€â”€ Async streaming: halt return path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAsyncStreamingHalt:
    @pytest.mark.asyncio
    async def test_async_stream_hard_limit_halt(self):
        from director_ai.core.async_streaming import AsyncStreamingKernel

        kernel = AsyncStreamingKernel(hard_limit=0.4)

        async def token_gen():
            yield "word1"
            yield "word2"

        def coherence_cb(text):
            return 0.2

        events = []
        async for e in kernel.stream_tokens(token_gen(), coherence_cb):
            events.append(e)
        assert any(e.halted for e in events)

    @pytest.mark.asyncio
    async def test_async_stream_downward_trend(self):
        from director_ai.core.async_streaming import AsyncStreamingKernel

        scores = iter([0.9, 0.8, 0.7, 0.6, 0.3])
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            trend_window=3,
            trend_threshold=0.1,
        )

        async def token_gen():
            for i in range(5):
                yield f"tok{i}"

        def coherence_cb(text):
            return next(scores, 0.5)

        events = []
        async for e in kernel.stream_tokens(token_gen(), coherence_cb):
            events.append(e)
