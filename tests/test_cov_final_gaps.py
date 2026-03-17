"""Final coverage gaps: nli scoring, server, backends, cli."""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from director_ai.core.config import DirectorConfig

_HAS_FASTAPI = __import__("importlib").util.find_spec("fastapi") is not None
_skip_no_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


# ── NLI model_score_batch / onnx_score_batch ────────────────────────


class TestNliModelScoreBatch:
    def test_model_score_batch_standard(self):
        """Cover _model_score_batch for standard (non-FactCG) model."""
        import numpy as np
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "deberta"
        scorer._model_name = "deberta-standard"
        scorer.max_length = 512
        scorer._model = MagicMock()
        scorer._tokenizer = MagicMock()
        scorer._minicheck = None
        scorer._minicheck_loaded = True
        scorer._custom_backend = None
        scorer._last_token_count = 0
        scorer._label_indices = None

        mock_device = MagicMock()
        scorer._model.parameters.return_value = iter([MagicMock(device=mock_device)])

        mock_probs = np.array([[0.1, 0.3, 0.6], [0.05, 0.25, 0.7]])
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.softmax.return_value.cpu.return_value.numpy.return_value = mock_probs
        scorer._model.return_value.logits = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = scorer._model_score_batch([("p1", "h1"), ("p2", "h2")])
            assert len(result) == 2

    def test_model_score_batch_factcg(self):
        """Cover _model_score_batch for FactCG model."""
        import numpy as np
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "deberta"
        scorer._model_name = "UniEval-FactCG"
        scorer.max_length = 512
        scorer._model = MagicMock()
        scorer._tokenizer = MagicMock()
        scorer._last_token_count = 0
        scorer._label_indices = None

        mock_device = MagicMock()
        scorer._model.parameters.return_value = iter([MagicMock(device=mock_device)])

        mock_probs = np.array([[0.1, 0.9]])
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.softmax.return_value.cpu.return_value.numpy.return_value = mock_probs
        scorer._model.return_value.logits = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = scorer._model_score_batch([("p", "h")])
            assert len(result) == 1


class TestNliOnnxScoreBatch:
    def test_onnx_score_batch_standard(self):
        """Cover _onnx_score_batch for standard path."""
        import numpy as np
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "onnx"
        scorer._model_name = "deberta-standard"
        scorer.max_length = 512
        scorer._onnx_session = MagicMock()
        scorer._tokenizer = MagicMock()
        scorer._model = None
        scorer._minicheck = None
        scorer._minicheck_loaded = True
        scorer._custom_backend = None
        scorer._last_token_count = 0
        scorer._label_indices = None

        input_mock = MagicMock()
        input_mock.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_mock]

        ids = np.array([[1, 2, 3]], dtype=np.int64)
        scorer._tokenizer.return_value = {"input_ids": ids}

        probs = np.array([[0.1, 0.3, 0.6]])
        scorer._onnx_session.run.return_value = [probs]

        result = scorer._onnx_score_batch([("premise", "hypothesis")])
        assert len(result) == 1

    def test_onnx_score_batch_factcg(self):
        """Cover _onnx_score_batch for FactCG path."""
        import numpy as np
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "onnx"
        scorer._model_name = "UniEval-FactCG"
        scorer.max_length = 512
        scorer._onnx_session = MagicMock()
        scorer._tokenizer = MagicMock()
        scorer._last_token_count = 0
        scorer._label_indices = None

        input_mock = MagicMock()
        input_mock.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_mock]

        ids = np.array([[1, 2, 3]], dtype=np.int64)
        scorer._tokenizer.return_value = {"input_ids": ids}

        probs = np.array([[0.2, 0.8]])
        scorer._onnx_session.run.return_value = [probs]

        result = scorer._onnx_score_batch([("p", "h")])
        assert len(result) == 1


class TestNliMinicheck:
    def test_minicheck_score_batch(self):
        """Cover _minicheck_score_batch path."""
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "minicheck"
        scorer._minicheck_loaded = True
        scorer._minicheck = MagicMock()
        scorer._model = None
        scorer._tokenizer = None
        scorer._onnx_session = None
        scorer._custom_backend = None
        scorer._label_indices = None

        scorer._minicheck.score.return_value = [0.9]
        result = scorer._minicheck_score_batch([("premise", "hypothesis")])
        assert len(result) == 1


class TestNliDecomposeClaims:
    def test_decompose_claims_empty(self):
        """Cover line 751 — empty hypothesis."""
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer._spacy_nlp = None
        claims = scorer.decompose_claims("")
        assert claims == []


# ── Backends: RustBackend ───────────────────────────────────────────


class TestRustBackend:
    def test_rust_backend_mock(self):
        mock_bk = MagicMock()
        mock_config = MagicMock()
        mock_bk.BackfireConfig.return_value = mock_config
        mock_scorer = MagicMock()
        mock_bk.RustCoherenceScorer.return_value = mock_scorer

        mock_score_obj = MagicMock()
        mock_score_obj.score = 0.85
        mock_scorer.review.return_value = (True, mock_score_obj)

        with patch.dict(sys.modules, {"backfire_kernel": mock_bk}):
            from director_ai.core.backends import RustBackend

            be = RustBackend(threshold=0.6)
            result = be.score("premise", "hypothesis")
            assert result == 0.85

            results = be.score_batch([("p1", "h1"), ("p2", "h2")])
            assert len(results) == 2


# ── SDK Guard: async proxy paths ───────────────────────────────────


class TestSdkGuardAsyncOpenAI:
    @pytest.mark.asyncio
    async def test_async_openai_proxy(self):
        from director_ai.core.scorer import CoherenceScorer

        from director_ai.integrations.sdk_guard import _OpenAICompletionsProxy

        scorer = CoherenceScorer(use_nli=False)
        original = MagicMock()
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        )
        original.create = AsyncMock(return_value=response)

        proxy = _OpenAICompletionsProxy(original, scorer, "log")
        result = await proxy.create(messages=[{"role": "user", "content": "q"}])
        assert result is response

    @pytest.mark.asyncio
    async def test_async_openai_streaming(self):
        from director_ai.core.scorer import CoherenceScorer

        from director_ai.integrations.sdk_guard import _OpenAICompletionsProxy

        scorer = CoherenceScorer(use_nli=False)
        original = MagicMock()

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))],
        )

        async def _fake_stream(**kw):
            return [chunk]

        original.create = AsyncMock(return_value=[chunk])

        proxy = _OpenAICompletionsProxy(original, scorer, "log")
        result = await proxy.create(
            messages=[{"role": "user", "content": "q"}],
            stream=True,
        )
        chunks = list(result)
        assert len(chunks) == 1


class TestSdkGuardAsyncAnthropic:
    @pytest.mark.asyncio
    async def test_async_anthropic_proxy(self):
        from director_ai.core.scorer import CoherenceScorer

        from director_ai.integrations.sdk_guard import _AnthropicMessagesProxy

        scorer = CoherenceScorer(use_nli=False)
        original = MagicMock()
        response = SimpleNamespace(content=[SimpleNamespace(text="answer")])
        original.create = AsyncMock(return_value=response)

        proxy = _AnthropicMessagesProxy(original, scorer, "log")
        result = await proxy.create(messages=[{"role": "user", "content": "q"}])
        assert result is response

    @pytest.mark.asyncio
    async def test_async_anthropic_streaming(self):
        from director_ai.core.scorer import CoherenceScorer

        from director_ai.integrations.sdk_guard import _AnthropicMessagesProxy

        scorer = CoherenceScorer(use_nli=False)
        original = MagicMock()

        event = SimpleNamespace(text="hello")
        original.create = AsyncMock(return_value=[event])

        proxy = _AnthropicMessagesProxy(original, scorer, "log")
        result = await proxy.create(
            messages=[{"role": "user", "content": "q"}],
            stream=True,
        )
        events = list(result)
        assert len(events) == 1


# ── Server: rate limit path, WS streaming, delete session ──────────


@_skip_no_server
class TestServerRateLimit:
    def test_rate_limit_with_slowapi(self):
        """Cover rate_limit_rpm > 0 with slowapi available."""
        cfg = DirectorConfig(use_nli=False, rate_limit_rpm=100)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        assert app is not None


@_skip_no_server
class TestServerDeleteSession:
    def test_delete_session_missing(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.delete("/v1/sessions/nonexistent")
            assert resp.status_code == 404


@_skip_no_server
class TestServerWsStreaming:
    def test_ws_streaming_oversight(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c, c.websocket_connect("/v1/stream") as ws:
            ws.send_json(
                {
                    "prompt": "What is 2+2?",
                    "streaming_oversight": True,
                },
            )
            resp = ws.receive_json()
            assert resp.get("type") in ("token", "result", "halt", "error")


@_skip_no_server
class TestServerProcessEndpoint:
    def test_process_endpoint(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.post("/v1/process", json={"prompt": "hello"})
            assert resp.status_code == 200
            data = resp.json()
            assert "output" in data


@_skip_no_server
class TestServerBatchEndpoint:
    def test_batch_endpoint(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.post("/v1/batch", json={"prompts": ["hello", "world"]})
            assert resp.status_code == 200


@_skip_no_server
class TestServerStatsEndpoint:
    def test_stats_with_sqlite(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False, stats_backend="sqlite")
        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.get("/v1/stats")
            assert resp.status_code == 200


@_skip_no_server
class TestServerAuditLogging:
    def test_review_with_audit_logging(self, tmp_path):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        audit_path = str(tmp_path / "audit.db")
        cfg = DirectorConfig(use_nli=False, audit_log_path=audit_path)
        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.post(
                "/v1/review",
                json={"prompt": "sky?", "response": "The sky is blue."},
            )
            assert resp.status_code == 200


# ── CLI: remaining edge cases ───────────────────────────────────────


@_skip_no_server
class TestCliServeHost:
    def test_serve_with_host(self):
        """Cover --host parsing in serve command (line 702-703)."""
        with patch.dict(sys.modules, {"uvicorn": MagicMock()}):
            from director_ai.cli import main

            main(["serve", "--host", "127.0.0.1", "--port", "9999"])


class TestCliStressTest:
    def test_stress_test_basic(self, capsys):
        from director_ai.cli import main

        main(["stress-test", "--iterations", "2", "--concurrency", "1"])
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_stress_test_json(self, capsys):
        from director_ai.cli import main

        main(["stress-test", "--iterations", "2", "--json"])
        out = capsys.readouterr().out
        assert len(out) > 0


class TestCliBenchEdges:
    def test_bench_no_args(self, capsys):
        pytest.importorskip("benchmarks", reason="benchmarks not on sys.path")
        from director_ai.cli import main

        with contextlib.suppress(SystemExit):
            main(["bench"])

    def test_bench_with_seed(self, capsys):
        pytest.importorskip("benchmarks", reason="benchmarks not on sys.path")
        from director_ai.cli import main

        with contextlib.suppress(SystemExit):
            main(["bench", "--seed", "42"])


class TestCliIngestJsonDecodeDetail:
    def test_ingest_jsonl_with_text_key(self, tmp_path, capsys):
        f = tmp_path / "docs.jsonl"
        f.write_text('{"text": "Fact one."}\n{"text": "Fact two."}\n', encoding="utf-8")
        from director_ai.cli import main

        main(["ingest", str(f)])


# ── LangChain callback: raise_on_failure path ─────────────────────


class TestLangchainCallbackRaise:
    def test_on_llm_end_raise_on_failure(self):
        from director_ai.core.exceptions import CoherenceError
        from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

        handler = CoherenceCallbackHandler(
            use_nli=False,
            threshold=1.0,
            raise_on_failure=True,
        )
        handler._current_prompt = "What is 2+2?"
        response = SimpleNamespace(
            generations=[[SimpleNamespace(text="The answer is 4.")]],
        )
        with pytest.raises(CoherenceError):
            handler.on_llm_end(response)


# ── Providers: remaining edges ──────────────────────────────────────


class TestProvidersEdgesDeep:
    def test_openai_provider_generate_candidates(self):
        from director_ai.integrations.providers import OpenAIProvider

        with patch("director_ai.integrations.providers.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "hello"}}],
            }
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            provider = OpenAIProvider(api_key="fake-key")
            result = provider.generate_candidates("test", n=1)
            assert len(result) == 1
            assert result[0]["text"] == "hello"
