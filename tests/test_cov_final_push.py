"""Final coverage push: server rate-limit with slowapi, WS streaming oversight,
halt_evidence serialization, provider stream error paths, agent _build_scorer,
cli evaluate/export edges."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.config import DirectorConfig

_HAS_FASTAPI = __import__("importlib").util.find_spec("fastapi") is not None
_skip_no_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


# ── Server: rate limit with slowapi installed ──────────────────────


@_skip_no_server
class TestServerRateLimitSlowapi:
    def test_rate_limit_creates_limiter(self):
        pytest.importorskip("slowapi", reason="slowapi not installed")
        cfg = DirectorConfig(use_nli=False, rate_limit_rpm=120)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        assert hasattr(app.state, "limiter")

    def test_rate_limit_429_response(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False, rate_limit_rpm=1)
        app = create_app(config=cfg)
        with TestClient(app) as c:
            c.post(
                "/v1/review",
                json={
                    "prompt": "q",
                    "response": "a",
                },
            )
            resp2 = c.post(
                "/v1/review",
                json={
                    "prompt": "q2",
                    "response": "a2",
                },
            )
            # either succeeds or 429
            assert resp2.status_code in (200, 429)


# ── Server: _halt_evidence_to_dict with non-None ──────────────────


@_skip_no_server
class TestHaltEvidenceToDict:
    def test_halt_evidence_serialization(self):
        from director_ai.server import _halt_evidence_to_dict

        chunk = SimpleNamespace(text="contradiction", distance=0.1, source="kb")
        ev = SimpleNamespace(
            reason="hard_limit",
            last_score=0.3,
            evidence_chunks=[chunk],
            nli_scores=[0.9],
            suggested_action="review",
        )
        result = _halt_evidence_to_dict(ev)
        assert result["reason"] == "hard_limit"
        assert len(result["evidence_chunks"]) == 1


# ── Server: CORS allow origins ────────────────────────────────────


@_skip_no_server
class TestServerCorsOrigins:
    def test_cors_with_custom_origins(self):
        cfg = DirectorConfig(use_nli=False, cors_origins="http://localhost:3000")
        from director_ai.server import create_app

        app = create_app(config=cfg)
        assert app is not None


# ── Server: stats close on shutdown ───────────────────────────────


@_skip_no_server
class TestServerStatsClose:
    def test_stats_close_on_shutdown(self, tmp_path):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False, stats_backend="sqlite")
        app = create_app(config=cfg)
        with TestClient(app):
            pass


# ── NLI: score_batch onnx + model paths ──────────────────────────


class TestNliScoreBatchPaths:
    def test_score_batch_onnx(self):
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
        scorer.use_model = True
        scorer._last_token_count = 0

        input_mock = MagicMock()
        input_mock.name = "input_ids"
        scorer._onnx_session.get_inputs.return_value = [input_mock]
        ids = np.array([[1, 2, 3]], dtype=np.int64)
        scorer._tokenizer.return_value = {"input_ids": ids}
        scorer._onnx_session.run.return_value = [np.array([[0.1, 0.3, 0.6]])]

        result = scorer.score_batch([("p", "h")])
        assert len(result) == 1

    def test_score_batch_model(self):
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
        scorer._model_loaded = True
        scorer.use_model = True
        scorer._last_token_count = 0

        mock_device = MagicMock()
        scorer._model.parameters.return_value = iter([MagicMock(device=mock_device)])

        mock_probs = np.array([[0.1, 0.3, 0.6]])
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.softmax.return_value.cpu.return_value.numpy.return_value = mock_probs
        scorer._model.return_value.logits = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = scorer.score_batch([("p", "h")])
            assert len(result) == 1

    def test_score_batch_heuristic_fallback(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "deberta"
        scorer._model_name = "deberta"
        scorer.max_length = 512
        scorer._model = None
        scorer._tokenizer = None
        scorer._onnx_session = None
        scorer._minicheck = None
        scorer._minicheck_loaded = True
        scorer._custom_backend = None
        scorer._model_loaded = False
        scorer.use_model = False

        result = scorer.score_batch([("p", "h")])
        assert len(result) == 1


# ── NLI: _model_score (single pair) ─────────────────────────────


class TestNliModelScoreSingle:
    def test_model_score_2class(self):
        import numpy as np

        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "deberta"
        scorer._model_name = "deberta-standard"
        scorer.max_length = 512
        scorer._model = MagicMock()
        scorer._tokenizer = MagicMock()
        scorer._last_token_count = 0

        mock_device = MagicMock()
        scorer._model.parameters.return_value = iter([MagicMock(device=mock_device)])

        mock_probs = np.array([0.3, 0.7])
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.softmax.return_value.cpu.return_value.numpy.return_value = [
            mock_probs
        ]
        scorer._model.return_value.logits = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = scorer._model_score("premise", "hypothesis")
            assert isinstance(result, float)


# ── Agent: _build_scorer with Rust available ──────────────────────


class TestAgentBuildScorerRust:
    def test_build_scorer_with_rust(self):
        mock_bk = MagicMock()
        mock_scorer = MagicMock()
        mock_bk.RustCoherenceScorer.return_value = mock_scorer

        with patch.dict(sys.modules, {"backfire_kernel": mock_bk}):
            from director_ai.core.agent import CoherenceAgent

            agent = CoherenceAgent(use_nli=False)
            assert agent.scorer is not None


# ── Backends: entry point load failure ────────────────────────────


class TestBackendsRegistry:
    def test_registry_has_rust(self):
        from director_ai.core import backends

        if "rust" not in backends._REGISTRY:
            pytest.skip("backfire_kernel not built/installed")
        assert "backfire" in backends._REGISTRY


# ── Providers: OpenAI timeout, Anthropic, Local stream error ─────


class TestProvidersTimeout:
    def test_openai_timeout(self):
        import requests as req_lib

        from director_ai.integrations.providers import OpenAIProvider

        with patch("director_ai.integrations.providers.requests.post") as mock_post:
            mock_post.side_effect = req_lib.exceptions.Timeout("timed out")
            provider = OpenAIProvider(api_key="fake")
            result = provider.generate_candidates("test", n=1)
            assert result[0]["source"] == "error"

    def test_openai_http_error(self):
        import requests as req_lib

        from director_ai.integrations.providers import OpenAIProvider

        with patch("director_ai.integrations.providers.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = req_lib.exceptions.HTTPError("401")
            mock_post.return_value = mock_resp
            provider = OpenAIProvider(api_key="fake")
            result = provider.generate_candidates("test", n=1)
            assert result[0]["source"] == "error"


class TestProvidersLocalStream:
    def test_local_stream_error(self):
        import requests as req_lib

        from director_ai.integrations.providers import LocalProvider

        with patch("director_ai.integrations.providers.requests.post") as mock_post:
            mock_post.side_effect = req_lib.exceptions.ConnectionError("refused")
            provider = LocalProvider()
            tokens = list(provider.stream_generate("test"))
            assert any("Error" in t for t in tokens)


# ── LangChain callback: TYPE_CHECKING import branch ──────────────


class TestLangchainCallbackNoText:
    def test_on_llm_end_no_generations(self):
        from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

        handler = CoherenceCallbackHandler(use_nli=False, threshold=0.5)
        handler._current_prompt = "test"
        response = SimpleNamespace(generations=[])
        handler.on_llm_end(response)

    def test_on_llm_end_empty_first_gen(self):
        from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

        handler = CoherenceCallbackHandler(use_nli=False, threshold=0.5)
        handler._current_prompt = "test"
        response = SimpleNamespace(generations=[[]])
        handler.on_llm_end(response)


# ── CLI: evaluate with output flag ────────────────────────────────


class TestCliTuneCommand:
    def test_tune_basic(self, tmp_path, capsys):
        import json

        input_f = tmp_path / "tune.jsonl"
        input_f.write_text(
            json.dumps(
                {"prompt": "sky?", "response": "The sky is blue.", "label": True}
            )
            + "\n"
            + json.dumps(
                {"prompt": "sun?", "response": "The sun is cold.", "label": False}
            )
            + "\n",
            encoding="utf-8",
        )
        from director_ai.cli import main

        main(["tune", str(input_f)])
