"""Final coverage push — targets every remaining testable gap."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ── metrics: histogram overflow + gauge_inc/gauge_dec new gauge ──────


class TestMetricsHistogramOverflow:
    def test_histogram_truncates_when_exceeding_max_samples(self):
        from director_ai.core.metrics import _Histogram

        h = _Histogram(max_samples=20)
        for i in range(30):
            h.observe(float(i))
        assert h.count <= 20

    def test_gauge_inc_creates_gauge_on_first_call(self):
        from director_ai.core.metrics import MetricsCollector

        m = MetricsCollector(enabled=True)
        m.gauge_inc("new_gauge_inc_test")
        data = m.get_metrics()
        assert data["gauges"]["new_gauge_inc_test"] == 1.0

    def test_gauge_dec_creates_gauge_on_first_call(self):
        from director_ai.core.metrics import MetricsCollector

        m = MetricsCollector(enabled=True)
        m.gauge_dec("new_gauge_dec_test", 2.0)
        data = m.get_metrics()
        assert data["gauges"]["new_gauge_dec_test"] == -2.0


# ── sdk_guard: periodic check fires _handle_failure ─────────────────


class TestSdkGuardPeriodicCheck:
    def _make_scorer(self, approved=True):
        scorer = MagicMock()
        score = SimpleNamespace(
            score=0.3 if not approved else 0.9,
            h_logical=0.0,
            h_factual=0.0,
            warning=False,
            evidence=None,
        )
        scorer.review.return_value = (approved, score)
        return scorer

    def test_openai_stream_periodic_check_triggers_on_interval(self):
        from director_ai.integrations.sdk_guard import (
            STREAM_CHECK_INTERVAL,
            _GuardedOpenAIStream,
        )

        chunks = []
        for i in range(STREAM_CHECK_INTERVAL + 1):
            c = SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=f"t{i}"))]
            )
            chunks.append(c)

        scorer = self._make_scorer(approved=False)
        stream = _GuardedOpenAIStream(iter(chunks), scorer, "log", "test prompt")
        collected = list(stream)
        assert len(collected) == len(chunks)
        assert scorer.review.call_count >= 2  # periodic + final

    def test_anthropic_stream_periodic_check_triggers(self):
        from director_ai.integrations.sdk_guard import (
            STREAM_CHECK_INTERVAL,
            _GuardedAnthropicStream,
        )

        events = []
        for i in range(STREAM_CHECK_INTERVAL + 1):
            # _extract_anthropic_event_text checks event.text first
            e = SimpleNamespace(text=f"w{i}")
            events.append(e)

        scorer = self._make_scorer(approved=False)
        stream = _GuardedAnthropicStream(iter(events), scorer, "log", "test prompt")
        collected = list(stream)
        assert len(collected) == len(events)
        assert scorer.review.call_count >= 2


# ── providers: local stream SSE parse errors ────────────────────────


class TestLocalProviderStreamParsing:
    def test_local_stream_json_decode_error_skips_line(self):
        from director_ai.integrations.providers import LocalProvider

        p = LocalProvider(api_url="http://fake:8080/v1/chat/completions")
        lines = [
            b"",  # empty line — hits continue on line 382
            b"event: ping",  # non-data line — hits continue
            b"data: not-json",
            b'data: {"choices":[{"delta":{"content":"hi"}}]}',
            b"data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [raw.decode() for raw in lines]
        mock_resp.raise_for_status.return_value = None

        with patch(
            "director_ai.integrations.providers.requests.post", return_value=mock_resp
        ):
            tokens = list(p.stream_generate("test"))
        assert tokens == ["hi"]

    def test_local_stream_index_error_skips_line(self):
        from director_ai.integrations.providers import LocalProvider

        p = LocalProvider(api_url="http://fake:8080/v1/chat/completions")
        lines = [
            'data: {"choices":[]}',  # IndexError on [0]
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = lines
        mock_resp.raise_for_status.return_value = None

        with patch(
            "director_ai.integrations.providers.requests.post", return_value=mock_resp
        ):
            tokens = list(p.stream_generate("test"))
        assert tokens == ["ok"]


# ── config: rate_limit_rpm < 0, nli_devices, list coerce ────────────


class TestConfigEdgeCases:
    def test_rate_limit_rpm_negative_raises(self):
        import pytest

        from director_ai.core.config import DirectorConfig

        with pytest.raises(ValueError, match="rate_limit_rpm"):
            DirectorConfig(use_nli=False, rate_limit_rpm=-1)

    def test_nli_devices_parsing(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(use_nli=True, nli_devices="cpu,cuda:0")
        store = MagicMock()
        scorer = cfg.build_scorer(store=store)
        assert scorer is not None

    def test_coerce_list_type(self):
        from director_ai.core.config import _coerce

        result = _coerce("a, b, c", "list[str]")
        assert result == ["a", "b", "c"]


# ── nli: _model_score 2-class branch + minicheck MiniCheck raises ───


class TestNli2ClassAndMinicheck:
    def test_minicheck_init_runtime_error(self):
        """MiniCheck() raising RuntimeError falls through to heuristic."""
        scorer = object.__new__(
            __import__("director_ai.core.nli", fromlist=["NLIScorer"]).NLIScorer
        )
        scorer._model_name = "DeBERTa-v3-base"
        scorer._minicheck = None
        scorer._minicheck_loaded = False

        fake_minicheck = MagicMock()
        fake_minicheck.MiniCheck.side_effect = RuntimeError("GPU OOM")
        with patch.dict(sys.modules, {"minicheck": fake_minicheck}):
            result = scorer._ensure_minicheck()
        assert result is False


# ── cli: quickstart unknown arg, ingest empty line, bench --model ────


class TestCliEdgeBranches:
    def test_quickstart_unknown_arg_skipped(self):
        """Extra args in quickstart are silently skipped (line 114)."""
        import os
        import tempfile

        from director_ai.cli import _cmd_quickstart

        with tempfile.TemporaryDirectory() as td:
            orig = os.getcwd()
            try:
                os.chdir(td)
                _cmd_quickstart(["--unknown-flag", "value", "--profile", "fast"])
                assert os.path.isdir("director_guard")
            finally:
                os.chdir(orig)

    def test_ingest_skips_empty_lines_in_jsonl(self, tmp_path):
        """Empty lines in JSONL are skipped (line 411)."""
        from director_ai.cli import _cmd_ingest

        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text(
            '{"text":"first"}\n\n\n{"text":"second"}\n',
            encoding="utf-8",
        )

        with patch("director_ai.core.config.DirectorConfig.from_env") as mock_from_env:
            mock_cfg = MagicMock()
            mock_store = MagicMock()
            mock_store.ingest.return_value = 2
            mock_cfg.build_store.return_value = mock_store
            mock_from_env.return_value = mock_cfg
            _cmd_ingest([str(jsonl)])
        texts = mock_store.ingest.call_args[0][0]
        assert len(texts) == 2

    def test_ingest_skips_oversized_file(self, tmp_path):
        """Files exceeding _INGEST_MAX_FILE_SIZE are skipped (lines 402-403)."""
        from director_ai import cli

        orig_limit = cli._INGEST_MAX_FILE_SIZE
        try:
            cli._INGEST_MAX_FILE_SIZE = 10  # 10 bytes
            big = tmp_path / "big.txt"
            big.write_text("x" * 100, encoding="utf-8")

            with patch(
                "director_ai.core.config.DirectorConfig.from_env"
            ) as mock_from_env:
                mock_cfg = MagicMock()
                mock_store = MagicMock()
                mock_store.ingest.return_value = 0
                mock_cfg.build_store.return_value = mock_store
                mock_from_env.return_value = mock_cfg
                cli._cmd_ingest([str(tmp_path)])
            texts = mock_store.ingest.call_args[0][0]
            assert len(texts) == 0
        finally:
            cli._INGEST_MAX_FILE_SIZE = orig_limit

    def test_eval_model_flag_parsed(self):
        """--model flag in eval is parsed (lines 469-473)."""
        from director_ai.cli import _cmd_eval

        mock_run = MagicMock(return_value={"results": []})
        mock_print = MagicMock()
        bench_mod = types.ModuleType("benchmarks.run_all")
        bench_mod._run_suite = mock_run
        bench_mod._print_comparison_table = mock_print

        with patch.dict(
            sys.modules,
            {
                "benchmarks": types.ModuleType("benchmarks"),
                "benchmarks.run_all": bench_mod,
            },
        ):
            _cmd_eval(["--model", "deberta-large"])
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == "deberta-large"

    def test_eval_unknown_flag_skipped(self):
        """Unknown flags in eval skip via else branch (line 473)."""
        from director_ai.cli import _cmd_eval

        mock_run = MagicMock(return_value={"results": []})
        mock_print = MagicMock()
        bench_mod = types.ModuleType("benchmarks.run_all")
        bench_mod._run_suite = mock_run
        bench_mod._print_comparison_table = mock_print

        with patch.dict(
            sys.modules,
            {
                "benchmarks": types.ModuleType("benchmarks"),
                "benchmarks.run_all": bench_mod,
            },
        ):
            _cmd_eval(["--unknown-flag", "val"])
        mock_run.assert_called_once()

    def test_tune_output_flag(self, tmp_path):
        """--output flag in tune sets output path (line 636)."""
        from director_ai.cli import _cmd_tune

        labeled = tmp_path / "labeled.jsonl"
        out = tmp_path / "config.yaml"
        labeled.write_text(
            json.dumps({"prompt": "q", "response": "a", "label": True})
            + "\n"
            + json.dumps({"prompt": "q2", "response": "a2", "label": False})
            + "\n",
            encoding="utf-8",
        )

        tune_result = SimpleNamespace(
            threshold=0.65,
            w_logic=0.5,
            w_fact=0.5,
            balanced_accuracy=0.85,
            precision=0.9,
            recall=0.8,
            f1=0.85,
            samples=2,
        )
        with patch("director_ai.core.tuner.tune", return_value=tune_result):
            _cmd_tune([str(labeled), "--output", str(out)])
        assert out.exists()

    def test_serve_unknown_flag_skipped(self):
        """Unknown flags in serve are skipped (line 723)."""
        from director_ai.cli import _cmd_serve

        mock_cfg = MagicMock()
        mock_cfg.profile = "fast"
        mock_cfg.server_host = "0.0.0.0"
        mock_cfg.server_port = 8080

        mock_uv = MagicMock()
        with (
            patch(
                "director_ai.core.config.DirectorConfig.from_profile",
                return_value=mock_cfg,
            ),
            patch.dict(sys.modules, {"uvicorn": mock_uv}),
            patch("director_ai.server.create_app", MagicMock()),
        ):
            _cmd_serve(["--unknown", "val", "--profile", "fast"])
        mock_uv.run.assert_called_once()


# ── server: NLI gauge, rate_limit no slowapi, halted-all stats ───────


_HAS_FASTAPI = __import__("importlib").util.find_spec("fastapi") is not None
_skip_no_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


@_skip_no_server
class TestServerNliGaugeAndRateLimit:
    def test_nli_gauge_set_on_startup(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.server import create_app

        cfg = DirectorConfig(
            use_nli=True,
            coherence_threshold=0.6,
            hard_limit=0.3,
            soft_limit=0.5,
        )
        create_app(config=cfg)

    def test_rate_limit_warning_when_slowapi_missing(self):
        import director_ai.server as srv

        orig = srv._SLOWAPI_AVAILABLE
        try:
            srv._SLOWAPI_AVAILABLE = False
            from director_ai.core.config import DirectorConfig

            cfg = DirectorConfig(use_nli=False, rate_limit_rpm=100)
            app = srv.create_app(config=cfg)
            assert not hasattr(app.state, "limiter")
        finally:
            srv._SLOWAPI_AVAILABLE = orig


# ── vector_store: qdrant _ensure_collection create branch ────────────


class TestVectorStoreQdrantEnsureCollection:
    def test_qdrant_creates_collection_on_not_found(self):
        mock_qdrant = MagicMock()
        fake_qdrant_client = types.ModuleType("qdrant_client")
        fake_qdrant_client.QdrantClient = MagicMock(return_value=mock_qdrant)
        fake_qdrant_models = types.ModuleType("qdrant_client.models")
        fake_qdrant_models.Distance = MagicMock()
        fake_qdrant_models.Distance.COSINE = "Cosine"
        fake_qdrant_models.VectorParams = MagicMock()

        mock_qdrant.get_collection.side_effect = Exception("Not found")

        with patch.dict(
            sys.modules,
            {
                "qdrant_client": fake_qdrant_client,
                "qdrant_client.models": fake_qdrant_models,
            },
        ):
            from director_ai.core.vector_store import QdrantBackend

            store = object.__new__(QdrantBackend)
            store._client = mock_qdrant
            store._collection = "test_col"
            store._vector_size = 384
            store._ensure_collection()

        mock_qdrant.create_collection.assert_called_once()


# ── Branch partial: sdk_guard _extract_prompt non-text content ────────


class TestExtractPromptBranches:
    def test_content_list_with_non_text_blocks(self):
        """Content list with non-text dict — falls through."""
        from director_ai.integrations.sdk_guard import _extract_prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "url": "http://example.com/img.png"},
                    {"type": "tool_use", "data": "xyz"},
                ],
            },
        ]
        result = _extract_prompt(messages)
        assert isinstance(result, str)

    def test_content_list_empty(self):
        """Empty content list → for-loop body never entered → return str(content)."""
        from director_ai.integrations.sdk_guard import _extract_prompt

        messages = [{"role": "user", "content": []}]
        result = _extract_prompt(messages)
        assert result == "[]"


# ── Branch partial: sdk_guard empty streams (final_check with no buffer) ──


class TestSdkGuardEmptyStreams:
    def _make_scorer(self):
        scorer = MagicMock()
        score = SimpleNamespace(
            score=0.9,
            h_logical=0.0,
            h_factual=0.0,
            warning=False,
            evidence=None,
        )
        scorer.review.return_value = (True, score)
        return scorer

    def test_openai_empty_stream_final_check_empty_buffer(self):
        """No deltas in stream → buffer empty → final_check exits (238->exit)."""
        from director_ai.integrations.sdk_guard import _GuardedOpenAIStream

        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=""))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
            ),
        ]
        scorer = self._make_scorer()
        stream = _GuardedOpenAIStream(iter(chunks), scorer, "log", "prompt")
        collected = list(stream)
        assert len(collected) == 2
        assert scorer.review.call_count == 0

    def test_anthropic_empty_stream_final_check_empty_buffer(self):
        """No text in events → buffer empty → final_check exits (346->exit)."""
        from director_ai.integrations.sdk_guard import _GuardedAnthropicStream

        events = [
            SimpleNamespace(type="message_start"),
            SimpleNamespace(type="content_block_start"),
        ]
        scorer = self._make_scorer()
        stream = _GuardedAnthropicStream(iter(events), scorer, "log", "prompt")
        collected = list(stream)
        assert len(collected) == 2
        assert scorer.review.call_count == 0


# ── Branch partial: sdk_guard async iteration with empty deltas ───────


class TestSdkGuardAsyncIteration:
    def _make_scorer(self):
        scorer = MagicMock()
        score = SimpleNamespace(
            score=0.9,
            h_logical=0.0,
            h_factual=0.0,
            warning=False,
            evidence=None,
        )
        scorer.review.return_value = (True, score)
        return scorer

    @pytest.mark.asyncio
    async def test_async_openai_stream_empty_and_nonempty_deltas(self):
        """Async OpenAI: some empty deltas (208->213) + not-at-interval (222->227)."""
        from director_ai.integrations.sdk_guard import _GuardedOpenAIStream

        async def _async_chunks():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=""))]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))]
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=""))]
            )

        scorer = self._make_scorer()
        stream = _GuardedOpenAIStream(_async_chunks(), scorer, "log", "prompt")
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        assert len(collected) == 3

    @pytest.mark.asyncio
    async def test_async_anthropic_stream_empty_and_nonempty_text(self):
        """Async Anthropic: events with and without text."""
        from director_ai.integrations.sdk_guard import _GuardedAnthropicStream

        async def _async_events():
            yield SimpleNamespace(type="message_start")
            yield SimpleNamespace(text="hello")
            yield SimpleNamespace(type="content_block_stop")

        scorer = self._make_scorer()
        stream = _GuardedAnthropicStream(_async_events(), scorer, "log", "prompt")
        collected = []
        async for event in stream:
            collected.append(event)
        assert len(collected) == 3


# ── Branch partial: streaming check_halt window above threshold ───────


class TestStreamingWindowAboveThreshold:
    def test_window_full_but_average_above_threshold(self):
        """Window fills up but avg >= threshold — falls through."""
        from director_ai.core.streaming import StreamingKernel

        k = StreamingKernel(
            hard_limit=0.1,
            window_size=3,
            window_threshold=0.4,
            trend_window=10,
            trend_threshold=0.3,
        )
        for score in [0.8, 0.7, 0.9, 0.85]:
            halted = k.check_halt(score)
        assert halted is False


# ── Branch partial: langgraph rewrite with empty context ──────────────


class TestLanggraphEmptyContextRewrite:
    def test_rewrite_mode_with_no_context(self):
        """on_fail='rewrite' + empty context → 68->71 (no rewrite done)."""
        from director_ai.integrations.langgraph import director_ai_node

        node = director_ai_node(
            facts={"key": "value"},
            on_fail="rewrite",
            threshold=0.99,
        )
        state = {
            "query": "What is the meaning of life?",
            "response": "Unknown cosmic answer that won't pass threshold.",
        }
        result = node(state)
        assert (
            "director_ai_rewritten" not in result
            or result.get("director_ai_rewritten") is not True
        )


# ── Branch partial: providers Anthropic non-text block ────────────────


class TestProviderAnthropicNonTextBlock:
    def test_anthropic_response_with_non_text_blocks(self):
        """Content blocks without type='text' → 222->221 (loop continues)."""
        from director_ai.integrations.providers import AnthropicProvider

        p = AnthropicProvider(api_key="fake-key")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "content": [
                {"type": "tool_use", "data": "xyz"},
                {"type": "text", "text": "actual answer"},
            ],
        }
        with patch(
            "director_ai.integrations.providers.requests.post", return_value=mock_resp
        ):
            candidates = p.generate_candidates("test", n=1)
        assert candidates[0]["text"] == "actual answer"


# ── Branch partial: providers OpenAI SSE empty content delta ──────────


class TestProviderOpenAIEmptyContent:
    def test_openai_sse_empty_content_skipped(self):
        """SSE delta with empty content → 163->153 (yield skipped, next line)."""
        from director_ai.integrations.providers import OpenAIProvider

        p = OpenAIProvider(api_key="fake-key")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":""}}]}',
            'data: {"choices":[{"delta":{"content":"word"}}]}',
            "data: [DONE]",
        ]
        with patch(
            "director_ai.integrations.providers.requests.post", return_value=mock_resp
        ):
            tokens = list(p.stream_generate("test"))
        assert tokens == ["word"]


# ── Branch partial: policy non-matching pattern ───────────────────────


class TestPolicyNonMatchingPattern:
    def test_some_patterns_match_some_dont(self):
        """Multiple forbidden patterns, only some match → 132->131 (loop continues)."""
        from director_ai.core.policy import Policy

        policy = Policy.from_dict(
            {
                "forbidden": ["badword", "anotherbad"],
                "required": [],
                "max_length": 1000,
            }
        )
        violations = policy.check("this contains badword but not the other")
        assert len(violations) == 1
        assert "badword" in violations[0].detail


# ── Branch partial: vector_store SentenceTransformer zero similarity ──


class TestVectorStoreZeroSimilarity:
    def test_query_with_zero_similarity_items_filtered(self):
        """similarity <= 0 → 128->127 (item not appended)."""
        import threading

        import numpy as np

        from director_ai.core.vector_store import SentenceTransformerBackend

        backend = object.__new__(SentenceTransformerBackend)
        backend._lock = threading.Lock()
        backend._docs = [
            {"id": "a", "text": "hello", "metadata": {}},
            {"id": "b", "text": "world", "metadata": {}},
        ]
        backend._embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
        ]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1.0, 0.0, 0.0])
        backend._model = mock_model

        results = backend.query("test", n_results=10)
        assert len(results) == 1
        assert results[0]["id"] == "a"
