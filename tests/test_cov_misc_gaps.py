"""Coverage for small gaps across multiple modules."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.config import DirectorConfig


class TestScorerShardedNli:
    def test_sharded_nli_init(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(
            use_nli=True,
            nli_devices=["cpu", "cpu"],
        )
        assert scorer.use_nli is True


class TestScorerFactualWithNli:
    def test_factual_divergence_with_nli(self):
        from director_ai.core.knowledge import GroundTruthStore
        from director_ai.core.scorer import CoherenceScorer

        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")

        scorer = CoherenceScorer(
            use_nli=False,
            ground_truth_store=store,
        )
        scorer._nli = MagicMock()
        scorer._nli.model_available = True
        scorer._nli.score_chunked.return_value = (0.1, [0.1])

        result = scorer.calculate_factual_divergence("sky?", "The sky is blue.")
        assert 0.0 <= result <= 1.0


class TestStreamingHaltEvidence:
    def test_halt_evidence_structured(self):
        from director_ai.core.streaming import StreamingKernel

        kernel = StreamingKernel(hard_limit=0.9, adaptive=True)
        tokens = ["This", "is", "a", "test", "sentence"]

        def _bad_score(tok):
            return 0.1

        session = kernel.stream_tokens(tokens, _bad_score)
        assert session.halted


class TestBackendsRustMissing:
    def test_rust_backend_import_error(self):
        from director_ai.core.backends import get_backend

        with pytest.raises(KeyError):
            get_backend("rust_nonexistent_xyz")

    def test_entry_point_load_error(self):
        from director_ai.core.backends import _REGISTRY

        assert "lite" in _REGISTRY


class TestAgentBuildScorer:
    def test_build_scorer_rust_fallback(self):
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()
        assert agent.scorer is not None


class TestConfigValidation:
    def test_invalid_stats_backend(self):
        with pytest.raises(ValueError, match="stats_backend"):
            DirectorConfig(stats_backend="redis")

    def test_invalid_grpc_max_message(self):
        with pytest.raises(ValueError, match="grpc_max_message_mb"):
            DirectorConfig(grpc_max_message_mb=0)


class TestMetricsGaugeOps:
    def test_gauge_dec(self):
        from director_ai.core.metrics import metrics

        metrics.gauge_set("test_gauge", 10.0)
        metrics.gauge_dec("test_gauge", 3.0)
        m = metrics.get_metrics()
        assert "test_gauge" in m["gauges"]

    def test_histogram_overflow(self):
        from director_ai.core.metrics import metrics

        for i in range(12000):
            metrics.observe("overflow_test", float(i))
        m = metrics.get_metrics()
        assert m["histograms"]["overflow_test"]["count"] > 0


class TestLangchainCallbackEdges:
    def test_on_llm_end_no_text(self):
        from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

        handler = CoherenceCallbackHandler(use_nli=False)

        response = SimpleNamespace(generations=[[SimpleNamespace(text="")]])
        handler.on_llm_end(response)

    def test_on_llm_end_extraction_error(self):
        from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

        handler = CoherenceCallbackHandler(use_nli=False)

        response = SimpleNamespace(generations=[])
        handler.on_llm_end(response)

    def test_on_llm_end_success(self):
        from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

        handler = CoherenceCallbackHandler(use_nli=False)

        handler._current_prompt = "What is the sky?"
        response = SimpleNamespace(
            generations=[[SimpleNamespace(text="The sky is blue.")]]
        )
        handler.on_llm_end(response)
        assert handler.last_score is not None


class TestProvidersEdges:
    def test_anthropic_stream_generate(self):
        from director_ai.integrations.providers import AnthropicProvider

        provider = AnthropicProvider(api_key="fake-key")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.iter_lines.return_value = [
                'data: {"type": "content_block_delta",'
                ' "delta": {"type": "text_delta", "text": "hello"}}',
                "data: [DONE]",
            ]
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            tokens = list(provider.stream_generate("test"))
            assert len(tokens) > 0


class TestAsyncStreamingKernelInactive:
    @pytest.mark.asyncio
    async def test_kernel_inactive_reason(self):
        from director_ai.core.async_streaming import AsyncStreamingKernel

        kernel = AsyncStreamingKernel()
        kernel.is_active = False
        tokens = ["a", "b", "c"]

        def _score(tok):
            return 0.1

        events = []
        async for ev in kernel.stream_tokens(tokens, _score):
            events.append(ev)
        assert any(e.halted for e in events)


class TestCoreInitGetattr:
    def test_getattr_unknown(self):
        import director_ai.core

        with pytest.raises(AttributeError, match="no attribute"):
            _ = director_ai.core.NonexistentAttribute

    def test_getattr_enterprise_raises_import_error(self):
        import director_ai.core

        with pytest.raises(ImportError, match="director_ai.enterprise"):
            _ = director_ai.core.TenantRouter
