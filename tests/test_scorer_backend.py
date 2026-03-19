# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Scorer Backend Tests

from director_ai.core import CoherenceScorer
from director_ai.core.nli import NLIScorer


class TestScorerBackendForwarding:
    def test_default_backend_is_deberta(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        assert scorer.scorer_backend == "deberta"

    def test_backend_param_forwarded(self):
        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=True,
            scorer_backend="minicheck",
        )
        assert scorer.scorer_backend == "minicheck"
        assert scorer._nli is not None
        assert scorer._nli.backend == "minicheck"

    def test_onnx_path_forwarded(self):
        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=True,
            scorer_backend="onnx",
            onnx_path="/tmp/fake_onnx",
        )
        assert scorer.onnx_path == "/tmp/fake_onnx"
        assert scorer._nli is not None
        assert scorer._nli._onnx_path == "/tmp/fake_onnx"


class TestHybridBackend:
    def test_hybrid_backend_requires_provider(self):
        import pytest

        with pytest.raises(
            ValueError,
            match="hybrid backend requires llm_judge_provider",
        ):
            CoherenceScorer(threshold=0.5, use_nli=False, scorer_backend="hybrid")

    def test_hybrid_backend_auto_enables_judge(self):
        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            scorer_backend="hybrid",
            llm_judge_provider="openai",
        )
        assert scorer._llm_judge_enabled is True

    def test_hybrid_review_calls_judge(self):
        from unittest.mock import patch

        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            scorer_backend="hybrid",
            llm_judge_provider="openai",
        )
        with patch.object(scorer, "_llm_judge_check", return_value=0.3):
            scorer.review("What color is the sky?", "The sky is blue.")
            assert scorer._llm_judge_enabled is True
            assert scorer.scorer_backend == "hybrid"


class TestLLMJudgeParsing:
    def test_parse_json_yes(self):
        assert CoherenceScorer._parse_judge_reply(
            '{"verdict": "YES", "confidence": 90}',
        )[0]

    def test_parse_json_no(self):
        assert not CoherenceScorer._parse_judge_reply(
            '{"verdict": "NO", "confidence": 20}',
        )[0]

    def test_parse_fallback_string_yes(self):
        assert CoherenceScorer._parse_judge_reply("YES, I believe so")[0]

    def test_parse_fallback_string_no(self):
        assert not CoherenceScorer._parse_judge_reply("NO, it is incorrect")[0]

    def test_parse_malformed_json_fallback(self):
        assert CoherenceScorer._parse_judge_reply("{invalid json YES}")[0]

    def test_custom_model_stored(self):
        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            scorer_backend="hybrid",
            llm_judge_provider="openai",
            llm_judge_model="gpt-4o",
        )
        assert scorer._llm_judge_model == "gpt-4o"

    def test_judge_check_uses_custom_model(self):
        from unittest.mock import MagicMock, patch

        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            scorer_backend="hybrid",
            llm_judge_provider="openai",
            llm_judge_model="gpt-4o",
        )
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"verdict": "YES", "confidence": 85}'
        mock_client.chat.completions.create.return_value = mock_resp

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = scorer._llm_judge_check("prompt", "response", 0.5)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs[1]["model"] == "gpt-4o"
        assert result < 0.5  # agrees â†’ lower divergence


class TestRustBackend:
    def test_rust_fallback_without_backfire_kernel(self):
        """When backfire_kernel is not installed, _rust_scorer is None."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"backfire_kernel": None}):
            scorer = CoherenceScorer(
                threshold=0.5,
                use_nli=False,
                scorer_backend="rust",
            )
            assert scorer._rust_scorer is None
            assert scorer.scorer_backend == "rust"

    def test_rust_dispatch_with_mock(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(threshold=0.5, use_nli=False, scorer_backend="rust")
        mock_score = MagicMock()
        mock_score.score = 0.85
        mock_score.h_logical = 0.1
        mock_score.h_factual = 0.05
        mock_rust = MagicMock()
        mock_rust.review.return_value = (True, mock_score)
        scorer._rust_scorer = mock_rust

        approved, cs = scorer.review("test prompt", "test response")
        mock_rust.review.assert_called_once_with("test prompt", "test response")
        assert approved is True

    def test_rust_threshold_forwarded(self):
        """Threshold value is stored even when Rust import fails."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"backfire_kernel": None}):
            scorer = CoherenceScorer(
                threshold=0.7,
                use_nli=False,
                scorer_backend="rust",
            )
            assert scorer.threshold == 0.7

    def test_rust_knowledge_callback_wiring(self):
        from unittest.mock import MagicMock, patch

        mock_store = MagicMock()
        mock_store.retrieve_context.return_value = "test context"

        mock_config_cls = MagicMock()
        mock_scorer_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "backfire_kernel": MagicMock(
                    BackfireConfig=mock_config_cls,
                    RustCoherenceScorer=mock_scorer_cls,
                ),
            },
        ):
            scorer = CoherenceScorer(
                threshold=0.5,
                use_nli=False,
                scorer_backend="rust",
                ground_truth_store=mock_store,
            )
            assert scorer._rust_scorer is not None
            call_kwargs = mock_scorer_cls.call_args[1]
            assert call_kwargs["knowledge_callback"] is not None


class TestRustDivergenceDispatch:
    def test_rust_logical_divergence(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(threshold=0.5, use_nli=False, scorer_backend="rust")
        mock_score = MagicMock()
        mock_score.score = 0.9
        mock_score.h_logical = 0.15
        mock_rust = MagicMock()
        mock_rust.review.return_value = (True, mock_score)
        scorer._rust_scorer = mock_rust

        result = scorer.calculate_logical_divergence("p", "r")
        assert result == 0.15

    def test_rust_factual_divergence(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(threshold=0.5, use_nli=False, scorer_backend="rust")
        mock_score = MagicMock()
        mock_score.score = 0.8
        mock_score.h_factual = 0.2
        mock_rust = MagicMock()
        mock_rust.review.return_value = (True, mock_score)
        scorer._rust_scorer = mock_rust

        result = scorer.calculate_factual_divergence("p", "r")
        assert result == 0.2

    def test_rust_strict_mode_import_raises(self):
        from unittest.mock import patch

        import pytest

        with (
            patch.dict("sys.modules", {"backfire_kernel": None}),
            pytest.raises(ImportError),
        ):
            CoherenceScorer(
                threshold=0.5,
                use_nli=False,
                scorer_backend="rust",
                strict_mode=True,
            )


class TestPrivacyModeRedaction:
    def test_privacy_mode_judge_redacts(self):
        from unittest.mock import MagicMock, patch

        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            scorer_backend="hybrid",
            llm_judge_provider="openai",
            privacy_mode=True,
        )
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"verdict": "YES", "confidence": 80}'
        mock_client.chat.completions.create.return_value = mock_resp

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            scorer._llm_judge_check("email: user@test.com", "response", 0.5)

        sent_prompt = mock_client.chat.completions.create.call_args[1]["messages"][0][
            "content"
        ]
        assert "user@test.com" not in sent_prompt
        assert "[EMAIL]" in sent_prompt


class TestNLIBatchLength:
    def test_batch_returns_correct_length(self):
        nli = NLIScorer(use_model=False, backend="deberta")
        pairs = [("premise", "hypothesis")] * 5
        results = nli.score_batch(pairs)
        assert len(results) == 5

    def test_empty_batch(self):
        nli = NLIScorer(use_model=False, backend="deberta")
        assert nli.score_batch([]) == []

    def test_minicheck_batch_fallback_length(self):
        nli = NLIScorer(use_model=False, backend="minicheck")
        pairs = [("a", "b"), ("c", "d"), ("e", "f")]
        results = nli.score_batch(pairs)
        assert len(results) == 3


class TestAggregationPassthrough:
    def test_scorer_passes_aggregation_params(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig.from_profile("summarization")
        cfg.llm_judge_provider = "openai"
        scorer = cfg.build_scorer()
        assert scorer._fact_inner_agg == "min"
        assert scorer._fact_outer_agg == "trimmed_mean"
        assert scorer._fact_retrieval_top_k == 8
        assert scorer.W_LOGIC == 0.0
        assert scorer.W_FACT == 1.0

    def test_default_config_max_max(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        assert scorer._fact_inner_agg == "max"
        assert scorer._fact_outer_agg == "max"

    def test_scorer_calls_nli_with_agg_params(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(threshold=0.5, use_nli=True)
        scorer._fact_inner_agg = "min"
        scorer._fact_outer_agg = "mean"
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.score_chunked.return_value = (0.3, [0.3])
        scorer._nli = mock_nli

        mock_store = MagicMock()
        mock_store.retrieve_context.return_value = "some context"
        scorer.ground_truth_store = mock_store

        scorer.calculate_factual_divergence("prompt", "output")
        mock_store.retrieve_context.assert_called_once_with(
            "prompt",
            top_k=3,
            tenant_id="",
        )
        mock_nli.score_chunked.assert_called_once_with(
            "some context",
            "output",
            inner_agg="min",
            outer_agg="mean",
            premise_ratio=0.4,
            overlap_ratio=0.5,
        )


class TestWLogicZeroShortCircuit:
    def test_w_logic_zero_skips_logical_divergence(self):
        from unittest.mock import MagicMock, patch

        scorer = CoherenceScorer(threshold=0.2, use_nli=True, w_logic=0.0, w_fact=1.0)
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli._ensure_model.return_value = True
        mock_nli._score_chunked_with_counts.return_value = (0.3, [0.3], 1, 1)
        scorer._nli = mock_nli

        mock_store = MagicMock()
        mock_store.retrieve_context_with_chunks.return_value = [
            MagicMock(text="some context", distance=0.0, source="test"),
        ]
        scorer.ground_truth_store = mock_store

        # Patch isinstance check so VectorGroundTruthStore matches
        with patch(
            "director_ai.core.scorer.CoherenceScorer.calculate_logical_divergence",
        ) as mock_logic:
            h_logic, h_fact, coherence, _ = scorer._heuristic_coherence(
                "prompt",
                "action",
            )
            mock_logic.assert_not_called()
            assert h_logic == 0.0

    def test_default_retrieval_top_k(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        assert scorer._fact_retrieval_top_k == 3


class TestLocalJudgeFallbackPaths:
    """Exercise local-judge code paths without torch (model=None)."""

    def _scorer(self):
        return CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="local",
            llm_judge_model="",
            scorer_backend="hybrid",
        )

    def test_local_judge_check_returns_nli_score(self):
        assert self._scorer()._local_judge_check("p", "r", nli_score=0.42) == 0.42

    def test_should_not_escalate_without_model(self):
        assert self._scorer()._should_escalate(0.5) is False

    def test_llm_judge_check_routes_local_fallback(self):
        assert self._scorer()._llm_judge_check("p", "r", 0.37) == 0.37
