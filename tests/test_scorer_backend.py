# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Scorer Backend Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from director_ai.core import CoherenceScorer
from director_ai.core.nli import NLIScorer


class TestScorerBackendForwarding:
    def test_default_backend_is_deberta(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        assert scorer.scorer_backend == "deberta"

    def test_backend_param_forwarded(self):
        scorer = CoherenceScorer(
            threshold=0.5, use_nli=True, scorer_backend="minicheck"
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
            ValueError, match="hybrid backend requires llm_judge_provider"
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
            '{"verdict": "YES", "confidence": 90}'
        )

    def test_parse_json_no(self):
        assert not CoherenceScorer._parse_judge_reply(
            '{"verdict": "NO", "confidence": 20}'
        )

    def test_parse_fallback_string_yes(self):
        assert CoherenceScorer._parse_judge_reply("YES, I believe so")

    def test_parse_fallback_string_no(self):
        assert not CoherenceScorer._parse_judge_reply("NO, it is incorrect")

    def test_parse_malformed_json_fallback(self):
        assert CoherenceScorer._parse_judge_reply("{invalid json YES}")

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
        assert result < 0.5  # agrees → lower divergence


class TestRustBackend:
    def test_rust_fallback_without_backfire_kernel(self):
        """When backfire_kernel is not installed, _rust_scorer is None."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"backfire_kernel": None}):
            scorer = CoherenceScorer(
                threshold=0.5, use_nli=False, scorer_backend="rust"
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
                threshold=0.7, use_nli=False, scorer_backend="rust"
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
                )
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
