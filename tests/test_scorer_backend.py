# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Scorer Backend Tests
"""Multi-angle tests for scorer backend dispatch pipeline.

Covers: DeBERTa, ONNX, Lite, Rust, MiniCheck backend routing, score
ranges, batch scoring, heuristic fallback, pipeline integration with
CoherenceScorer, and performance documentation.
"""

from director_ai.core import CoherenceScorer
from director_ai.core.nli import NLIScorer


class TestReviewBatchContracts:
    def test_rejects_short_logical_batch(self):
        from unittest.mock import MagicMock

        import pytest

        scorer = CoherenceScorer(threshold=0.5, use_nli=True)
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.score_batch.return_value = [0.1]
        scorer._nli = mock_nli

        with pytest.raises(RuntimeError, match="logical NLI batch"):
            scorer.review_batch(
                [
                    ("prompt one", "answer one"),
                    ("prompt two", "answer two"),
                ],
            )

    def test_rejects_short_factual_batch_instead_of_using_placeholder_score(self):
        from unittest.mock import MagicMock

        import pytest

        scorer = CoherenceScorer(threshold=0.5, use_nli=True)
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.score_batch.side_effect = [
            [0.1, 0.1],
            [0.2],
        ]
        scorer._nli = mock_nli

        mock_store = MagicMock()
        mock_store.retrieve_context.return_value = "verified context"
        scorer.ground_truth_store = mock_store

        with pytest.raises(RuntimeError, match="factual NLI batch"):
            scorer.review_batch(
                [
                    ("prompt one", "grounded answer one"),
                    ("prompt two", "grounded answer two"),
                ],
            )


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

    def test_multi_device_nli_uses_sharded_scorer(self, monkeypatch):
        captured = {}

        class FakeShardedNLIScorer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "director_ai.core.scoring.sharded_nli.ShardedNLIScorer",
            FakeShardedNLIScorer,
        )

        scorer = CoherenceScorer(
            use_nli=True,
            scorer_backend="onnx",
            nli_devices=["cuda:0", "cuda:1"],
            nli_model="custom-nli",
            nli_quantize_8bit=True,
            nli_torch_dtype="float16",
            onnx_path="model.onnx",
            onnx_batch_size=7,
            onnx_flush_timeout_ms=25,
        )

        assert isinstance(scorer._nli, FakeShardedNLIScorer)
        assert captured == {
            "devices": ["cuda:0", "cuda:1"],
            "use_model": True,
            "model_name": "custom-nli",
            "backend": "onnx",
            "quantize_8bit": True,
            "torch_dtype": "float16",
            "onnx_path": "model.onnx",
            "onnx_batch_size": 7,
            "onnx_flush_timeout_ms": 25,
        }

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
        assert result < 0.5  # agrees → lower divergence


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

        sent_messages = mock_client.chat.completions.create.call_args[1]["messages"]
        sent_prompt = "\n".join(message["content"] for message in sent_messages)
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


class TestCoherenceScorerInternalContracts:
    """Focused contracts for scorer-owned routing and compatibility surfaces."""

    def test_judge_compatibility_properties_proxy_to_composed_judge(self):
        model = object()
        tokenizer = object()

        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="local",
        )
        scorer._local_judge_model = model
        scorer._local_judge_tokenizer = tokenizer
        scorer._local_judge_device = "cpu"
        scorer._judge.task_judge_thresholds["rag"] = 0.12

        assert scorer._local_judge_model is model
        assert scorer._local_judge_tokenizer is tokenizer
        assert scorer._local_judge_device == "cpu"
        assert scorer._judge_cache is scorer._judge._judge_cache
        assert scorer._task_judge_thresholds["rag"] == 0.12

    def test_close_shuts_down_lazy_parallel_pool(self):
        scorer = CoherenceScorer(use_nli=False)

        pool = scorer._get_parallel_pool()
        assert scorer._parallel_pool is pool

        scorer.close()

        assert scorer._parallel_pool is None

    def test_dialogue_aggregation_profile_only_applies_to_default_profile(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "director_ai.core.scoring._divergence.detect_task_type",
            lambda _prompt, _response="": "dialogue",
        )
        scorer = CoherenceScorer(use_nli=False)

        assert scorer._resolve_agg_profile("User: hello\nAssistant: hi") == (
            "min",
            "mean",
            "min",
            "mean",
        )

        scorer._use_prompt_as_premise = True
        assert scorer._resolve_agg_profile("User: hello\nAssistant: hi") == (
            "max",
            "max",
            "max",
            "max",
        )

    def test_prompt_premise_factual_divergence_uses_confidence_weighted_nli(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=True)
        scorer._use_prompt_as_premise = True
        scorer._confidence_weighted_agg = True
        scorer._qa_premise_ratio = 0.7
        scorer._detect_task_type = lambda _prompt, _response="": "qa"
        scorer._should_escalate = lambda _score, task_type="default": False

        nli = MagicMock()
        nli.model_available = True
        nli.score_chunked_confidence_weighted.return_value = (0.23, [0.23, 0.31])
        scorer._nli = nli

        assert scorer.calculate_factual_divergence("Question?", "Answer.") == 0.23
        nli.score_chunked_confidence_weighted.assert_called_once_with(
            "Question?",
            "Answer.",
            inner_agg="max",
            premise_ratio=0.7,
            overlap_ratio=0.5,
        )

    def test_prompt_premise_evidence_preserves_confidence_weighted_metadata(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=True)
        scorer._use_prompt_as_premise = True
        scorer._confidence_weighted_agg = True
        scorer._detect_task_type = lambda _prompt, _response="": "summarization"
        scorer._should_escalate = lambda _score, task_type="default": False

        nli = MagicMock()
        nli.model_available = True
        nli.last_token_count = 19
        nli.last_estimated_cost = 0.0019
        nli.score_chunked_confidence_weighted.return_value = (0.18, [0.12, 0.18])
        scorer._nli = nli

        divergence, evidence = scorer.calculate_factual_divergence_with_evidence(
            "Source document with enough context.",
            "Faithful summary.",
        )

        assert divergence == 0.18
        assert evidence is not None
        assert evidence.nli_premise == "Source document with enough context."
        assert evidence.chunk_scores == [0.12, 0.18]
        assert evidence.premise_chunk_count == 1
        assert evidence.hypothesis_chunk_count == 2
        assert evidence.token_count == 19
        assert evidence.estimated_cost_usd == 0.0019

    def test_batch_requires_sequential_for_semantics_changing_modes(self):
        scorer = CoherenceScorer(use_nli=True)
        items = [("Question?", "Answer."), ("Another?", "Another answer.")]

        scorer._adaptive_router = object()
        assert scorer._review_batch_requires_sequential(items) is True
        scorer._adaptive_router = None

        scorer._retrieval_abstention_threshold = 0.2
        assert scorer._review_batch_requires_sequential(items) is True
        scorer._retrieval_abstention_threshold = 0.0

        scorer._confidence_weighted_agg = True
        assert scorer._review_batch_requires_sequential(items) is True
        scorer._confidence_weighted_agg = False

        scorer._fact_outer_agg = "mean"
        assert scorer._review_batch_requires_sequential(items) is True
        scorer._fact_outer_agg = "max"

        assert (
            scorer._review_batch_requires_sequential(
                [("Question?", "Sentence. " * 12), ("Another?", "Short.")]
            )
            is True
        )

    def test_minicheck_claim_coverage_proxy_delegates_to_task_scoring(
        self,
        monkeypatch,
    ):
        calls = []

        def fake_claim_coverage(mc_scorer, source, summary):
            calls.append((mc_scorer, source, summary))
            return 0.75, [0.1, 0.8], ["Claim one.", "Claim two."]

        monkeypatch.setattr(
            "director_ai.core.scoring._task_scoring.minicheck_claim_coverage",
            fake_claim_coverage,
        )
        mc_scorer = object()

        assert CoherenceScorer._minicheck_claim_coverage(
            mc_scorer,
            "source",
            "summary",
        ) == (0.75, [0.1, 0.8], ["Claim one.", "Claim two."])
        assert calls == [(mc_scorer, "source", "summary")]

    def test_get_minicheck_scorer_caches_success_and_failure(self, monkeypatch):
        from unittest.mock import MagicMock

        import director_ai.core.scoring.scorer as scorer_module

        ready_mc = MagicMock()
        ready_mc._ensure_minicheck.return_value = True
        monkeypatch.setattr(scorer_module, "NLIScorer", lambda **_kwargs: ready_mc)

        scorer = CoherenceScorer(use_nli=False)
        assert scorer._get_minicheck_scorer() is ready_mc
        assert scorer._get_minicheck_scorer() is ready_mc

        unavailable_mc = MagicMock()
        unavailable_mc._ensure_minicheck.return_value = False
        monkeypatch.setattr(
            scorer_module,
            "NLIScorer",
            lambda **_kwargs: unavailable_mc,
        )
        scorer = CoherenceScorer(use_nli=False)

        assert scorer._get_minicheck_scorer() is None
        assert scorer._get_minicheck_scorer() is None

    def test_meta_classifier_loads_caches_and_fails_open_or_closed(
        self,
        monkeypatch,
    ):
        import types
        from unittest.mock import patch

        classifier = object()
        fake_module = types.SimpleNamespace(
            DatasetTypeClassifier=lambda _path: classifier,
        )
        scorer = CoherenceScorer(use_nli=False)
        scorer._meta_classifier_path = "classifier.pkl"
        with patch.dict(
            "sys.modules",
            {"director_ai.core.scoring.meta_classifier": fake_module},
        ):
            assert scorer._get_meta_classifier() is classifier
            assert scorer._get_meta_classifier() is classifier

        class BrokenClassifier:
            def __init__(self, _path):
                raise RuntimeError("corrupt classifier")

        broken_module = types.SimpleNamespace(DatasetTypeClassifier=BrokenClassifier)
        scorer = CoherenceScorer(use_nli=False)
        scorer._meta_classifier_path = "broken.pkl"
        with patch.dict(
            "sys.modules",
            {"director_ai.core.scoring.meta_classifier": broken_module},
        ):
            assert scorer._get_meta_classifier() is None
        assert scorer._meta_classifier_path == ""

        class MissingClassifier:
            def __init__(self, _path):
                raise FileNotFoundError("missing classifier")

        missing_module = types.SimpleNamespace(DatasetTypeClassifier=MissingClassifier)
        scorer = CoherenceScorer(use_nli=False)
        scorer._meta_classifier_path = "missing.pkl"
        with patch.dict(
            "sys.modules",
            {"director_ai.core.scoring.meta_classifier": missing_module},
        ):
            assert scorer._get_meta_classifier() is None
        assert scorer._meta_classifier_path == ""

        scorer = CoherenceScorer(use_nli=False)
        scorer._meta_classifier_path = "broken.pkl"
        scorer._adaptive_threshold_fail_closed = True
        with patch.dict(
            "sys.modules",
            {"director_ai.core.scoring.meta_classifier": broken_module},
        ):
            import pytest

            with pytest.raises(RuntimeError, match="classifier unavailable"):
                scorer._get_meta_classifier()

    def test_dialogue_and_summarization_paths_require_model_backed_nli(self):
        import pytest

        scorer = CoherenceScorer(use_nli=False)
        with pytest.raises(RuntimeError, match="dialogue factual divergence"):
            scorer._dialogue_factual_divergence("User: hi", "Assistant: hi")
        with pytest.raises(RuntimeError, match="summarization factual divergence"):
            scorer._summarization_factual_divergence("source", "summary")

    def test_python_heuristics_cover_factual_and_logical_edge_signals(
        self,
        monkeypatch,
    ):
        import pytest

        import director_ai.core.scoring._divergence as divergence_module

        monkeypatch.setattr(
            divergence_module, "rust_heuristic_factual_divergence", None
        )
        monkeypatch.setattr(
            divergence_module, "rust_heuristic_logical_divergence", None
        )

        assert CoherenceScorer._heuristic_factual("", "answer") == 0.5
        assert CoherenceScorer._heuristic_logical("answer") == 0.5
        assert CoherenceScorer._heuristic_logical(
            "This is consistent with reality",
        ) == pytest.approx(0.1)
        assert CoherenceScorer._heuristic_logical("The opposite is true") == 0.9
        assert (
            CoherenceScorer._heuristic_logical(
                "It depends on your perspective",
            )
            == 0.5
        )

        factual = CoherenceScorer._heuristic_factual(
            "Paris is not in Germany.",
            "Paris is in Germany and Mars approves.",
        )
        assert factual == pytest.approx(0.4)

        logical = CoherenceScorer._heuristic_logical(
            "Paris is in France.",
            "Paris is in Europe.",
        )
        assert 0.0 <= logical <= 1.0

    def test_logical_divergence_uses_model_backed_chunked_nli(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=True)
        scorer._logic_inner_agg = "min"
        scorer._logic_outer_agg = "mean"
        nli = MagicMock()
        nli.model_available = True
        nli.score_chunked.return_value = (0.27, [0.27])
        scorer._nli = nli

        assert scorer.calculate_logical_divergence("prompt", "answer") == 0.27
        nli.score_chunked.assert_called_once_with(
            "prompt",
            "answer",
            inner_agg="min",
            outer_agg="mean",
            premise_ratio=0.4,
        )

    def test_compute_divergence_combines_current_component_scores(self):
        import pytest

        scorer = CoherenceScorer(use_nli=False, w_logic=0.25, w_fact=0.75)
        scorer.calculate_logical_divergence = lambda _prompt, _action: 0.2
        scorer.calculate_factual_divergence = lambda _prompt, _action: 0.6

        assert scorer.compute_divergence("prompt", "action") == pytest.approx(0.5)

    async def test_areview_delegates_to_sync_review_with_context(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=False)
        expected = (True, MagicMock())
        scorer.review = MagicMock(return_value=expected)
        session = object()

        assert (
            await scorer.areview("prompt", "action", session=session, tenant_id="t")
            == expected
        )
        scorer.review.assert_called_once_with(
            "prompt",
            "action",
            session=session,
            tenant_id="t",
        )


class TestScorerFactualDivergenceBranches:
    """Focused contracts for factual divergence routing and evidence metadata."""

    def test_prompt_premise_counted_nli_and_escalation_paths(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=True)
        scorer._use_prompt_as_premise = True
        scorer._confidence_weighted_agg = False
        scorer._detect_task_type = lambda _prompt, _response="": "summarization"
        scorer._should_escalate = lambda _score, task_type="default": True
        scorer._llm_judge_check = MagicMock(return_value=0.11)

        nli = MagicMock()
        nli.model_available = True
        nli.last_token_count = 13
        nli.last_estimated_cost = 0.0013
        nli.score_chunked.return_value = (0.27, ["chunk-score"])
        nli._score_chunked_with_counts.return_value = (0.27, [0.27], 1, 1)
        scorer._nli = nli

        assert scorer.calculate_factual_divergence("source", "summary") == 0.11
        nli.score_chunked.assert_called_once_with(
            "source",
            "summary",
            inner_agg="max",
            outer_agg="max",
            premise_ratio=0.4,
            overlap_ratio=0.5,
        )

        divergence, evidence = scorer.calculate_factual_divergence_with_evidence(
            "source",
            "summary",
        )

        assert divergence == 0.11
        assert evidence is not None
        assert evidence.nli_premise == "source"
        assert evidence.chunk_scores == [0.27]
        assert evidence.premise_chunk_count == 1
        assert evidence.hypothesis_chunk_count == 1
        scorer._llm_judge_check.assert_any_call("source", "summary", 0.27)

    def test_factual_divergence_returns_neutral_without_store_or_context(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=False)
        assert scorer.calculate_factual_divergence("prompt", "answer") == 0.5

        store = MagicMock()
        store.retrieve_context.return_value = ""
        scorer.ground_truth_store = store

        assert scorer.calculate_factual_divergence("prompt", "answer") == 0.5
        divergence, evidence = scorer.calculate_factual_divergence_with_evidence(
            "prompt",
            "answer",
        )
        assert divergence == 0.5
        assert evidence is None

    def test_adaptive_router_can_skip_retrieval(self):
        from unittest.mock import MagicMock

        class Decision:
            retrieve = False
            task_type = "creative"
            confidence = 0.91

        router = MagicMock()
        router.should_retrieve.return_value = Decision()
        store = MagicMock()
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._adaptive_router = router

        assert scorer.calculate_factual_divergence("write a poem", "poem") == 0.5
        router.should_retrieve.assert_called_once_with("write a poem", "poem")
        store.retrieve_context.assert_not_called()

    def test_rag_claim_decomposition_blends_sentence_support(self):
        from unittest.mock import MagicMock

        store = MagicMock()
        store.retrieve_context.return_value = "verified source context"
        scorer = CoherenceScorer(use_nli=True, ground_truth_store=store)
        scorer._detect_task_type = lambda _prompt, _response="": "rag"
        scorer._should_escalate = lambda _score, task_type="default": False
        scorer._claim_support_threshold = 0.6
        scorer._claim_coverage_alpha = 0.4

        nli = MagicMock()
        nli.model_available = True
        nli.score_chunked.side_effect = [
            (0.5, []),
            (0.2, []),
            (0.8, []),
        ]
        nli._split_sentences.return_value = ["Supported claim.", "Unsupported claim."]
        scorer._nli = nli

        long_response = "Supported claim. " + ("filler " * 20) + "Unsupported claim."
        assert scorer.calculate_factual_divergence("prompt", long_response) == 0.5
        assert nli.score_chunked.call_count == 3

    def test_evidence_strict_mode_rejects_without_model_backed_nli(self):
        from unittest.mock import MagicMock

        store = MagicMock()
        store.retrieve_context.return_value = "verified source context"
        scorer = CoherenceScorer(
            use_nli=False,
            strict_mode=True,
            ground_truth_store=store,
        )

        divergence, evidence = scorer.calculate_factual_divergence_with_evidence(
            "prompt",
            "answer",
        )

        assert divergence == 0.9
        assert evidence is not None
        assert evidence.nli_score == 0.9
        assert evidence.nli_premise == "verified source context"

    def test_evidence_claim_attribution_path_records_claim_metadata(self):
        from unittest.mock import MagicMock

        import pytest

        from director_ai.core.types import ClaimAttribution

        store = MagicMock()
        store.retrieve_context.return_value = "verified source context"
        scorer = CoherenceScorer(use_nli=True, ground_truth_store=store)
        scorer._detect_task_type = lambda _prompt, _response="": "rag"
        scorer._should_escalate = lambda _score, task_type="default": False

        attribution = ClaimAttribution(
            claim="Supported claim.",
            claim_index=0,
            source_sentence="verified source context",
            source_index=0,
            divergence=0.1,
            supported=True,
        )
        nli = MagicMock()
        nli.model_available = True
        nli.last_token_count = 9
        nli._cost_per_token = 0.001
        nli.reset_token_counter.return_value = None
        nli._score_chunked_with_counts.return_value = (0.2, [0.2], 1, 1)
        nli.score_claim_coverage_with_attribution.return_value = (
            1.0,
            [0.1],
            ["Supported claim."],
            [attribution],
        )
        scorer._nli = nli

        long_response = "Supported claim. " + ("grounded detail " * 12)
        divergence, evidence = scorer.calculate_factual_divergence_with_evidence(
            "prompt",
            long_response,
        )

        assert divergence == 0.2
        assert evidence is not None
        assert evidence.claim_coverage == 1.0
        assert evidence.per_claim_divergences == [0.1]
        assert evidence.claims == ["Supported claim."]
        assert evidence.attributions == [attribution]
        assert evidence.estimated_cost_usd == pytest.approx(0.009)

    def test_vector_store_evidence_uses_chunks_and_confidence_weighted_nli(self):
        from unittest.mock import MagicMock

        from director_ai.core.retrieval.vector_store import VectorGroundTruthStore
        from director_ai.core.types import EvidenceChunk

        store = VectorGroundTruthStore()
        store.retrieve_context_with_chunks = MagicMock(
            return_value=[
                EvidenceChunk(text="chunk one", distance=0.2, source="doc-a"),
                EvidenceChunk(text="chunk two", distance=0.1, source="doc-b"),
            ],
        )
        scorer = CoherenceScorer(use_nli=True, ground_truth_store=store)
        scorer._confidence_weighted_agg = True
        scorer._detect_task_type = lambda _prompt, _response="": "rag"
        scorer._should_escalate = lambda _score, task_type="default": False

        nli = MagicMock()
        nli.model_available = True
        nli.last_token_count = 5
        nli._cost_per_token = 0.002
        nli.reset_token_counter.return_value = None
        nli.score_chunked_confidence_weighted.return_value = (0.18, [0.18, 0.21])
        scorer._nli = nli

        divergence, evidence = scorer.calculate_factual_divergence_with_evidence(
            "prompt",
            "short answer",
            tenant_id="tenant-a",
        )

        assert divergence == 0.18
        store.retrieve_context_with_chunks.assert_called_once_with(
            "prompt",
            top_k=3,
            tenant_id="tenant-a",
        )
        nli.score_chunked_confidence_weighted.assert_called_once_with(
            "chunk one; chunk two",
            "short answer",
            inner_agg="max",
            premise_ratio=0.4,
            overlap_ratio=0.5,
        )
        assert evidence is not None
        assert evidence.chunks == [
            EvidenceChunk(text="chunk one", distance=0.2, source="doc-a"),
            EvidenceChunk(text="chunk two", distance=0.1, source="doc-b"),
        ]
        assert evidence.chunk_scores == [0.18, 0.21]
        assert evidence.hypothesis_chunk_count == 2
        assert evidence.estimated_cost_usd == 0.01

    def test_injection_detection_continues_when_sanitizer_is_unavailable(
        self,
        monkeypatch,
    ):
        import director_ai.core.safety.sanitizer as sanitizer_module

        class BrokenSanitizer:
            def __init__(self):
                raise RuntimeError("sanitizer unavailable")

        monkeypatch.setattr(sanitizer_module, "InputSanitizer", BrokenSanitizer)
        scorer = CoherenceScorer(use_nli=False)

        scorer.enable_injection_detection(injection_threshold=0.81)

        detector = scorer._get_injection_detector()
        assert detector is not None
        assert detector._sanitizer is None
        assert detector._cfg.injection_threshold == 0.81


class TestScorerClaimSupportIntegration:
    """CoherenceScorer summarization contract for claim-support blending."""

    @staticmethod
    def _scorer_with_nli_claim_support(support: float, layer_a_divergence: float):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(
            threshold=0.15,
            use_nli=True,
            w_logic=0.0,
            w_fact=1.0,
        )
        scorer._use_prompt_as_premise = True
        scorer._summarization_nli_baseline = 0.20
        scorer._claim_coverage_enabled = True
        scorer._claim_coverage_alpha = 0.4
        scorer._claim_support_threshold = 0.6
        scorer._minicheck_nli = None
        scorer._detect_task_type = lambda _prompt, _response="": "summarization"

        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli._score_chunked_with_counts.return_value = (
            layer_a_divergence,
            [layer_a_divergence],
            1,
            1,
        )
        mock_nli.score_chunked.return_value = (layer_a_divergence, [])

        from director_ai.core.types import ClaimAttribution

        claims = [f"Claim {idx}." for idx in range(5)]
        supported_count = int(support * len(claims))
        divergences = [0.1] * supported_count + [0.8] * (len(claims) - supported_count)
        attributions = [
            ClaimAttribution(
                claim=claim,
                claim_index=idx,
                source_sentence="source",
                source_index=0,
                divergence=divergence,
                supported=divergence < 0.6,
            )
            for idx, (claim, divergence) in enumerate(
                zip(claims, divergences, strict=False),
            )
        ]
        mock_nli.score_claim_coverage.return_value = (support, divergences, claims)
        mock_nli.score_claim_coverage_with_attribution.return_value = (
            support,
            divergences,
            claims,
            attributions,
        )
        scorer._nli = mock_nli
        return scorer

    def test_claim_support_blends_with_summarization_divergence(self):
        import pytest

        scorer = self._scorer_with_nli_claim_support(
            support=0.8,
            layer_a_divergence=0.3,
        )

        divergence, _evidence = scorer._summarization_factual_divergence(
            "source document",
            "summary",
        )

        expected_layer_a = max(0.0, (0.3 - 0.20) / (1.0 - 0.20))
        expected = 0.4 * (1.0 - 0.8) + 0.6 * expected_layer_a
        assert divergence == pytest.approx(expected)

    def test_disabling_claim_support_uses_summarization_divergence_only(self):
        import pytest

        scorer = self._scorer_with_nli_claim_support(
            support=0.5,
            layer_a_divergence=0.4,
        )
        scorer._claim_coverage_enabled = False

        divergence, evidence = scorer._summarization_factual_divergence(
            "source document",
            "summary",
        )

        expected_layer_a = max(0.0, (0.4 - 0.20) / (1.0 - 0.20))
        assert divergence == pytest.approx(expected_layer_a)
        assert evidence.claim_coverage is None

    def test_claim_support_evidence_preserves_claim_level_attribution(self):
        import pytest

        scorer = self._scorer_with_nli_claim_support(
            support=0.6,
            layer_a_divergence=0.25,
        )

        _divergence, evidence = scorer._summarization_factual_divergence(
            "source document",
            "summary",
        )

        assert evidence.claim_coverage == pytest.approx(0.6)
        assert evidence.per_claim_divergences == [0.1, 0.1, 0.1, 0.8, 0.8]
        assert evidence.claims == [
            "Claim 0.",
            "Claim 1.",
            "Claim 2.",
            "Claim 3.",
            "Claim 4.",
        ]


class TestCoherenceScorerReviewContracts:
    """Review-path contracts owned by ``CoherenceScorer``."""

    def test_score_cache_scope_combines_session_and_store_scope(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=False)
        session = MagicMock()
        session.__len__.return_value = 2
        session.context_text = "prior verified context"
        store = MagicMock()
        store.cache_scope.return_value = "kb-version-7"
        scorer.ground_truth_store = store

        assert scorer._score_cache_scope(session=session, tenant_id="tenant-a") == (
            "session:prior verified context\x1fstore:kb-version-7"
        )
        store.cache_scope.assert_called_once_with(tenant_id="tenant-a")

    def test_review_cache_hit_finalises_without_recomputing(self):
        from dataclasses import dataclass
        from unittest.mock import MagicMock

        @dataclass
        class CachedScore:
            score: float
            h_logical: float
            h_factual: float

        cache = MagicMock()
        cache.get.return_value = CachedScore(score=0.91, h_logical=0.1, h_factual=0.2)
        scorer = CoherenceScorer(use_nli=False, cache=cache)
        scorer._heuristic_coherence = MagicMock()

        approved, score = scorer.review("prompt", "answer", tenant_id="tenant-a")

        assert approved is True
        assert score.score == 0.91
        assert score.h_logical == 0.1
        assert score.h_factual == 0.2
        scorer._heuristic_coherence.assert_not_called()
        cache.get.assert_called_once_with(
            "prompt",
            "answer",
            tenant_id="tenant-a",
            scope="",
        )

    def test_review_cache_put_uses_session_and_store_scope(self):
        from unittest.mock import MagicMock

        cache = MagicMock()
        cache.get.return_value = None
        scorer = CoherenceScorer(use_nli=False, cache=cache)
        scorer._heuristic_coherence = MagicMock(return_value=(0.2, 0.3, 0.76, None))

        session = MagicMock()
        session.__len__.return_value = 1
        session.context_text = "previous turn"
        session.add_turn.return_value = None
        store = MagicMock()
        store.cache_scope.return_value = "store-snapshot"
        scorer.ground_truth_store = store

        approved, score = scorer.review(
            "prompt",
            "answer",
            session=session,
            tenant_id="tenant-a",
        )

        assert approved is True
        assert score.score == 0.76
        cache.put.assert_called_once_with(
            "prompt",
            "answer",
            0.76,
            0.2,
            0.3,
            tenant_id="tenant-a",
            scope="session:previous turn\x1fstore:store-snapshot",
        )
        session.add_turn.assert_called_once_with("prompt", "answer", 0.76)

    def test_review_blends_cross_turn_divergence_and_updates_interlock(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=True)
        nli = MagicMock()
        nli.model_available = True
        nli.score.return_value = 0.4
        scorer._nli = nli
        scorer._heuristic_coherence = MagicMock(return_value=(0.1, 0.5, 0.74, None))
        scorer._detect_task_type = lambda _prompt, _action="": "default"

        session = MagicMock()
        session.__len__.return_value = 1
        session.context_text = "earlier answer"
        session.update_contradictions.return_value = SimpleNamespace(
            contradiction_index=0.33,
            trend=0.44,
        )
        session.intent_drift.update.return_value = SimpleNamespace(
            drift_risk=0.55,
            triggered=True,
        )

        approved, score = scorer.review("prompt", "new answer", session=session)

        assert approved is True
        assert score.cross_turn_divergence == 0.4
        assert score.contradiction_index == 0.33
        assert score.intent_drift_risk == 0.55
        assert score.intent_drift_triggered is True
        session.intent_drift.update.assert_called_once_with(
            intent_divergence=0.4,
            injection_risk=0.0,
            contradiction_trend=0.44,
        )

    def test_review_continues_when_optional_session_and_injection_hooks_fail(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=True)
        nli = MagicMock()
        nli.model_available = True
        scorer._nli = nli
        scorer._heuristic_coherence = MagicMock(return_value=(0.1, 0.2, 0.86, None))
        scorer._detect_task_type = lambda _prompt, _action="": "default"

        session = MagicMock()
        session.__len__.return_value = 1
        session.context_text = ""
        session.update_contradictions.side_effect = RuntimeError("tracker down")

        detector = MagicMock()
        detector.detect.side_effect = RuntimeError("detector down")
        scorer._injection_detector = detector
        scorer._injection_fail_closed = False

        approved, score = scorer.review("prompt", "answer", session=session)

        assert approved is True
        assert score.injection_risk is None
        session.add_turn.assert_called_once_with("prompt", "answer", score.score)

    def test_review_injection_detector_fail_closed_raises(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=False)
        scorer._heuristic_coherence = MagicMock(return_value=(0.1, 0.2, 0.86, None))
        detector = MagicMock()
        detector.detect.side_effect = RuntimeError("detector down")
        scorer._injection_detector = detector
        scorer._injection_fail_closed = True

        import pytest

        with pytest.raises(RuntimeError, match="detector down"):
            scorer.review("prompt", "answer")

    def test_verified_scorer_fail_closed_on_insufficient_coverage(self, monkeypatch):
        from types import SimpleNamespace

        from director_ai.core.types import EvidenceChunk, ScoringEvidence

        class FakeVerifiedScorer:
            def __init__(self, nli_scorer):
                self.nli_scorer = nli_scorer

            def verify(self, response, source, *, atomic, evidence_top_k):
                assert response == "Atomic claim."
                assert source == "source text"
                assert atomic is True
                assert evidence_top_k == 3
                return SimpleNamespace(
                    approved=True,
                    coverage=0.25,
                    claims=["Atomic claim."],
                    contradicted_count=0,
                    fabricated_count=0,
                    to_dict=lambda: {"approved": True, "coverage": 0.25},
                )

        monkeypatch.setattr(
            "director_ai.core.scoring.verified_scorer.VerifiedScorer",
            FakeVerifiedScorer,
        )
        evidence = ScoringEvidence(
            chunks=[EvidenceChunk(text="source text", distance=0.1)],
            nli_premise="fallback source",
            nli_hypothesis="Atomic claim.",
            nli_score=0.2,
        )
        scorer = CoherenceScorer(use_nli=False)
        scorer._verified_scorer_enabled = True
        approved, score = scorer._finalise_review(
            0.7,
            0.1,
            0.2,
            "Atomic claim.",
            evidence,
            detected_task_type="rag",
        )

        verified_approved, verified_score = scorer._apply_verified_scorer(
            score,
            task_type="rag",
            threshold=0.5,
        )

        assert approved is True
        assert verified_approved is False
        assert verified_score.approved is False
        assert verified_score.verified_approved is False
        assert verified_score.verified_coverage == 0.25
        assert verified_score.verified_claim_count == 1

    def test_reasoning_tier_applies_structured_rejection(self):
        from types import SimpleNamespace

        from director_ai.core.types import CoherenceScore, ScoringEvidence

        scorer = CoherenceScorer(use_nli=False, reasoning_enabled=True)
        scorer._reasoning.should_escalate = lambda _score, centre=0.5: True
        scorer._reasoning.reason = lambda *args, **kwargs: SimpleNamespace(
            approved=False,
            confidence=0.91,
            rationale="unsupported safety claim",
            harm_category=SimpleNamespace(value="misinformation"),
            adjusted_score=0.22,
        )
        evidence = ScoringEvidence(
            chunks=[],
            nli_premise="source",
            nli_hypothesis="answer",
            nli_score=0.4,
        )
        score = CoherenceScore(
            score=0.52,
            approved=True,
            h_logical=0.3,
            h_factual=0.2,
            evidence=evidence,
            detected_task_type="rag",
        )

        approved, adjusted = scorer._apply_reasoning_tier(
            (True, score),
            "prompt",
            "answer",
            evidence,
            threshold=0.5,
        )

        assert approved is False
        assert adjusted.score == 0.22
        assert adjusted.reasoning_escalated is True
        assert adjusted.reasoning_confidence == 0.91
        assert adjusted.reasoning_harm_category == "misinformation"
        assert adjusted.reasoning_rationale == "unsupported safety claim"

    def test_reasoning_tier_keeps_lower_verdict_when_backend_returns_none(self):
        from director_ai.core.types import CoherenceScore

        scorer = CoherenceScorer(use_nli=False, reasoning_enabled=True)
        scorer._reasoning.should_escalate = lambda _score, centre=0.5: True
        scorer._reasoning.reason = lambda *args, **kwargs: None
        score = CoherenceScore(score=0.51, approved=True, h_logical=0.2, h_factual=0.3)

        assert scorer._apply_reasoning_tier(
            (True, score),
            "prompt",
            "answer",
            None,
            threshold=0.5,
        ) == (True, score)

    def test_dialogue_and_summarization_routes_use_specialised_divergence(self):
        from director_ai.core.types import ScoringEvidence

        scorer = CoherenceScorer(use_nli=True)
        nli = type(
            "ReadyNLI",
            (),
            {"model_available": True, "_ensure_model": lambda self: True},
        )()
        scorer._nli = nli
        scorer._detect_task_type = lambda _prompt, _action="": "dialogue"
        scorer._dialogue_factual_divergence = lambda *_args: (0.24, None)

        h_logic, h_fact, coherence, evidence = scorer._heuristic_coherence(
            "User: hi",
            "Assistant: hi",
        )

        assert h_logic == 0.0
        assert h_fact == 0.24
        assert 0.0 <= coherence <= 1.0
        assert evidence is None

        summary_evidence = ScoringEvidence(
            chunks=[],
            nli_premise="source",
            nli_hypothesis="summary",
            nli_score=0.18,
        )
        scorer._detect_task_type = lambda _prompt, _action="": "summarization"
        scorer._summarization_factual_divergence = lambda *_args: (
            0.18,
            summary_evidence,
        )

        h_logic, h_fact, _coherence, evidence = scorer._heuristic_coherence(
            "source",
            "summary",
        )

        assert h_logic == 0.0
        assert h_fact == 0.18
        assert evidence is summary_evidence

    def test_prompt_premise_evidence_returns_neutral_without_model(self):
        scorer = CoherenceScorer(use_nli=False)

        divergence, evidence = (
            scorer._calculate_prompt_premise_divergence_with_evidence(
                "source",
                "summary",
            )
        )

        assert divergence == 0.5
        assert evidence is None

    def test_factual_divergence_retrieves_when_adaptive_router_allows_it(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        store = MagicMock()
        store.retrieve_context.return_value = "verified context"
        router = MagicMock()
        router.should_retrieve.return_value = SimpleNamespace(
            retrieve=True,
            task_type="rag",
            confidence=0.88,
        )
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._adaptive_router = router

        assert scorer.calculate_factual_divergence("prompt", "verified answer") == 0.5
        store.retrieve_context.assert_called_once_with("prompt", top_k=3, tenant_id="")

    def test_vector_retrieval_abstention_continues_for_close_match(self):
        from unittest.mock import MagicMock

        from director_ai.core.retrieval.vector_store import VectorGroundTruthStore
        from director_ai.core.types import EvidenceChunk

        store = VectorGroundTruthStore()
        store.retrieve_context = MagicMock(return_value="verified context")
        store.retrieve_context_with_chunks = MagicMock(
            return_value=[EvidenceChunk(text="verified context", distance=0.1)]
        )
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        scorer._retrieval_abstention_threshold = 0.4

        assert scorer.calculate_factual_divergence("prompt", "verified context") == 0.0
        store.retrieve_context_with_chunks.assert_called_once_with(
            "prompt",
            top_k=3,
            tenant_id="",
        )

    def test_review_uses_adaptive_threshold_and_meta_classifier(self):
        from unittest.mock import MagicMock

        meta_classifier = MagicMock()
        meta_classifier.predict_threshold.return_value = (0.25, 0.9)
        scorer = CoherenceScorer(use_nli=False, threshold=0.8)
        scorer._adaptive_threshold_enabled = True
        scorer._task_type_thresholds = {"rag": 0.7}
        scorer._get_meta_classifier = MagicMock(return_value=meta_classifier)
        scorer._heuristic_coherence = MagicMock(return_value=(0.1, 0.2, 0.6, None))
        scorer._detect_task_type = lambda _prompt, _action="": "rag"

        approved, score = scorer.review("prompt", "answer")

        assert approved is True
        assert score.detected_task_type == "rag"
        meta_classifier.predict_threshold.assert_called_once_with("prompt", "answer")

    def test_review_batch_empty_and_sequential_fallback_paths(self):
        from unittest.mock import MagicMock

        scorer = CoherenceScorer(use_nli=False)
        assert scorer.review_batch([]) == []

        scorer.review = MagicMock(return_value=(True, object()))
        assert scorer.review_batch([("prompt", "answer")]) == [
            (True, scorer.review.return_value[1])
        ]
        scorer.review.assert_called_once_with("prompt", "answer", tenant_id="")

        scorer = CoherenceScorer(use_nli=True)
        scorer._nli = MagicMock(model_available=True)
        scorer._review_batch_requires_sequential = MagicMock(return_value=True)
        scorer.review = MagicMock(return_value=(True, object()))

        assert scorer.review_batch([("p1", "a1"), ("p2", "a2")]) == [
            scorer.review.return_value,
            scorer.review.return_value,
        ]
        assert scorer.review.call_count == 2

    def test_review_batch_applies_adaptive_and_meta_thresholds(self):
        from unittest.mock import MagicMock

        store = MagicMock()
        store.retrieve_context.return_value = ""
        scorer = CoherenceScorer(use_nli=True, threshold=0.9, ground_truth_store=store)
        scorer._adaptive_threshold_enabled = True
        scorer._task_type_thresholds = {"rag": 0.8}
        scorer._detect_task_type = lambda _prompt, _action="": "rag"
        meta_classifier = MagicMock()
        meta_classifier.predict_threshold.return_value = (0.1, 0.95)
        scorer._get_meta_classifier = MagicMock(return_value=meta_classifier)

        nli = MagicMock()
        nli.model_available = True
        nli.score_batch.return_value = [0.2, 0.3]
        scorer._nli = nli

        results = scorer.review_batch([("p1", "a1"), ("p2", "a2")])

        assert [approved for approved, _score in results] == [True, True]
        assert meta_classifier.predict_threshold.call_count == 2
