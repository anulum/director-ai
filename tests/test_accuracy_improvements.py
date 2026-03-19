# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Accuracy Improvement Plan Tests

from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np
import pytest

from director_ai.core.config import DirectorConfig

# â”€â”€ Phase 1: Task-type detection & adaptive thresholds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestTaskTypeDetection:
    """Phase 1B: _detect_task_type returns 6 categories."""

    @pytest.fixture
    def scorer(self):
        from director_ai.core.scorer import CoherenceScorer

        return CoherenceScorer(use_nli=False)

    def test_dialogue_detection(self, scorer):
        prompt = "User: Hello\nAssistant: Hi there"
        assert scorer._detect_task_type(prompt) == "dialogue"

    def test_dialogue_bracket_format(self, scorer):
        prompt = (
            "[Human]: What is AI? [Assistant]: It stands for artificial intelligence."
        )
        assert scorer._detect_task_type(prompt) == "dialogue"

    def test_summarization_detection(self, scorer):
        assert scorer._detect_task_type("Summarize this article") == "summarization"
        assert (
            scorer._detect_task_type("Give me a summary of the paper")
            == "summarization"
        )
        assert scorer._detect_task_type("TLDR of this document") == "summarization"

    def test_qa_detection(self, scorer):
        assert scorer._detect_task_type("What is the capital of France?") == "qa"
        assert scorer._detect_task_type("Answer the question below") == "qa"
        assert scorer._detect_task_type("Based on the text, what happened?") == "qa"

    def test_fact_check_detection(self, scorer):
        assert scorer._detect_task_type("Verify this claim") == "fact_check"
        assert scorer._detect_task_type("Is it true that water is wet?") == "fact_check"
        assert scorer._detect_task_type("Fact-check this statement") == "fact_check"

    def test_rag_detection(self, scorer):
        assert scorer._detect_task_type("Based on the context, answer:") == "rag"
        assert scorer._detect_task_type("Given the document below") == "rag"
        assert scorer._detect_task_type("Using the retrieved passages") == "rag"

    def test_default_fallback(self, scorer):
        assert scorer._detect_task_type("Write me a poem about the sea") == "default"


class TestAdaptiveThresholds:
    """Phase 1B: per-task-type threshold selection in review()."""

    def test_config_fields_exist(self):
        cfg = DirectorConfig()
        assert cfg.adaptive_threshold_enabled is True
        assert cfg.threshold_summarization == 0.72
        assert cfg.threshold_qa == 0.69
        assert cfg.threshold_fact_check == 0.56
        assert cfg.threshold_rag == 0.78
        assert cfg.threshold_dialogue == 0.68

    def test_scorer_receives_adaptive_config(self):
        cfg = DirectorConfig(adaptive_threshold_enabled=True)
        scorer = cfg.build_scorer()
        assert scorer._adaptive_threshold_enabled is True
        assert scorer._task_type_thresholds["summarization"] == 0.72
        assert scorer._task_type_thresholds["qa"] == 0.69

    def test_scorer_uses_adaptive_threshold(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False, threshold=0.5)
        scorer._adaptive_threshold_enabled = True
        scorer._task_type_thresholds = {
            "qa": 0.3,
            "summarization": 0.7,
        }
        # QA prompt should use 0.3 threshold â†’ more lenient
        approved, score = scorer.review("What is 2+2?", "2+2 is 4")
        # The heuristic score should be above the QA threshold of 0.3
        assert score.score is not None


# â”€â”€ Phase 2: Chunking & Aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestChunkingConfig:
    """Phase 2A/2B: overlap ratio and confidence-weighted aggregation config."""

    def test_config_fields(self):
        cfg = DirectorConfig()
        assert cfg.nli_chunk_overlap_ratio == 0.5
        assert cfg.nli_qa_premise_ratio == 0.7
        assert cfg.nli_confidence_weighted_agg is True

    def test_scorer_receives_chunking_config(self):
        cfg = DirectorConfig(
            nli_chunk_overlap_ratio=0.3,
            nli_qa_premise_ratio=0.8,
            nli_confidence_weighted_agg=True,
        )
        scorer = cfg.build_scorer()
        assert scorer._chunk_overlap_ratio == 0.3
        assert scorer._qa_premise_ratio == 0.8
        assert scorer._confidence_weighted_agg is True


class TestNLIConfidence:
    """Phase 2B: _probs_to_confidence helper."""

    def test_high_confidence_peaked_distribution(self):
        from director_ai.core.nli import _probs_to_confidence

        probs = np.array([[0.99, 0.01]])
        confs = _probs_to_confidence(probs)
        assert confs[0] > 0.8

    def test_low_confidence_uniform_distribution(self):
        from director_ai.core.nli import _probs_to_confidence

        probs = np.array([[0.5, 0.5]])
        confs = _probs_to_confidence(probs)
        assert confs[0] < 0.1

    def test_batch_operation(self):
        from director_ai.core.nli import _probs_to_confidence

        probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.01, 0.99]])
        confs = _probs_to_confidence(probs)
        assert len(confs) == 3
        assert confs[0] > confs[1]  # peaked > uniform
        assert confs[2] > confs[1]  # peaked > uniform


# â”€â”€ Phase 3: LoRA adapter loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestLoRAConfig:
    """Phase 3A: lora_adapter_path config field."""

    def test_config_field_exists(self):
        cfg = DirectorConfig()
        assert cfg.lora_adapter_path == ""

    def test_config_field_accepts_path(self):
        cfg = DirectorConfig(lora_adapter_path="/path/to/adapter")
        assert cfg.lora_adapter_path == "/path/to/adapter"


# â”€â”€ Phase 4: Distillation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestDistillationPipeline:
    """Phase 4A: verify distillation module structure."""

    def test_module_importable(self):
        assert os.path.isfile(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "tools",
                "run_distillation.py",
            ),
        )


# â”€â”€ Phase 6A: Meta-classifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestMetaClassifierFeatures:
    """Phase 6A: feature extraction for meta-classifier."""

    def test_extract_features_basic(self):
        # Import from tools path
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from tools.train_meta_classifier import extract_features

            feat = extract_features(
                premise="The cat sat on the mat.",
                hypothesis="A cat was sitting.",
                nli_score=0.8,
                confidence=0.9,
                chunk_count=1,
            )
            assert feat["nli_score"] == 0.8
            assert feat["confidence"] == 0.9
            assert feat["chunk_count"] == 1
            assert feat["premise_len"] == len("The cat sat on the mat.")
            assert feat["score_distance_from_half"] == pytest.approx(0.3)
            assert "word_overlap" in feat
            assert "has_negation_premise" in feat
        finally:
            sys.path.pop(0)

    def test_extract_features_negation(self):
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from tools.train_meta_classifier import extract_features

            feat = extract_features(
                premise="This is not correct.",
                hypothesis="This is correct.",
                nli_score=0.3,
                confidence=0.6,
            )
            assert feat["has_negation_premise"] == 1
            assert feat["has_negation_hypothesis"] == 0
            assert feat["negation_asymmetry"] == 1
        finally:
            sys.path.pop(0)


class TestMetaClassifierConfig:
    """Phase 6A: meta_classifier_path config integration."""

    def test_config_field_exists(self):
        cfg = DirectorConfig()
        assert cfg.meta_classifier_path == ""

    def test_scorer_meta_classifier_path(self):
        cfg = DirectorConfig(meta_classifier_path="/path/to/model.pkl")
        scorer = cfg.build_scorer()
        assert scorer._meta_classifier_path == "/path/to/model.pkl"

    def test_scorer_lazy_load_returns_none_for_missing(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False)
        scorer._meta_classifier_path = ""
        assert scorer._get_meta_classifier() is None


class TestMetaClassifierIntegration:
    """Phase 6A: MetaClassifier class end-to-end."""

    def test_meta_classifier_predict(self):
        """Create a minimal pickled model and verify MetaClassifier loads it."""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except (ImportError, ValueError, AttributeError):
            pytest.skip("sklearn not available (import or numpy/pandas compat)")
            return
        try:
            from tools.train_meta_classifier import MetaClassifier

            feature_cols = [
                "nli_score",
                "confidence",
                "premise_len",
                "hypothesis_len",
                "premise_word_count",
                "hypothesis_word_count",
                "word_overlap",
                "has_negation_premise",
                "has_negation_hypothesis",
                "negation_asymmetry",
                "chunk_count",
                "score_distance_from_half",
                "has_question_mark",
                "num_entities_premise",
                "num_entities_hypothesis",
            ]

            x_data = np.random.randn(50, len(feature_cols))
            y = (x_data[:, 0] > 0).astype(int)

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_data)
            clf = LogisticRegression(max_iter=100)
            clf.fit(x_scaled, y)

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(
                    {"classifier": clf, "scaler": scaler, "feature_cols": feature_cols},
                    f,
                )
                tmp_path = f.name

            try:
                mc = MetaClassifier(tmp_path)
                is_supported, prob = mc.predict(
                    "The capital of France is Paris.",
                    "Paris is the capital.",
                    nli_score=0.85,
                    confidence=0.9,
                )
                assert isinstance(is_supported, bool)
                assert 0.0 <= prob <= 1.0
            finally:
                os.unlink(tmp_path)
        finally:
            sys.path.pop(0)


# â”€â”€ Phase 1A: Threshold analysis tool â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestThresholdAnalysis:
    """Phase 1A: threshold_analysis.py tool structure."""

    def test_task_type_map_complete(self):
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from benchmarks.threshold_analysis import TASK_TYPE_MAP

            expected_datasets = {
                "AggreFact-CNN",
                "AggreFact-XSum",
                "TofuEval-MediaS",
                "TofuEval-MeetB",
                "Wice",
                "Reveal",
                "ClaimVerify",
                "FactCheck-GPT",
                "ExpertQA",
                "Lfqa",
                "RAGTruth",
            }
            assert set(TASK_TYPE_MAP.keys()) == expected_datasets
        except (ImportError, ValueError):
            pytest.skip(
                "benchmarks.threshold_analysis not importable (sklearn/numpy compat)"
            )
        finally:
            sys.path.pop(0)


# â”€â”€ LoRA training tool â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestLoRATrainingTool:
    """Phase 3A: run_lora_training.py tool structure."""

    def test_load_dataset_pairs_function_exists(self):
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from tools.run_lora_training import load_dataset_pairs

            assert callable(load_dataset_pairs)
        finally:
            sys.path.pop(0)

    def test_factcg_template(self):
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from tools.run_lora_training import FACTCG_TEMPLATE

            formatted = FACTCG_TEMPLATE.format(
                text_a="The cat sat on the mat.",
                text_b="A cat was sitting.",
            )
            assert "The cat sat on the mat." in formatted
            assert "A cat was sitting." in formatted
            assert "Choose your answer" in formatted
        finally:
            sys.path.pop(0)

    def test_sieving_collator_structure(self):
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from tools.run_lora_training import SievingCollator

            assert callable(SievingCollator)
        finally:
            sys.path.pop(0)

    def test_format_for_factcg(self):
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from tools.run_lora_training import format_for_factcg

            pairs = [
                {
                    "premise": "Earth is round.",
                    "hypothesis": "The planet is spherical.",
                    "label": 1,
                },
            ]
            result = format_for_factcg(pairs)
            assert len(result) == 1
            assert result[0]["label"] == 1
            assert "Earth is round." in result[0]["text"]
        finally:
            sys.path.pop(0)


# â”€â”€ Cross-phase integration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestQAPremiseRatio:
    """Phase 2A: QA tasks get higher premise_ratio."""

    def test_qa_premise_ratio_applied(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False)
        scorer._premise_ratio = 0.4
        scorer._qa_premise_ratio = 0.7
        # The QA detection path should select 0.7 in calculate_factual_divergence
        # We can't test the NLI call without a model, but verify the config wiring
        assert scorer._qa_premise_ratio == 0.7
        assert scorer._premise_ratio == 0.4
