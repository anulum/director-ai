# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Scorer Tests
"""Multi-angle tests for NLI scorer core pipeline.

Covers: heuristic scoring, score range invariants, batch scoring,
empty/long inputs, ONNX fallback, invalid backend guard, determinism,
pipeline integration with CoherenceScorer, and performance documentation.
"""

import pytest

from director_ai.core.nli import NLIScorer


@pytest.mark.consumer
class TestNLIScorer:
    def test_heuristic_fallback_consistent(self):
        scorer = NLIScorer(use_model=False)
        h = scorer.score("test", "This is consistent with reality.")
        assert h == pytest.approx(0.1)

    def test_heuristic_fallback_contradiction(self):
        scorer = NLIScorer(use_model=False)
        h = scorer.score("test", "The opposite is true.")
        assert h == pytest.approx(0.9)

    def test_heuristic_fallback_neutral(self):
        scorer = NLIScorer(use_model=False)
        h = scorer.score("test", "The answer depends on your perspective.")
        assert h == pytest.approx(0.5)

    def test_heuristic_fallback_overlap(self):
        scorer = NLIScorer(use_model=False)
        h = scorer.score("The sky is blue", "The sky is blue and clear")
        assert 0.0 <= h <= 1.0

    def test_model_available_is_false_without_model(self):
        scorer = NLIScorer(use_model=False)
        assert scorer.model_available is False

    def test_score_batch(self):
        scorer = NLIScorer(use_model=False)
        pairs = [
            ("test", "consistent with reality"),
            ("test", "opposite is true"),
        ]
        results = scorer.score_batch(pairs)
        assert len(results) == 2
        assert results[0] < results[1]

    def test_score_batch_empty(self):
        scorer = NLIScorer(use_model=False)
        assert scorer.score_batch([]) == []

    def test_score_batch_matches_sequential(self):
        scorer = NLIScorer(use_model=False)
        pairs = [
            ("sky is blue", "consistent with reality"),
            ("earth is round", "opposite is true"),
            ("cats meow", "depends on your perspective"),
            ("water is wet", "random unrelated text"),
        ]
        batch = scorer.score_batch(pairs)
        sequential = [scorer.score(p, h) for p, h in pairs]
        assert batch == sequential

    def test_score_range(self):
        scorer = NLIScorer(use_model=False)
        for text in ["hello world", "anything", "random noise xyz"]:
            h = scorer.score("test prompt", text)
            assert 0.0 <= h <= 1.0

    def test_onnx_backend_without_path_falls_back(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        s = scorer.score("premise", "hypothesis")
        assert 0.0 <= s <= 1.0

    def test_onnx_backend_invalid_path_falls_back(self, tmp_path):
        scorer = NLIScorer(
            use_model=True,
            backend="onnx",
            onnx_path=str(tmp_path / "no_such_dir_xyz"),
        )
        s = scorer.score("premise", "hypothesis")
        assert 0.0 <= s <= 1.0

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            NLIScorer(backend="invalid")


class TestNLIClaimSupportScoring:
    """Behavioural tests for claim-level support scoring."""

    @staticmethod
    def _scorer_with_claim_scores(scores: dict[str, float]) -> NLIScorer:
        scorer = NLIScorer(use_model=False)

        def score_chunked(_source: str, claim: str, **_kwargs):
            return scores[claim], []

        def decompose_claims(text: str) -> list[str]:
            if not text:
                return []
            claims = [sentence.strip() for sentence in text.split(". ") if sentence.strip()]
            return [claim if claim.endswith(".") else claim + "." for claim in claims]

        scorer.score_chunked = score_chunked  # type: ignore[method-assign]
        scorer.decompose_claims = decompose_claims  # type: ignore[method-assign]
        return scorer

    def test_claim_support_fraction_uses_each_decomposed_statement(self, monkeypatch):
        monkeypatch.setattr("director_ai.core.scoring.nli._RUST_NLI", False)
        scorer = self._scorer_with_claim_scores(
            {
                "The sky is blue.": 0.1,
                "Water is wet.": 0.2,
                "Mars is flat.": 0.9,
            },
        )

        support, divergences, claims = scorer.score_claim_coverage(
            "Reference text discusses the sky and water.",
            "The sky is blue. Water is wet. Mars is flat.",
        )

        assert claims == ["The sky is blue.", "Water is wet.", "Mars is flat."]
        assert divergences == [0.1, 0.2, 0.9]
        assert support == pytest.approx(2.0 / 3.0)

    def test_claim_support_threshold_boundary_is_strictly_less_than_threshold(self, monkeypatch):
        monkeypatch.setattr("director_ai.core.scoring.nli._RUST_NLI", False)
        scorer = self._scorer_with_claim_scores({"Claim X.": 0.35})

        loose_support, _, _ = scorer.score_claim_coverage(
            "Reference.",
            "Claim X.",
            support_threshold=0.5,
        )
        strict_support, _, _ = scorer.score_claim_coverage(
            "Reference.",
            "Claim X.",
            support_threshold=0.3,
        )

        assert loose_support == 1.0
        assert strict_support == 0.0

    def test_empty_summary_falls_back_to_single_score_result(self):
        scorer = NLIScorer(use_model=False)
        scorer.decompose_claims = lambda _text: []  # type: ignore[method-assign]

        support, divergences, claims = scorer.score_claim_coverage("Reference.", "")

        assert support in (0.0, 1.0)
        assert len(divergences) == 1
        assert claims == [""]

    @pytest.mark.parametrize("failure", [RuntimeError, ValueError, TypeError])
    def test_mandatory_rust_claim_reducer_failures_are_not_hidden(
        self,
        monkeypatch,
        failure,
    ):
        scorer = self._scorer_with_claim_scores(
            {
                "Claim A.": 0.2,
                "Claim B.": 0.8,
            },
        )

        def unavailable_reducer(_divergences, _threshold):
            raise failure("ffi unavailable")

        monkeypatch.setattr("director_ai.core.scoring.nli._RUST_NLI", True)
        monkeypatch.setattr(
            "director_ai.core.scoring.nli.rust_coverage_from_divergences",
            unavailable_reducer,
        )

        with pytest.raises(failure, match="ffi unavailable"):
            scorer.score_claim_coverage(
                "Reference.",
                "Claim A. Claim B.",
                support_threshold=0.6,
            )


class TestNLIModelLoadingSafety:
    """NLI model-loading contracts for local ONNX and quantized model paths."""

    @staticmethod
    def _mock_runtime_modules():
        from unittest.mock import MagicMock

        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.InferenceSession.return_value = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
        return mock_ort, mock_transformers

    def test_onnx_loader_rejects_directory_outside_allowed_roots(self, tmp_path):
        import os
        import sys
        from unittest.mock import patch

        allowed = tmp_path / "allowed"
        outside = tmp_path / "outside"
        allowed.mkdir()
        outside.mkdir()
        (outside / "model.onnx").write_bytes(b"fake")
        mock_ort, mock_transformers = self._mock_runtime_modules()

        with (
            patch.dict(
                sys.modules,
                {"onnxruntime": mock_ort, "transformers": mock_transformers},
            ),
            patch.dict(os.environ, {"DIRECTOR_ONNX_ALLOWED_DIRS": str(allowed)}),
        ):
            from director_ai.core.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            tokenizer, session = _load_onnx_session(str(outside))

        assert tokenizer is None
        assert session is None
        mock_ort.InferenceSession.assert_not_called()

    def test_onnx_loader_rejects_model_file_symlink_escape(self, tmp_path):
        import os
        import sys
        from unittest.mock import patch

        allowed = tmp_path / "allowed"
        external = tmp_path / "external"
        bundle = allowed / "bundle"
        allowed.mkdir()
        external.mkdir()
        bundle.mkdir()
        (external / "model.onnx").write_bytes(b"fake")
        (bundle / "model.onnx").symlink_to(external / "model.onnx")
        mock_ort, mock_transformers = self._mock_runtime_modules()

        with (
            patch.dict(
                sys.modules,
                {"onnxruntime": mock_ort, "transformers": mock_transformers},
            ),
            patch.dict(os.environ, {"DIRECTOR_ONNX_ALLOWED_DIRS": str(allowed)}),
        ):
            from director_ai.core.nli import _load_onnx_session

            _load_onnx_session.cache_clear()
            tokenizer, session = _load_onnx_session(str(bundle))

        assert tokenizer is None
        assert session is None
        mock_ort.InferenceSession.assert_not_called()

    def test_quantized_model_load_falls_back_when_bitsandbytes_is_unavailable(self):
        import sys
        from unittest.mock import MagicMock, patch

        mock_torch = MagicMock()
        mock_torch.float16 = "fp16"
        mock_torch.bfloat16 = "bf16"
        mock_torch.float32 = "fp32"
        mock_transformers = MagicMock()
        mock_transformers.BitsAndBytesConfig = MagicMock(
            side_effect=ImportError("no bitsandbytes"),
        )
        tokenizer = MagicMock()
        model = MagicMock()
        model.to.return_value = model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        mock_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = model

        with patch.dict(
            sys.modules,
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            from director_ai.core.nli import _load_nli_model

            _load_nli_model.cache_clear()
            loaded_tokenizer, loaded_model = _load_nli_model(
                "quant-model-nobnb",
                quantize_8bit=True,
            )

        assert loaded_tokenizer is tokenizer
        assert loaded_model is model
