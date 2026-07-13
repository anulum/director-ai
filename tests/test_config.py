# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Configuration Manager Tests
"""Multi-angle tests for DirectorConfig pipeline configuration.

Covers: defaults, profile loading, env var override, YAML/JSON config files,
threshold validation, CORS, audit log, tenant routing, logging, NLI settings,
pipeline integration with CoherenceScorer/Agent/Server, and performance.
"""

import json
import logging
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

from director_ai.core.config import DirectorConfig


class TestDirectorConfig:
    """Tests for DirectorConfig dataclass."""

    def test_default_values(self):
        cfg = DirectorConfig()
        assert cfg.coherence_threshold == 0.6
        assert cfg.hard_limit == 0.5
        assert cfg.use_nli is True
        assert cfg.max_candidates == 3
        assert cfg.llm_provider == "mock"
        assert cfg.server_port == 8080
        assert cfg.batch_max_concurrency == 4
        assert cfg.metrics_enabled is True
        assert cfg.profile == "default"

    def test_custom_values(self):
        cfg = DirectorConfig(coherence_threshold=0.8, use_nli=True, server_port=9090)
        assert cfg.coherence_threshold == 0.8
        assert cfg.use_nli is True
        assert cfg.server_port == 9090

    def test_to_dict(self):
        cfg = DirectorConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["coherence_threshold"] == 0.6
        assert d["profile"] == "default"

    def test_server_host_defaults_to_loopback(self):
        # Secure default: a direct embedder that binds this value is not exposed
        # on all interfaces unless it opts in.
        assert DirectorConfig().server_host == "127.0.0.1"

    def test_to_dict_redacts_api_key(self):
        cfg = DirectorConfig(llm_api_key="sk-secret-123")
        d = cfg.to_dict()
        assert d["llm_api_key"] == "***"

    def test_to_dict_empty_key_not_redacted(self):
        cfg = DirectorConfig(llm_api_key="")
        d = cfg.to_dict()
        assert d["llm_api_key"] == ""

    def test_to_dict_redacts_license_secrets(self):
        cfg = DirectorConfig(
            license_key="LIC-SECRET-XYZ",
            license_file="/etc/director/license.key",
        )
        d = cfg.to_dict()
        assert d["license_key"] == "***"
        assert d["license_file"] == "***"

    def test_to_dict_empty_license_fields_not_redacted(self):
        cfg = DirectorConfig(license_key="", license_file="")
        d = cfg.to_dict()
        assert d["license_key"] == ""
        assert d["license_file"] == ""


class TestApiKeysEnvParsing:
    def test_comma_separated(self):
        from director_ai.core.config import _parse_api_keys_env

        assert _parse_api_keys_env("sk-a,sk-b") == ["sk-a", "sk-b"]

    def test_json_array(self):
        from director_ai.core.config import _parse_api_keys_env

        assert _parse_api_keys_env('["sk-a","sk-b"]') == ["sk-a", "sk-b"]

    def test_json_array_does_not_embed_brackets(self):
        from director_ai.core.config import _parse_api_keys_env

        # The footgun: a JSON array must not produce a literal key that still
        # carries brackets and quotes.
        keys = _parse_api_keys_env('["sk-prod-xxx"]')
        assert keys == ["sk-prod-xxx"]
        assert "[" not in keys[0] and '"' not in keys[0]

    def test_whitespace_and_blanks_dropped(self):
        from director_ai.core.config import _parse_api_keys_env

        assert _parse_api_keys_env(" sk-a , , sk-b ") == ["sk-a", "sk-b"]
        assert _parse_api_keys_env('[" sk-a ", "", "sk-b"]') == ["sk-a", "sk-b"]

    def test_empty_returns_empty_list(self):
        from director_ai.core.config import _parse_api_keys_env

        assert _parse_api_keys_env("") == []
        assert _parse_api_keys_env("   ") == []

    def test_malformed_json_falls_back_to_comma(self):
        from director_ai.core.config import _parse_api_keys_env

        # A bracketed-but-invalid value degrades to comma splitting rather than
        # silently dropping the keys.
        assert _parse_api_keys_env("[sk-a,sk-b") == ["[sk-a", "sk-b"]

    def test_json_non_list_falls_back_to_comma(self):
        from director_ai.core.config import _parse_api_keys_env

        # A JSON object is not a key list; fall back to comma semantics.
        assert _parse_api_keys_env('{"k":"v"}') == ['{"k":"v"}']

    def test_production_mode_rejects_mock_llm_provider(self):
        with pytest.raises(ValueError, match="production_mode requires a real LLM"):
            DirectorConfig(production_mode=True, api_keys={"tenant-api-key"})


class TestProfileLoading:
    """Tests for from_profile()."""

    def test_fast_profile(self):
        cfg = DirectorConfig.from_profile("fast")
        assert cfg.profile == "fast"
        assert cfg.use_nli is False
        assert cfg.max_candidates == 1
        assert cfg.metrics_enabled is False

    def test_thorough_profile(self):
        cfg = DirectorConfig.from_profile("thorough")
        assert cfg.profile == "thorough"
        assert cfg.use_nli is True
        assert cfg.max_candidates == 3

    def test_research_profile(self):
        cfg = DirectorConfig.from_profile("research")
        assert cfg.profile == "research"
        assert cfg.use_nli is True
        assert cfg.max_candidates == 5
        assert cfg.coherence_threshold == 0.7

    def test_production_profile_requires_api_keys_from_env(self, monkeypatch):
        # No hard-coded key: without env-injected secrets, production fails closed.
        monkeypatch.delenv("DIRECTOR_API_KEYS", raising=False)
        monkeypatch.delenv("DIRECTOR_API_KEY_TENANT_MAP", raising=False)
        with pytest.raises(
            ValueError, match="production_mode requires api_keys or api_key_tenant_map"
        ):
            DirectorConfig.from_profile("production")

    def test_production_profile_loads_with_env_tenant_map(self, monkeypatch):
        monkeypatch.delenv("DIRECTOR_API_KEYS", raising=False)
        monkeypatch.setenv(
            "DIRECTOR_API_KEY_TENANT_MAP", '{"real-prod-key":"tenant-default"}'
        )
        monkeypatch.setenv(
            "DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS",
            '{"kid-1":"prod-signing-secret-at-least-32-chars-xx"}',
        )
        cfg = DirectorConfig.from_profile("production")
        assert "real-prod-key" in cfg.api_key_tenant_map
        assert "director-production-local-validation-key" not in cfg.api_key_tenant_map

    def test_production_profile_is_fail_closed_and_observable(self, monkeypatch):
        monkeypatch.setenv(
            "DIRECTOR_API_KEY_TENANT_MAP", '{"real-prod-key":"tenant-default"}'
        )
        monkeypatch.setenv(
            "DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS",
            '{"kid-1":"prod-signing-secret-at-least-32-chars-xx"}',
        )
        cfg = DirectorConfig.from_profile("production")

        assert cfg.profile == "production"
        assert cfg.production_mode is True
        assert cfg.mode == "grounded"
        assert cfg.use_nli is True
        assert cfg.coherence_require_model_backed_nli is True
        assert cfg.adaptive_threshold_fail_closed is True
        assert cfg.injection_detection_enabled is True
        assert cfg.injection_fail_closed_on_error is True
        assert cfg.tenant_routing is True
        assert cfg.metrics_enabled is True
        assert cfg.metrics_require_auth is True
        assert cfg.rate_limit_rpm == 120
        assert cfg.review_queue_enabled is True
        assert cfg.audit_log_path
        assert cfg.compliance_db_path
        assert cfg.feedback_db_path
        assert cfg.stats_backend == "sqlite"
        assert cfg.log_json is True
        assert cfg.otel_enabled is True

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            DirectorConfig.from_profile("nonexistent")

    @pytest.mark.parametrize(
        "name",
        ["medical", "finance", "legal", "summarization", "research"],
    )
    def test_high_stakes_profiles_use_hybrid(self, name):
        cfg = DirectorConfig.from_profile(name)
        assert cfg.scorer_backend == "hybrid"
        assert cfg.llm_judge_enabled is True

    @pytest.mark.parametrize(
        "name,threshold,hard,soft,nli,reranker,wl,wf",
        [
            ("medical", 0.30, 0.20, 0.35, True, True, 0.5, 0.5),
            ("finance", 0.30, 0.20, 0.35, True, True, 0.4, 0.6),
            ("legal", 0.30, 0.20, 0.35, True, False, 0.6, 0.4),
            ("creative", 0.40, 0.30, 0.45, False, False, 0.7, 0.3),
            ("customer_support", 0.55, 0.40, 0.60, False, False, 0.5, 0.5),
            ("summarization", 0.15, 0.08, 0.25, True, False, 0.0, 1.0),
        ],
    )
    def test_domain_profile(self, name, threshold, hard, soft, nli, reranker, wl, wf):
        cfg = DirectorConfig.from_profile(name)
        assert cfg.profile == name
        assert cfg.coherence_threshold == pytest.approx(threshold)
        assert cfg.hard_limit == pytest.approx(hard)
        assert cfg.soft_limit == pytest.approx(soft)
        assert cfg.use_nli is nli
        assert cfg.reranker_enabled is reranker
        assert cfg.w_logic == pytest.approx(wl)
        assert cfg.w_fact == pytest.approx(wf)

    def test_profile_metadata_contains_operational_fields(self):
        meta = DirectorConfig.profile_metadata("medical")

        assert meta.name == "medical"
        assert meta.required_dependencies == ("nli", "vector")
        assert "medical" in meta.intended_workload.lower()
        assert meta.validation_status
        assert meta.expected_false_halt_risk
        assert meta.calibration_required is True
        assert meta.min_calibration_samples >= 20
        assert "director-ai tune" in meta.calibration_command

    def test_regulated_profile_metadata_matches_validation_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        medical = json.loads(
            (root / "benchmarks/results/medical_eval.json").read_text(encoding="utf-8")
        )["pubmedqa"]
        finance = json.loads(
            (root / "benchmarks/results/finance_eval.json").read_text(encoding="utf-8")
        )["financebench"]
        legal = json.loads(
            (root / "benchmarks/results/legal_eval.json").read_text(encoding="utf-8")
        )["cuad"]

        assert medical["fpr"] == pytest.approx(1.0)
        assert "FPR=1.0" in DirectorConfig.profile_metadata("medical").validation_status
        assert finance["fpr"] == pytest.approx(1.0)
        assert "FPR=1.0" in DirectorConfig.profile_metadata("finance").validation_status
        assert legal["fpr"] == pytest.approx(1.0)
        assert "FPR=1.0" in DirectorConfig.profile_metadata("legal").validation_status

    def test_public_docs_do_not_claim_stock_regulated_profiles_are_measured(self):
        root = Path(__file__).resolve().parents[1]
        docs = "\n".join(
            [
                (root / "docs-site/api/config.md").read_text(encoding="utf-8"),
                (root / "docs-site/benchmarks.md").read_text(encoding="utf-8"),
                (root / "docs-site/guide/why-director-ai.md").read_text(
                    encoding="utf-8"
                ),
                (root / "docs-site/deployment/runbooks.md").read_text(encoding="utf-8"),
                (root / "docs-site/glossary.md").read_text(encoding="utf-8"),
            ]
        )

        stale_claims = (
            "Healthcare (measured on PubMedQA)",
            "Financial services (measured on FinanceBench)",
            "measured profile (threshold 0.30)",
            "0% FPR at t",
            "77.3% / 66.2% / 59.9%",
            "medical/finance/legal (measured profiles)",
        )
        for claim in stale_claims:
            assert claim not in docs
        assert "calibration required" in docs
        assert "100.0% FPR" in docs

    def test_profile_metadata_serializes_dependencies_as_list(self):
        data = DirectorConfig.profile_metadata("summarization").to_dict()

        assert data["name"] == "summarization"
        assert data["required_dependencies"] == ["nli"]
        assert data["validation_status"]
        assert data["calibration_required"] is True
        assert data["min_calibration_samples"] >= 20
        assert "director-ai tune" in data["calibration_command"]

    @pytest.mark.parametrize("name", ["medical", "finance", "legal", "summarization"])
    def test_regulated_and_summarization_profiles_enable_verified_scorer(self, name):
        cfg = DirectorConfig.from_profile(name)
        scorer = cfg.build_scorer()

        assert cfg.verified_scorer_enabled is True
        assert scorer._verified_scorer_enabled is True
        assert scorer._verified_scorer_atomic is True
        assert scorer._verified_scorer_evidence_top_k == 3
        assert scorer._verified_scorer_min_coverage == pytest.approx(0.5)

    def test_verified_scorer_config_validates_bounds(self):
        with pytest.raises(ValueError, match="verified_scorer_low_confidence_margin"):
            DirectorConfig(verified_scorer_low_confidence_margin=1.5)
        with pytest.raises(ValueError, match="verified_scorer_min_coverage"):
            DirectorConfig(verified_scorer_min_coverage=-0.1)
        with pytest.raises(ValueError, match="verified_scorer_evidence_top_k"):
            DirectorConfig(verified_scorer_evidence_top_k=0)

    def test_span_detector_config_validates_bounds(self):
        with pytest.raises(ValueError, match="span_token_threshold"):
            DirectorConfig(span_token_threshold=1.5)
        with pytest.raises(ValueError, match="span_min_tokens"):
            DirectorConfig(span_min_tokens=0)
        with pytest.raises(ValueError, match="span_max_length"):
            DirectorConfig(span_max_length=0)

    def test_list_profile_metadata_matches_builtin_profiles(self):
        names = {meta.name for meta in DirectorConfig.list_profile_metadata()}

        for name in (
            "fast",
            "lite",
            "rules",
            "embed",
            "thorough",
            "research",
            "production",
            "medical",
            "finance",
            "legal",
            "creative",
            "customer_support",
            "summarization",
        ):
            assert name in names

    def test_unknown_profile_metadata_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            DirectorConfig.profile_metadata("nonexistent")


class TestSummarizationAggregation:
    def test_summarization_profile_uses_min_inner_trimmed_mean_outer(self):
        cfg = DirectorConfig.from_profile("summarization")
        assert cfg.nli_fact_inner_agg == "min"
        assert cfg.nli_fact_outer_agg == "trimmed_mean"

    def test_summarization_profile_logic_agg(self):
        cfg = DirectorConfig.from_profile("summarization")
        assert cfg.nli_logic_inner_agg == "min"
        assert cfg.nli_logic_outer_agg == "mean"

    def test_summarization_profile_premise_ratio(self):
        cfg = DirectorConfig.from_profile("summarization")
        assert cfg.nli_premise_ratio == 0.85

    def test_summarization_profile_thresholds(self):
        cfg = DirectorConfig.from_profile("summarization")
        assert cfg.coherence_threshold == 0.15
        assert cfg.hard_limit == 0.08
        assert cfg.soft_limit == 0.25

    def test_summarization_profile_w_logic_zero(self):
        cfg = DirectorConfig.from_profile("summarization")
        assert cfg.w_logic == 0.0
        assert cfg.w_fact == 1.0

    def test_summarization_profile_retrieval_top_k(self):
        cfg = DirectorConfig.from_profile("summarization")
        assert cfg.nli_fact_retrieval_top_k == 8

    def test_summarization_profile_prompt_as_premise(self):
        cfg = DirectorConfig.from_profile("summarization")
        assert cfg.nli_use_prompt_as_premise is True

    def test_default_profile_uses_max_max(self):
        cfg = DirectorConfig()
        assert cfg.nli_fact_inner_agg == "max"
        assert cfg.nli_fact_outer_agg == "max"
        assert cfg.nli_logic_inner_agg == "max"
        assert cfg.nli_logic_outer_agg == "max"
        assert cfg.nli_premise_ratio == 0.4
        assert cfg.nli_fact_retrieval_top_k == 3


class TestEnvLoading:
    """Tests for from_env()."""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_COHERENCE_THRESHOLD", "0.8")
        monkeypatch.setenv("DIRECTOR_USE_NLI", "true")
        monkeypatch.setenv("DIRECTOR_SERVER_PORT", "9999")
        monkeypatch.setenv("DIRECTOR_FINETUNE_MODELS_DIR", "/srv/director-models")
        cfg = DirectorConfig.from_env()
        assert cfg.coherence_threshold == 0.8
        assert cfg.use_nli is True
        assert cfg.server_port == 9999
        assert cfg.finetune_models_dir == "/srv/director-models"

    def test_env_ignores_unknown(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_TOTALLY_UNKNOWN", "value")
        cfg = DirectorConfig.from_env()
        assert cfg.coherence_threshold == 0.6  # default unchanged

    def test_custom_prefix(self, monkeypatch):
        monkeypatch.setenv("DAI_COHERENCE_THRESHOLD", "0.9")
        cfg = DirectorConfig.from_env(prefix="DAI_")
        assert cfg.coherence_threshold == 0.9

    def test_feedback_db_path_from_env(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_FEEDBACK_DB_PATH", "/tmp/director-feedback.db")
        cfg = DirectorConfig.from_env()
        assert cfg.feedback_db_path == "/tmp/director-feedback.db"


class TestYamlLoading:
    """Tests for from_yaml()."""

    def test_load_json_file(self):
        data = {"coherence_threshold": 0.75, "use_nli": True, "profile": "custom"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            cfg = DirectorConfig.from_yaml(path)
            assert cfg.coherence_threshold == 0.75
            assert cfg.use_nli is True
            assert cfg.profile == "custom"
        finally:
            os.unlink(path)

    def test_load_ignores_unknown_keys(self):
        data = {"coherence_threshold": 0.5, "not_a_real_field": "ignored"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            cfg = DirectorConfig.from_yaml(path)
            assert cfg.coherence_threshold == 0.5
        finally:
            os.unlink(path)

    def test_load_non_dict_returns_default(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('"just a string"')
            path = f.name

        try:
            cfg = DirectorConfig.from_yaml(path)
            assert cfg.coherence_threshold == 0.6
        finally:
            os.unlink(path)

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DirectorConfig.from_yaml("/nonexistent/path.json")


class TestBuildStore:
    """Tests for DirectorConfig.build_store()."""

    def test_build_store_returns_vector_store(self):
        from director_ai.core.vector_store import VectorGroundTruthStore

        cfg = DirectorConfig()
        store = cfg.build_store()
        assert isinstance(store, VectorGroundTruthStore)

    def test_build_store_memory_backend_default(self):
        from director_ai.core.vector_store import InMemoryBackend

        cfg = DirectorConfig(
            vector_backend="memory", hybrid_retrieval=False, reranker_enabled=False
        )
        store = cfg.build_store()
        assert isinstance(store.backend, InMemoryBackend)

    def test_build_scorer_receives_store(self):
        cfg = DirectorConfig()
        scorer = cfg.build_scorer()
        assert scorer.ground_truth_store is not None

    def test_build_scorer_custom_store_override(self):
        from director_ai.core.vector_store import VectorGroundTruthStore

        cfg = DirectorConfig()
        custom_store = VectorGroundTruthStore()
        scorer = cfg.build_scorer(store=custom_store)
        assert scorer.ground_truth_store is custom_store

    def test_build_store_reranker_wraps_backend(self, monkeypatch):
        from unittest.mock import MagicMock

        mock_ce = MagicMock()
        mock_module = MagicMock()
        mock_module.CrossEncoder = MagicMock(return_value=mock_ce)
        monkeypatch.setitem(
            __import__("sys").modules,
            "sentence_transformers",
            mock_module,
        )

        from director_ai.core.vector_store import RerankedBackend

        cfg = DirectorConfig(reranker_enabled=True)
        store = cfg.build_store()
        assert isinstance(store.backend, RerankedBackend)
        mock_module.CrossEncoder.assert_called_once_with(
            cfg.reranker_model,
            device="cpu",
            revision=cfg.reranker_model_revision,
        )

    def test_grounded_recipe_wraps_hybrid_before_reranker(self, monkeypatch):
        from unittest.mock import MagicMock

        mock_module = MagicMock()
        mock_module.CrossEncoder = MagicMock(return_value=MagicMock())
        monkeypatch.setitem(
            __import__("sys").modules,
            "sentence_transformers",
            mock_module,
        )

        from director_ai.core.vector_store import HybridBackend, RerankedBackend

        cfg = DirectorConfig(
            mode="grounded",
            vector_backend="memory",
            hybrid_retrieval=True,
            hybrid_rrf_k=47,
            reranker_enabled=True,
            reranker_top_k_multiplier=5,
        )
        store = cfg.build_store()

        assert isinstance(store.backend, RerankedBackend)
        assert store.backend._multiplier == 5
        assert isinstance(store.backend._base, HybridBackend)
        assert store.backend._base._rrf_k == 47

    def test_build_store_passes_fusion_method_and_weights(self):
        from director_ai.core.vector_store import HybridBackend

        cfg = DirectorConfig(
            mode="grounded",
            vector_backend="memory",
            hybrid_retrieval=True,
            hybrid_fusion_method="convex",
            hybrid_sparse_weight=0.3,
            hybrid_dense_weight=0.7,
            reranker_enabled=False,
        )

        store = cfg.build_store()

        assert isinstance(store.backend, HybridBackend)
        assert store.backend._fusion == "convex"
        assert store.backend._sparse_w == 0.3
        assert store.backend._dense_w == 0.7

    def test_retrieval_recipe_metadata_exposes_grounded_contract(self):
        cfg = DirectorConfig(mode="grounded")

        recipe = cfg.retrieval_recipe()

        assert recipe["name"] == "grounded-hybrid-rerank-v1"
        assert recipe["mode"] == "grounded"
        assert recipe["embedding_model"] == cfg.embedding_model
        assert recipe["hybrid"]["enabled"] is True
        assert recipe["hybrid"]["fusion"] == "reciprocal_rank_fusion"
        assert recipe["hybrid"]["rrf_k"] == 60
        assert recipe["hybrid"]["sparse_weight"] == 1.0
        assert recipe["hybrid"]["dense_weight"] == 1.0

    def test_retrieval_recipe_surfaces_non_default_fusion_method(self):
        cfg = DirectorConfig(
            mode="grounded",
            hybrid_fusion_method="combmnz",
            hybrid_sparse_weight=0.4,
            hybrid_dense_weight=0.6,
        )

        recipe = cfg.retrieval_recipe()

        assert recipe["hybrid"]["fusion"] == "combmnz"
        assert recipe["hybrid"]["sparse_weight"] == 0.4
        assert recipe["hybrid"]["dense_weight"] == 0.6
        assert recipe["reranker"]["enabled"] is True
        assert recipe["reranker"]["top_k_multiplier"] == 3
        assert recipe["abstention"]["threshold"] == pytest.approx(0.3)
        assert "embedding_api_key" not in recipe

    def test_build_store_skips_unavailable_reranker_outside_production(
        self, monkeypatch
    ):
        from unittest.mock import MagicMock

        mock_module = MagicMock()
        mock_module.CrossEncoder = MagicMock(side_effect=FileNotFoundError("missing"))
        monkeypatch.setitem(
            __import__("sys").modules,
            "sentence_transformers",
            mock_module,
        )

        cfg = DirectorConfig(reranker_enabled=True, production_mode=False)
        store = cfg.build_store()
        assert store.backend.__class__.__name__ != "RerankedBackend"

    def test_build_store_fails_unavailable_reranker_in_production(self, monkeypatch):
        from unittest.mock import MagicMock

        mock_module = MagicMock()
        mock_module.CrossEncoder = MagicMock(side_effect=FileNotFoundError("missing"))
        monkeypatch.setitem(
            __import__("sys").modules,
            "sentence_transformers",
            mock_module,
        )

        cfg = DirectorConfig(
            reranker_enabled=True,
            production_mode=True,
            api_keys=("test-key",),
            llm_api_url="https://llm.internal.example/v1",
            knowledge_write_hmac_keys='{"kid-1":"signing-secret-at-least-32-chars-xx"}',
        )
        with pytest.raises(RuntimeError, match="reranker model could not load"):
            cfg.build_store()

    def test_build_store_sentence_transformer_backend(self, monkeypatch):
        from unittest.mock import MagicMock

        mock_st = MagicMock()
        mock_st.SentenceTransformer = MagicMock()
        monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", mock_st)

        from director_ai.core.vector_store import SentenceTransformerBackend

        cfg = DirectorConfig(
            vector_backend="sentence-transformer",
            hybrid_retrieval=False,
            reranker_enabled=False,
        )
        store = cfg.build_store()
        assert isinstance(store.backend, SentenceTransformerBackend)

    def test_build_store_registry_fallback(self):
        from director_ai.core.vector_store import InMemoryBackend

        cfg = DirectorConfig(
            vector_backend="__nonexistent_backend__",
            hybrid_retrieval=False,
            reranker_enabled=False,
        )
        store = cfg.build_store()
        assert isinstance(store.backend, InMemoryBackend)

    def test_hybrid_rrf_k_must_be_positive_integer(self):
        with pytest.raises(ValueError, match="hybrid_rrf_k"):
            DirectorConfig(hybrid_rrf_k=0)

    def test_hybrid_rrf_k_must_not_be_bool(self):
        with pytest.raises(ValueError, match="hybrid_rrf_k must be an integer"):
            DirectorConfig(hybrid_rrf_k=True)

    def test_llm_judge_rubric_must_be_boolean(self):
        with pytest.raises(ValueError, match="llm_judge_rubric must be a boolean"):
            DirectorConfig(llm_judge_rubric="yes")

    def test_llm_judge_ensemble_must_be_integer(self):
        with pytest.raises(ValueError, match="llm_judge_ensemble must be an integer"):
            DirectorConfig(llm_judge_ensemble=True)

    @pytest.mark.parametrize("value", [0, 6])
    def test_llm_judge_ensemble_range(self, value):
        with pytest.raises(ValueError, match="between 1 and 5"):
            DirectorConfig(llm_judge_ensemble=value)

    def test_build_scorer_passes_rubric_and_ensemble(self):
        cfg = DirectorConfig(
            llm_judge_rubric=True,
            llm_judge_ensemble=3,
        )

        scorer = cfg.build_scorer()

        assert scorer._judge._rubric is True
        assert scorer._judge._ensemble_n == 3

    def test_hybrid_fusion_method_is_canonicalised(self):
        cfg = DirectorConfig(hybrid_fusion_method="  ZScore ")

        assert cfg.hybrid_fusion_method == "zscore"

    def test_hybrid_fusion_method_rejects_unknown(self):
        with pytest.raises(ValueError, match="fusion_method must be one of"):
            DirectorConfig(hybrid_fusion_method="borda")

    @pytest.mark.parametrize(
        "field",
        ["hybrid_sparse_weight", "hybrid_dense_weight"],
    )
    def test_hybrid_fusion_weight_must_be_non_negative(self, field):
        with pytest.raises(ValueError, match=f"{field} must be non-negative"):
            DirectorConfig(**{field: -0.5})

    @pytest.mark.parametrize(
        "field",
        ["hybrid_sparse_weight", "hybrid_dense_weight"],
    )
    def test_hybrid_fusion_weight_must_be_numeric(self, field):
        with pytest.raises(ValueError, match=f"{field} must be numeric"):
            DirectorConfig(**{field: True})

    def test_hybrid_fusion_weights_must_not_both_be_zero(self):
        with pytest.raises(ValueError, match="at least one hybrid fusion weight"):
            DirectorConfig(hybrid_sparse_weight=0.0, hybrid_dense_weight=0.0)


class TestValidationBoundaries:
    """Negative tests for __post_init__ validation constraints."""

    def test_coherence_threshold_below_zero(self):
        with pytest.raises(ValueError, match="coherence_threshold"):
            DirectorConfig(coherence_threshold=-0.1)

    def test_coherence_threshold_above_one(self):
        with pytest.raises(ValueError, match="coherence_threshold"):
            DirectorConfig(coherence_threshold=1.1)

    def test_hard_limit_below_zero(self):
        with pytest.raises(ValueError, match="hard_limit"):
            DirectorConfig(hard_limit=-0.01)

    def test_hard_limit_above_one(self):
        with pytest.raises(ValueError, match="hard_limit"):
            DirectorConfig(hard_limit=1.5)

    def test_soft_limit_below_zero(self):
        with pytest.raises(ValueError, match="soft_limit"):
            DirectorConfig(soft_limit=-0.1)

    def test_soft_limit_above_one(self):
        with pytest.raises(ValueError, match="soft_limit"):
            DirectorConfig(soft_limit=2.0)

    def test_soft_limit_below_hard_limit(self):
        with pytest.raises(ValueError, match="soft_limit.*hard_limit"):
            DirectorConfig(hard_limit=0.7, soft_limit=0.3)

    def test_max_candidates_zero(self):
        with pytest.raises(ValueError, match="max_candidates"):
            DirectorConfig(max_candidates=0)

    def test_history_window_zero(self):
        with pytest.raises(ValueError, match="history_window"):
            DirectorConfig(history_window=0)

    def test_temperature_above_two(self):
        with pytest.raises(ValueError, match="llm_temperature"):
            DirectorConfig(llm_temperature=2.5)

    def test_temperature_below_zero(self):
        with pytest.raises(ValueError, match="llm_temperature"):
            DirectorConfig(llm_temperature=-0.1)

    def test_max_tokens_zero(self):
        with pytest.raises(ValueError, match="llm_max_tokens"):
            DirectorConfig(llm_max_tokens=0)

    def test_batch_concurrency_zero(self):
        with pytest.raises(ValueError, match="batch_max_concurrency"):
            DirectorConfig(batch_max_concurrency=0)

    def test_server_port_zero(self):
        with pytest.raises(ValueError, match="server_port"):
            DirectorConfig(server_port=0)

    def test_server_port_above_65535(self):
        with pytest.raises(ValueError, match="server_port"):
            DirectorConfig(server_port=70000)

    def test_server_workers_zero(self):
        with pytest.raises(ValueError, match="server_workers"):
            DirectorConfig(server_workers=0)

    def test_valid_boundary_values_pass(self):
        cfg = DirectorConfig(
            coherence_threshold=0.0,
            hard_limit=0.0,
            soft_limit=0.0,
            llm_temperature=0.0,
            server_port=1,
        )
        assert cfg.coherence_threshold == 0.0
        assert cfg.server_port == 1

    def test_valid_upper_boundary_values_pass(self):
        cfg = DirectorConfig(
            coherence_threshold=1.0,
            hard_limit=1.0,
            soft_limit=1.0,
            llm_temperature=2.0,
            server_port=65535,
        )
        assert cfg.server_port == 65535


class TestEnvCoercionErrors:
    """Error paths in from_env() type coercion."""

    def test_invalid_bool_raises(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_USE_NLI", "maybe")
        with pytest.raises(ValueError, match="invalid bool"):
            DirectorConfig.from_env()

    def test_invalid_int_raises(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_SERVER_PORT", "not_a_number")
        with pytest.raises(ValueError, match="Invalid value"):
            DirectorConfig.from_env()

    def test_invalid_float_raises(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_COHERENCE_THRESHOLD", "xyz")
        with pytest.raises(ValueError, match="Invalid value"):
            DirectorConfig.from_env()


class TestNewV25Fields:
    """Tests for v2.5.0 config fields: stats, source, gRPC."""

    def test_default_stats_backend(self):
        cfg = DirectorConfig()
        assert cfg.stats_backend == "prometheus"
        assert cfg.stats_db_path == "~/.director-ai/stats.db"

    def test_sqlite_stats_backend(self):
        cfg = DirectorConfig(stats_backend="sqlite")
        assert cfg.stats_backend == "sqlite"

    def test_invalid_stats_backend(self):
        with pytest.raises(ValueError, match="stats_backend"):
            DirectorConfig(stats_backend="redis")

    def test_default_source_fields(self):
        cfg = DirectorConfig()
        assert cfg.source_endpoint_enabled is True
        assert cfg.source_repository_url.startswith("https://github.com/")

    def test_grpc_defaults(self):
        cfg = DirectorConfig()
        assert cfg.grpc_max_message_mb == 4
        assert cfg.grpc_deadline_seconds == 30.0

    def test_grpc_max_message_mb_below_one(self):
        with pytest.raises(ValueError, match="grpc_max_message_mb"):
            DirectorConfig(grpc_max_message_mb=0)

    def test_grpc_deadline_zero(self):
        with pytest.raises(ValueError, match="grpc_deadline_seconds"):
            DirectorConfig(grpc_deadline_seconds=0)

    def test_grpc_deadline_negative(self):
        with pytest.raises(ValueError, match="grpc_deadline_seconds"):
            DirectorConfig(grpc_deadline_seconds=-1.0)

    def test_env_override_stats_backend(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_STATS_BACKEND", "sqlite")
        cfg = DirectorConfig.from_env()
        assert cfg.stats_backend == "sqlite"

    def test_env_override_grpc(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_GRPC_MAX_MESSAGE_MB", "8")
        monkeypatch.setenv("DIRECTOR_GRPC_DEADLINE_SECONDS", "60.0")
        cfg = DirectorConfig.from_env()
        assert cfg.grpc_max_message_mb == 8
        assert cfg.grpc_deadline_seconds == 60.0

    def test_to_dict_includes_new_fields(self):
        cfg = DirectorConfig()
        d = cfg.to_dict()
        assert "stats_backend" in d
        assert "grpc_max_message_mb" in d
        assert "source_endpoint_enabled" in d
        assert "onnx_path" in d

    def test_onnx_path_default_empty(self):
        cfg = DirectorConfig()
        assert cfg.onnx_path == ""

    def test_onnx_path_from_env(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_ONNX_PATH", "/models/onnx")
        cfg = DirectorConfig.from_env()
        assert cfg.onnx_path == "/models/onnx"

    def test_build_scorer_passes_onnx_path(self):
        cfg = DirectorConfig(
            scorer_backend="onnx",
            use_nli=True,
            onnx_path="/tmp/onnx_model",
        )
        scorer = cfg.build_scorer()
        assert scorer._nli is not None
        assert scorer._nli._onnx_path == "/tmp/onnx_model"


class TestWeightValidation:
    """Tests for w_logic + w_fact constraint."""

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="w_logic.*w_fact.*1.0"):
            DirectorConfig(w_logic=0.3, w_fact=0.3)

    def test_zero_weights_skip_validation(self):
        cfg = DirectorConfig(w_logic=0.0, w_fact=0.0)
        assert cfg.w_logic == 0.0
        assert cfg.w_fact == 0.0

    def test_valid_weights_pass(self):
        cfg = DirectorConfig(w_logic=0.7, w_fact=0.3)
        assert cfg.w_logic == pytest.approx(0.7)
        assert cfg.w_fact == pytest.approx(0.3)


class TestClaimSupportConfigWiring:
    """DirectorConfig claim-support settings must reach the scorer boundary."""

    def test_claim_support_settings_are_wired_into_scorer(self):
        cfg = DirectorConfig(
            nli_claim_coverage_enabled=False,
            nli_claim_support_threshold=0.7,
            nli_claim_coverage_alpha=0.25,
        )

        scorer = cfg.build_scorer()

        assert scorer._claim_coverage_enabled is False
        assert scorer._claim_support_threshold == 0.7
        assert scorer._claim_coverage_alpha == 0.25


class TestConfigCoverageGaps:
    """Dedicated tests for DirectorConfig validation and scorer wiring branches."""

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"mode": "unsupported"}, "mode"),
            ({"rate_limit_rpm": -1}, "rate_limit_rpm"),
            ({"sanitizer_block_threshold": 1.1}, "sanitizer_block_threshold"),
            (
                {
                    "reranker_enabled": True,
                    "reranker_model": " ",
                },
                "reranker_model",
            ),
            (
                {
                    "injection_require_model_backed_nli": True,
                    "injection_detection_enabled": False,
                },
                "injection_require_model_backed_nli",
            ),
            (
                {
                    "injection_fail_closed_on_error": True,
                    "injection_detection_enabled": False,
                },
                "injection_fail_closed_on_error",
            ),
            (
                {
                    "adaptive_threshold_fail_closed": True,
                    "adaptive_threshold_enabled": False,
                },
                "adaptive_threshold_fail_closed",
            ),
            (
                {
                    "production_mode": True,
                    "dry_run": True,
                    "api_keys": {"k"},
                    "llm_api_url": "https://llm.internal/v1",
                },
                "dry_run",
            ),
            (
                {
                    "production_mode": True,
                    "sanitize_inputs": False,
                    "api_keys": {"k"},
                    "llm_api_url": "https://llm.internal/v1",
                },
                "sanitize_inputs",
            ),
            (
                {
                    "production_mode": True,
                    "api_keys": {"k"},
                    "llm_provider": "local",
                    "llm_api_url": "",
                },
                "local LLM",
            ),
            (
                {
                    "knowledge_write_require_signature": True,
                    "knowledge_write_hmac_keys": {},
                },
                "knowledge_write_hmac_keys",
            ),
            (
                {
                    "vector_backend": "sentence-transformer",
                    "embedding_model": " ",
                },
                "embedding_model",
            ),
            (
                {
                    "vector_backend": "http-faiss",
                    "embedding_base_url": " ",
                    "embedding_model": "embed",
                },
                "embedding_base_url",
            ),
            (
                {
                    "vector_backend": "http-faiss",
                    "embedding_base_url": "https://embed.internal",
                    "embedding_model": " ",
                },
                "embedding_model",
            ),
            (
                {
                    "vector_backend": "remanentia",
                    "remanentia_base_url": " ",
                },
                "remanentia_base_url",
            ),
            ({"embedding_timeout_s": 0}, "embedding_timeout_s"),
            ({"embedding_vector_size": 0}, "embedding_vector_size"),
            ({"remanentia_timeout_s": 0}, "remanentia_timeout_s"),
        ],
    )
    def test_validation_edges(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DirectorConfig(**kwargs)

    def test_grounded_mode_sets_retrieval_abstention_default(self):
        cfg = DirectorConfig(mode="grounded", retrieval_abstention_threshold=0.0)

        assert cfg.use_nli is True
        assert cfg.retrieval_abstention_threshold == pytest.approx(0.3)

    def test_hardened_mode_enforces_fail_closed_settings(self):
        cfg = DirectorConfig(
            hardened=True,
            api_keys={"tenant-key"},
            llm_api_url="https://llm.internal/v1",
            llm_provider="openai",
            knowledge_write_hmac_keys='{"kid-1":"signing-secret-at-least-32-chars-xx"}',
        )

        assert cfg.production_mode is True
        assert cfg.knowledge_write_require_signature is True
        assert cfg.use_nli is True
        assert cfg.coherence_require_model_backed_nli is True
        assert cfg.injection_detection_enabled is True
        assert cfg.injection_require_model_backed_nli is True
        assert cfg.injection_fail_closed_on_error is True
        assert cfg.strict_mode is True

    def test_build_store_general_mode_returns_ground_truth_store(self):
        from director_ai.core.retrieval.knowledge import GroundTruthStore

        store = DirectorConfig(mode="general").build_store()

        assert isinstance(store, GroundTruthStore)

    def test_build_store_remanentia_skips_local_decorators(self, monkeypatch):
        class FakeRemanentiaBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        import director_ai.core.retrieval.vector_store as vector_store_module

        monkeypatch.setattr(
            vector_store_module,
            "RemanentiaVectorBackend",
            FakeRemanentiaBackend,
        )

        cfg = DirectorConfig(
            mode="grounded",
            vector_backend="remanentia",
            remanentia_base_url="https://remanentia.internal",
            hybrid_retrieval=True,
            reranker_enabled=True,
            parent_child_enabled=True,
            hyde_enabled=True,
            query_decomposition_enabled=True,
            contextual_compression_enabled=True,
            multi_vector_enabled=True,
        )
        store = cfg.build_store()

        backend_chain = []
        backend = store.backend
        while backend is not None:
            backend_chain.append(backend)
            backend = getattr(backend, "_base", None)

        assert any(isinstance(item, FakeRemanentiaBackend) for item in backend_chain)
        assert all(item.__class__.__name__ != "HybridBackend" for item in backend_chain)
        assert all(
            item.__class__.__name__ != "RerankedBackend" for item in backend_chain
        )
        remanentia = next(
            item for item in backend_chain if isinstance(item, FakeRemanentiaBackend)
        )
        assert remanentia.kwargs["base_url"] == "https://remanentia.internal"

    def test_resolve_scorer_backend_auto_paths(self, monkeypatch):
        import importlib.util

        cfg = DirectorConfig(scorer_backend="auto", onnx_path="/tmp/model.onnx")
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert cfg._resolve_scorer_backend() == "onnx"

        cfg = DirectorConfig(scorer_backend="auto", onnx_path="", use_nli=True)
        assert cfg._resolve_scorer_backend() == "deberta"

        cfg = DirectorConfig(
            mode="grounded",
            scorer_backend="auto",
            onnx_path="",
            use_nli=False,
            hybrid_retrieval=False,
            reranker_enabled=False,
        )
        assert cfg._resolve_scorer_backend() == "lite"

    def test_build_scorer_wires_optional_runtime_features(self, monkeypatch):
        import director_ai.core.scoring.scorer as scorer_module

        captured = {}

        class FakeNLI:
            def __init__(self):
                self.loaded = []

            def _load_lora_adapter(self, path):
                self.loaded.append(path)

        class FakeJudge:
            pass

        class FakeScorer:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs
                self._nli = FakeNLI()
                self._judge = FakeJudge()
                self._adaptive_threshold_enabled = False
                self._adaptive_threshold_fail_closed = False

            def enable_injection_detection(self, **kwargs):
                captured["injection"] = kwargs

            def enable_adaptive_retrieval(self, **kwargs):
                captured["adaptive_retrieval"] = kwargs

            def _get_meta_classifier(self):
                return object()

            def _has_model_backed_nli(self):
                return True

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)

        cfg = DirectorConfig(
            llm_judge_provider="local",
            llm_judge_local_model="local-judge",
            onnx_path="/tmp/model.onnx",
            w_logic=0.6,
            w_fact=0.4,
            nli_devices="cpu,cuda:0",
            injection_detection_enabled=True,
            lora_adapter_path="/tmp/adapter",
            meta_classifier_path="/tmp/meta.pkl",
            adaptive_retrieval_enabled=True,
            adaptive_retrieval_threshold=0.42,
            dry_run=True,
            cost_tracking_enabled=True,
        )

        scorer = cfg.build_scorer(store=object())

        assert captured["kwargs"]["llm_judge_model"] == "local-judge"
        assert captured["kwargs"]["onnx_path"] == "/tmp/model.onnx"
        assert captured["kwargs"]["w_logic"] == 0.6
        assert captured["kwargs"]["w_fact"] == 0.4
        assert captured["kwargs"]["nli_devices"] == ["cpu", "cuda:0"]
        assert captured["injection"]["injection_threshold"] == cfg.injection_threshold
        assert captured["adaptive_retrieval"] == {"threshold": 0.42}
        assert scorer._nli.loaded == ["/tmp/adapter"]
        assert scorer._meta_classifier_path == "/tmp/meta.pkl"
        assert scorer._dry_run is True
        assert scorer._cost_analyser is not None
        assert scorer._judge._cost_callback is not None

    def test_build_scorer_passes_strict_mode(self, monkeypatch):
        # Regression: hardened mode set a non-field ``strict_mode`` attribute
        # that build_scorer never forwarded, so the scorer's fail-closed
        # divergence path stayed disabled under hardened.
        import director_ai.core.scoring.scorer as scorer_module

        captured = {}

        class FakeScorer:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs
                self._nli = None
                self._judge = None
                self._adaptive_threshold_enabled = False
                self._adaptive_threshold_fail_closed = False

            def _has_model_backed_nli(self):
                return True

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)

        DirectorConfig(strict_mode=True).build_scorer(store=object())
        assert captured["kwargs"]["strict_mode"] is True

        DirectorConfig().build_scorer(store=object())
        assert captured["kwargs"]["strict_mode"] is False

    def test_hardened_mode_strict_mode_reaches_scorer(self, monkeypatch):
        import director_ai.core.scoring.scorer as scorer_module

        captured = {}

        class FakeScorer:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs
                self._nli = None
                self._judge = None
                self._adaptive_threshold_enabled = False
                self._adaptive_threshold_fail_closed = False

            def enable_injection_detection(self, **kwargs):
                pass

            def _get_meta_classifier(self):
                return object()

            def _has_model_backed_nli(self):
                return True

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)

        cfg = DirectorConfig(
            hardened=True,
            api_keys={"tenant-key"},
            llm_api_url="https://llm.internal/v1",
            llm_provider="openai",
            knowledge_write_hmac_keys='{"kid-1":"signing-secret-at-least-32-chars-xx"}',
        )
        assert cfg.strict_mode is True
        cfg.build_scorer(store=object())
        assert captured["kwargs"]["strict_mode"] is True

    def test_build_scorer_redis_cache_import_failure_is_nonfatal(self, monkeypatch):
        import director_ai.core.scoring.scorer as scorer_module

        class FakeScorer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self._nli = None
                self._judge = object()
                self._adaptive_threshold_enabled = False
                self._adaptive_threshold_fail_closed = False

            def _has_model_backed_nli(self):
                return True

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)
        monkeypatch.setitem(sys.modules, "director_ai.enterprise.redis", None)

        cfg = DirectorConfig(redis_url="redis://cache.internal/0")
        scorer = cfg.build_scorer(store=object())

        assert "cache" not in scorer.kwargs
        assert "cache_size" not in scorer.kwargs

    def test_build_store_uses_enterprise_redis_when_available(self, monkeypatch):
        class FakeRedisGroundTruthStore:
            def __init__(self, *, redis_url, prefix):
                self.redis_url = redis_url
                self.prefix = prefix

        redis_module = types.ModuleType("director_ai.enterprise.redis")
        redis_module.RedisGroundTruthStore = FakeRedisGroundTruthStore
        redis_module.RedisScoreCache = object
        monkeypatch.setitem(
            sys.modules,
            "director_ai.enterprise",
            types.ModuleType("director_ai.enterprise"),
        )
        monkeypatch.setitem(sys.modules, "director_ai.enterprise.redis", redis_module)

        cfg = DirectorConfig(
            mode="grounded",
            redis_url="redis://cache.internal/0",
            redis_prefix="dai:",
            hybrid_retrieval=False,
            reranker_enabled=False,
        )
        store = cfg.build_store()

        assert isinstance(store, FakeRedisGroundTruthStore)
        assert store.redis_url == "redis://cache.internal/0"
        assert store.prefix == "dai:facts:"

    def test_build_scorer_uses_enterprise_redis_cache_when_available(self, monkeypatch):
        import director_ai.core.scoring.scorer as scorer_module

        class FakeRedisScoreCache:
            def __init__(self, *, redis_url, prefix, ttl_seconds):
                self.redis_url = redis_url
                self.prefix = prefix
                self.ttl_seconds = ttl_seconds

        class FakeScorer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self._nli = None
                self._judge = object()
                self._adaptive_threshold_enabled = False
                self._adaptive_threshold_fail_closed = False

            def _has_model_backed_nli(self):
                return True

        redis_module = types.ModuleType("director_ai.enterprise.redis")
        redis_module.RedisGroundTruthStore = object
        redis_module.RedisScoreCache = FakeRedisScoreCache
        monkeypatch.setitem(
            sys.modules,
            "director_ai.enterprise",
            types.ModuleType("director_ai.enterprise"),
        )
        monkeypatch.setitem(sys.modules, "director_ai.enterprise.redis", redis_module)
        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)

        cfg = DirectorConfig(
            redis_url="redis://cache.internal/0",
            redis_prefix="dai:",
            cache_ttl=123,
        )
        scorer = cfg.build_scorer(store=object())

        cache = scorer.kwargs["cache"]
        assert isinstance(cache, FakeRedisScoreCache)
        assert cache.redis_url == "redis://cache.internal/0"
        assert cache.prefix == "dai:cache:"
        assert cache.ttl_seconds == 123

    def test_build_store_hyde_template_wraps_backend(self):
        cfg = DirectorConfig(
            mode="grounded",
            hybrid_retrieval=False,
            reranker_enabled=False,
            hyde_enabled=True,
            hyde_prompt_template="Write a hypothetical answer: {query}",
        )

        store = cfg.build_store()

        assert store.backend.__class__.__name__ == "HyDEBackend"
        assert store.backend._template == "Write a hypothetical answer: {query}"

    def test_production_tenant_routing_requires_binding_map(self):
        with pytest.raises(ValueError, match="api_key_tenant_map"):
            DirectorConfig(
                production_mode=True,
                tenant_routing=True,
                api_keys={"tenant-key"},
                llm_api_url="https://llm.internal/v1",
                knowledge_write_hmac_keys=(
                    '{"kid-1":"signing-secret-at-least-32-chars-xx"}'
                ),
            )

    def test_production_wildcard_host_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="DirectorAI.Config"):
            cfg = DirectorConfig(
                production_mode=True,
                tenant_routing=True,
                api_key_tenant_map='{"tenant-key":"tenant-a"}',
                llm_api_url="https://llm.internal/v1",
                server_host="0.0.0.0",
                knowledge_write_hmac_keys=(
                    '{"kid-1":"signing-secret-at-least-32-chars-xx"}'
                ),
            )

        assert cfg.server_host == "0.0.0.0"
        assert "binding to 0.0.0.0" in caplog.text

    def test_model_revision_health_uses_local_judge_model(self, monkeypatch):
        import director_ai.core.model_revisions as model_revisions_module

        captured = {}

        def fake_model_revision_health(models):
            captured["models"] = models
            return {"ok": True}

        monkeypatch.setattr(
            model_revisions_module,
            "model_revision_health",
            fake_model_revision_health,
        )

        cfg = DirectorConfig(
            llm_judge_provider="local",
            llm_judge_local_model="local-judge",
            llm_judge_model="remote-judge",
        )
        health = cfg.model_revision_health()

        assert health == {"ok": True}
        assert captured["models"]["local_judge"] == ("local-judge", None)

    def test_coerce_numeric_lists(self):
        from director_ai.core.config import _coerce

        assert _coerce("1, 2,3", "list[int]") == [1, 2, 3]
        assert _coerce("0.1,0.2", "list[float]") == [0.1, 0.2]

    def test_coerce_string_list_splits_and_strips(self):
        # The plain ``list[str]`` branch splits on commas and drops blank
        # entries without per-element numeric conversion.
        from director_ai.core.config import _coerce

        assert _coerce("alpha, beta ,, gamma", "list[str]") == [
            "alpha",
            "beta",
            "gamma",
        ]
        assert _coerce("solo", "list") == ["solo"]

    def test_coerce_tuple_fields(self):
        from director_ai.core.config import _coerce

        # ``tuple[str, ...]`` must split on commas like a list — otherwise the
        # raw string survives and ``frozenset(value)`` becomes a per-character
        # set, silently corrupting allowlists.
        assert _coerce("public, internal", "tuple[str, ...]") == (
            "public",
            "internal",
        )
        assert _coerce("1,2", "tuple[int, ...]") == (1, 2)
        assert _coerce("0.5,0.25", "tuple[float, ...]") == (0.5, 0.25)

    def test_from_env_api_keys_accepts_json_array(self, monkeypatch):
        # Regression: from_env routed api_keys through the generic list
        # coercion, which split a JSON array on commas and embedded
        # brackets/quotes into the literal keys (auth never matched).
        from director_ai.core.config import DirectorConfig

        monkeypatch.setenv("DIRECTOR_API_KEYS", '["sk-alpha","sk-beta"]')
        assert DirectorConfig.from_env().api_keys == ["sk-alpha", "sk-beta"]
        monkeypatch.setenv("DIRECTOR_API_KEYS", "sk-a,sk-b")
        assert DirectorConfig.from_env().api_keys == ["sk-a", "sk-b"]

    def test_from_env_tuple_field_is_split(self, monkeypatch):
        # Regression: tuple[str, ...] fields fell through _coerce unchanged and
        # stayed a raw string, breaking frozenset-based allowlist checks.
        from director_ai.core.config import DirectorConfig

        monkeypatch.setenv(
            "DIRECTOR_EVIDENCE_FIREWALL_ALLOWED_SENSITIVITY", "public,internal"
        )
        cfg = DirectorConfig.from_env()
        assert cfg.evidence_firewall_allowed_sensitivity == ("public", "internal")
        assert "public" in frozenset(cfg.evidence_firewall_allowed_sensitivity)

    @pytest.mark.parametrize(
        "method_name,kwargs,match",
        [
            (
                "enable_injection_detection",
                {
                    "injection_detection_enabled": True,
                    "injection_require_model_backed_nli": True,
                },
                "detector init failed",
            ),
            (
                "_get_meta_classifier",
                {
                    "adaptive_threshold_enabled": True,
                    "adaptive_threshold_fail_closed": True,
                },
                "classifier init failed",
            ),
        ],
    )
    def test_build_scorer_propagates_fail_closed_startup_errors(
        self,
        monkeypatch,
        method_name,
        kwargs,
        match,
    ):
        import director_ai.core.scoring.scorer as scorer_module

        class FakeScorer:
            def __init__(self, **init_kwargs):
                del init_kwargs
                self._nli = None
                self._judge = object()
                self._adaptive_threshold_enabled = kwargs.get(
                    "adaptive_threshold_enabled",
                    False,
                )
                self._adaptive_threshold_fail_closed = kwargs.get(
                    "adaptive_threshold_fail_closed",
                    False,
                )

            def enable_injection_detection(self, **init_kwargs):
                del init_kwargs
                if method_name == "enable_injection_detection":
                    raise RuntimeError("detector init failed")

            def _get_meta_classifier(self):
                if method_name == "_get_meta_classifier":
                    raise RuntimeError("classifier init failed")
                return object()

            def _has_model_backed_nli(self):
                return True

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)

        cfg = DirectorConfig(**kwargs)

        with pytest.raises(RuntimeError, match=match):
            cfg.build_scorer(store=object())

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (
                {"coherence_require_model_backed_nli": True},
                "coherence_require_model_backed_nli",
            ),
            (
                {
                    "injection_detection_enabled": True,
                    "injection_require_model_backed_nli": True,
                },
                "injection_require_model_backed_nli",
            ),
            (
                {
                    "adaptive_threshold_enabled": True,
                    "adaptive_threshold_fail_closed": True,
                },
                "adaptive_threshold_fail_closed",
            ),
        ],
    )
    def test_build_scorer_fail_closed_unavailable_components(
        self,
        monkeypatch,
        kwargs,
        match,
    ):
        import director_ai.core.scoring.scorer as scorer_module

        class FakeScorer:
            def __init__(self, **init_kwargs):
                del init_kwargs
                self._nli = None
                self._judge = object()
                self._adaptive_threshold_enabled = kwargs.get(
                    "adaptive_threshold_enabled",
                    False,
                )
                self._adaptive_threshold_fail_closed = kwargs.get(
                    "adaptive_threshold_fail_closed",
                    False,
                )

            def enable_injection_detection(self, **init_kwargs):
                del init_kwargs

            def _get_meta_classifier(self):
                return None

            def _has_model_backed_nli(self):
                return False

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)

        cfg = DirectorConfig(**kwargs)

        with pytest.raises(RuntimeError, match=match):
            cfg.build_scorer(store=object())

    def test_build_scorer_warns_when_nli_backend_lacks_lora_support(
        self, monkeypatch, caplog
    ):
        """A LoRA adapter set on a backend without LoRA support warns, not crashes.

        When ``lora_adapter_path`` is configured but the active NLI backend does
        not implement ``_load_lora_adapter``, ``build_scorer`` must log which
        backend was skipped and return the scorer unchanged rather than raising.
        """
        import director_ai.core.scoring.scorer as scorer_module

        class _NliWithoutLora:
            """An NLI backend deliberately missing ``_load_lora_adapter``."""

        class FakeScorer:
            def __init__(self, **init_kwargs):
                del init_kwargs
                self._nli = _NliWithoutLora()
                self._judge = object()
                self._adaptive_threshold_enabled = False
                self._adaptive_threshold_fail_closed = False

            def _has_model_backed_nli(self):
                return True

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)

        cfg = DirectorConfig(lora_adapter_path="/tmp/nonexistent-adapter")
        with caplog.at_level(logging.WARNING, logger="DirectorAI.Config"):
            scorer = cfg.build_scorer(store=object())

        assert isinstance(scorer._nli, _NliWithoutLora)
        assert "LoRA adapter not supported" in caplog.text
        assert "_NliWithoutLora" in caplog.text


class TestValidationBranchCoverage:
    """Exercise the normalisation/validation branches in config_validation."""

    def test_scorer_model_alias_resolves_nli_fields(self):
        # A registry alias rewrites the NLI model identity through
        # resolve_scorer_model_choice during __post_init__.
        from director_ai.core.scoring.model_choices import (
            resolve_scorer_model_choice,
        )

        choice = resolve_scorer_model_choice("balanced-default")
        cfg = DirectorConfig(scorer_model="balanced-default", use_nli=True)
        assert cfg.nli_model == choice.runtime_model
        assert cfg.nli_model_artifact_uri == choice.artifact_uri
        assert cfg.nli_model_revision == choice.revision
        assert cfg.nli_max_length == choice.max_length

    def test_grounded_mode_leaves_preset_retrieval_untouched(self):
        # mode 'auto' with every retrieval signal already off and a positive
        # abstention threshold skips both normalisation branches: use_nli stays
        # False and the preset threshold is preserved.
        cfg = DirectorConfig(
            mode="auto",
            use_nli=False,
            hybrid_retrieval=False,
            reranker_enabled=False,
            retrieval_abstention_threshold=0.5,
        )
        assert cfg.use_nli is False
        assert cfg.retrieval_abstention_threshold == 0.5

    def test_http_faiss_with_base_url_and_model_validates(self):
        # vector_backend 'http-faiss' passes validation once both the base URL
        # and the embedding model are set.
        cfg = DirectorConfig(
            vector_backend="http-faiss",
            embedding_base_url="http://embeddings.local",
            embedding_model="bge-small",
        )
        assert cfg.vector_backend == "http-faiss"
        assert cfg.embedding_base_url == "http://embeddings.local"


class TestBuildBackendBranches:
    """Cover the optional-backend dispatch branches in config_builders."""

    def test_build_store_chroma_backend(self, monkeypatch):
        import director_ai.core.retrieval.vector_store as vs

        captured = {}

        class FakeChroma:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def add(self, *_a, **_k):
                pass

            def query(self, *_a, **_k):
                return []

            def count(self):
                return 0

        monkeypatch.setattr(vs, "ChromaBackend", FakeChroma)
        cfg = DirectorConfig(
            vector_backend="chroma",
            chroma_collection="c1",
            chroma_persist_dir="/tmp/chroma",
            hybrid_retrieval=False,
            reranker_enabled=False,
        )
        store = cfg.build_store()
        assert isinstance(store.backend, FakeChroma)
        assert captured["kwargs"]["collection_name"] == "c1"
        assert captured["kwargs"]["persist_directory"] == "/tmp/chroma"

    def test_build_store_http_faiss_backend(self, monkeypatch):
        import director_ai.core.retrieval.vector_store as vs

        class FakeEmbed:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeFaiss:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def add(self, *_a, **_k):
                pass

            def query(self, *_a, **_k):
                return []

            def count(self):
                return 0

        monkeypatch.setattr(vs, "HttpEmbeddingFunction", FakeEmbed)
        monkeypatch.setattr(vs, "FAISSBackend", FakeFaiss)
        cfg = DirectorConfig(
            vector_backend="http-faiss",
            embedding_base_url="http://embed.local",
            embedding_model="bge",
            embedding_vector_size=256,
            hybrid_retrieval=False,
            reranker_enabled=False,
        )
        store = cfg.build_store()
        assert isinstance(store.backend, FakeFaiss)
        assert store.backend.kwargs["vector_size"] == 256

    def test_build_scorer_model_fallback_resolves(self, monkeypatch):
        import director_ai.core.model_registry as registry_mod
        import director_ai.core.scoring.scorer as scorer_module

        captured = {}

        class FakeScorer:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs
                self._nli = None
                self._judge = None
                self._adaptive_threshold_enabled = False
                self._adaptive_threshold_fail_closed = False

            def _has_model_backed_nli(self):
                return True

        class FakeResolved:
            model_id = "fallback/model"
            revision = "abc123"

        class FakeRegistry:
            def resolve(self, *_args, **_kwargs):
                return FakeResolved()

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)
        monkeypatch.setattr(registry_mod, "FallbackModelRegistry", FakeRegistry)
        DirectorConfig(model_fallback_enabled=True).build_scorer(store=object())
        assert captured["kwargs"]["nli_model"] == "fallback/model"
        assert captured["kwargs"]["nli_revision"] == "abc123"

    def test_build_scorer_injection_init_failure_requires_model_backed(
        self, monkeypatch
    ):
        import director_ai.core.scoring.scorer as scorer_module

        class FakeScorer:
            def __init__(self, **kwargs):
                self._nli = None
                self._judge = None
                self._adaptive_threshold_enabled = False
                self._adaptive_threshold_fail_closed = False

            def enable_injection_detection(self, **_kwargs):
                raise RuntimeError("injection detector init failed")

            def _has_model_backed_nli(self):
                return True

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)
        cfg = DirectorConfig(
            injection_detection_enabled=True,
            injection_require_model_backed_nli=True,
        )
        with pytest.raises(RuntimeError, match="injection detector init failed"):
            cfg.build_scorer(store=object())

    def test_build_scorer_injection_init_failure_without_model_backed(
        self, monkeypatch
    ):
        # Same failure path, but require_model_backed_nli=False: the handler
        # skips the model-backed metric and still re-raises.
        import director_ai.core.scoring.scorer as scorer_module

        class FakeScorer:
            def __init__(self, **kwargs):
                self._nli = None
                self._judge = None
                self._adaptive_threshold_enabled = False
                self._adaptive_threshold_fail_closed = False

            def enable_injection_detection(self, **_kwargs):
                raise RuntimeError("injection detector init failed")

            def _has_model_backed_nli(self):
                return True

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)
        cfg = DirectorConfig(injection_detection_enabled=True)
        with pytest.raises(RuntimeError, match="injection detector init failed"):
            cfg.build_scorer(store=object())

    def test_build_scorer_cost_callback_records(self, monkeypatch):
        import director_ai.core.scoring.scorer as scorer_module

        class FakeJudge:
            pass

        class FakeScorer:
            def __init__(self, **kwargs):
                self._nli = None
                self._judge = FakeJudge()
                self._adaptive_threshold_enabled = False
                self._adaptive_threshold_fail_closed = False

            def _has_model_backed_nli(self):
                return True

        monkeypatch.setattr(scorer_module, "CoherenceScorer", FakeScorer)
        scorer = DirectorConfig(cost_tracking_enabled=True).build_scorer(store=object())
        # Invoke the wired callback so the recording body runs.
        scorer._judge._cost_callback("gpt-x", 12, 34)
        assert "gpt-x" in scorer._cost_analyser._records


def test_profile_metadata_registry_contract():
    from director_ai.core.config_profiles import PROFILE_METADATA, ProfileMetadata

    assert PROFILE_METADATA
    for name, meta in PROFILE_METADATA.items():
        assert isinstance(meta, ProfileMetadata)
        assert meta.name == name
        assert meta.intended_workload
        assert meta.validation_status
        assert meta.min_calibration_samples >= 0
