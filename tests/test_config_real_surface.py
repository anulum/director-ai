# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for DirectorConfig loading and runtime wiring."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.retrieval.vector_store.base import InMemoryBackend
from director_ai.core.retrieval.vector_store.store import VectorGroundTruthStore
from director_ai.core.scoring.scorer import CoherenceScorer
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def test_env_loading_coerces_real_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment loading reads and coerces real process environment values."""
    monkeypatch.setenv("DIRECTOR_COHERENCE_THRESHOLD", "0.72")
    monkeypatch.setenv("DIRECTOR_MODE", "general")
    monkeypatch.setenv("DIRECTOR_USE_NLI", "false")
    monkeypatch.setenv("DIRECTOR_SERVER_PORT", "9173")
    monkeypatch.setenv("DIRECTOR_API_KEYS", '["tenant-key-a", "tenant-key-b"]')

    config = DirectorConfig.from_env()

    assert config.coherence_threshold == pytest.approx(0.72)
    assert config.mode == "general"
    assert config.use_nli is True
    assert config.server_port == 9173
    assert config.api_keys == ["tenant-key-a", "tenant-key-b"]


def test_env_loading_rejects_invalid_bool_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment loading should fail closed on invalid boolean text."""
    monkeypatch.setenv("DIRECTOR_PRODUCTION_MODE", "maybe")

    with pytest.raises(ValueError, match="invalid bool"):
        DirectorConfig.from_env()


def test_json_file_loading_filters_unknown_fields(tmp_path: Path) -> None:
    """JSON config loading keeps known fields and ignores unknown fields."""
    config_path = tmp_path / "director-config.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "operator-local",
                "mode": "general",
                "coherence_threshold": 0.81,
                "use_nli": False,
                "vector_backend": "memory",
                "unknown_future_field": "ignored",
            }
        ),
        encoding="utf-8",
    )

    config = DirectorConfig.from_yaml(str(config_path))

    assert config.profile == "operator-local"
    assert config.mode == "general"
    assert config.coherence_threshold == pytest.approx(0.81)
    assert config.use_nli is True
    assert config.vector_backend == "memory"
    assert "unknown_future_field" not in config.to_dict()


def test_production_profile_loads_real_managed_env_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production profile fails closed unless real env secrets are present."""
    monkeypatch.delenv("DIRECTOR_API_KEYS", raising=False)
    monkeypatch.setenv("DIRECTOR_API_KEYS", "prod-key-a,prod-key-b")
    monkeypatch.setenv(
        "DIRECTOR_API_KEY_TENANT_MAP",
        '{"prod-key-a":"tenant-a","prod-key-b":"tenant-b"}',
    )
    monkeypatch.setenv(
        "DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS",
        '{"kid-1":"prod-signing-secret-at-least-32-chars-xx"}',
    )

    config = DirectorConfig.from_profile("production")

    assert config.production_mode is True
    assert config.profile == "production"
    assert config.api_keys == ["prod-key-a", "prod-key-b"]
    assert json.loads(config.api_key_tenant_map) == {
        "prod-key-a": "tenant-a",
        "prod-key-b": "tenant-b",
    }
    assert config.knowledge_write_hmac_keys.startswith("{")
    assert config.llm_provider == "local"


def test_memory_store_and_scorer_build_without_fake_dependencies() -> None:
    """Memory retrieval and scorer construction use real runtime objects."""
    config = DirectorConfig(
        use_nli=False,
        vector_backend="memory",
        hybrid_retrieval=False,
        reranker_enabled=False,
    )

    store = config.build_store()
    scorer = config.build_scorer(store=store)

    assert isinstance(store, VectorGroundTruthStore)
    assert isinstance(store.backend, InMemoryBackend)
    assert isinstance(scorer, CoherenceScorer)
    assert scorer.ground_truth_store is store


def test_cors_default_and_explicit_origins_are_preserved() -> None:
    """CORS defaults and explicit origins should survive config construction."""
    default_config = DirectorConfig()
    explicit_config = DirectorConfig(cors_origins="https://console.example")

    assert default_config.cors_origins == ""
    assert explicit_config.cors_origins == "https://console.example"


def test_scorer_edge_cases_unit_guard_declares_real_surface_companion() -> None:
    """The scorer edge-case unit guard is backed by this real config surface."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_scorer_edge_cases.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_config_real_surface.py" in reason


@pytest.mark.parametrize(
    ("prompt", "response"),
    [
        ("", ""),
        ("   \n\t  ", "Normal response"),
        ("\u010co je 2+2?", "Odpove\u010f je 4"),
        ("test\x00prompt", "test\x00response"),
    ],
)
def test_configured_scorer_reviews_edge_inputs_through_public_surface(
    prompt: str,
    response: str,
) -> None:
    """A config-built scorer reviews edge inputs through the public API."""
    config = DirectorConfig(
        use_nli=False,
        scorer_backend="lite",
        vector_backend="memory",
        hybrid_retrieval=False,
        reranker_enabled=False,
    )
    scorer = config.build_scorer()

    approved, score = scorer.review(prompt, response)

    assert isinstance(approved, bool)
    assert score.approved is approved
    assert 0.0 <= score.score <= 1.0
    assert 0.0 <= score.h_logical <= 1.0
    assert 0.0 <= score.h_factual <= 1.0


def test_json_logging_configuration_installs_structured_handler() -> None:
    """JSON logging setup installs the configured structured formatter."""
    logger = logging.getLogger("DirectorAI")
    original_level = logger.level
    original_handlers = list(logger.handlers)
    config = DirectorConfig(log_json=True, log_level="DEBUG")

    try:
        config.configure_logging()

        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1
        assert logger.handlers[0].formatter is not None
        assert logger.handlers[0].formatter.__class__.__name__ == "JsonLogFormatter"
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)


def test_plain_logging_configuration_preserves_handlers() -> None:
    """Plain logging updates level without replacing existing handlers."""
    logger = logging.getLogger("DirectorAI")
    original_level = logger.level
    original_handlers = list(logger.handlers)
    sentinel = logging.NullHandler()
    logger.handlers = [sentinel]
    config = DirectorConfig(log_json=False, log_level="WARNING")

    try:
        config.configure_logging()

        assert logger.level == logging.WARNING
        assert logger.handlers == [sentinel]
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)


def test_default_optional_builders_remain_disabled() -> None:
    """Optional integration builders return ``None`` until explicitly enabled."""
    config = DirectorConfig()

    assert config.build_contradiction_halt() is None
    assert config.build_correctness_feedback() is None


def test_local_judge_revision_health_reports_local_model() -> None:
    """Revision health uses the local judge model when that provider is active."""
    config = DirectorConfig(
        llm_judge_provider="local",
        llm_judge_local_model="local-judge-model",
    )

    health = config.model_revision_health()

    assert health["ok"] is True
    checks = health["checks"]
    assert isinstance(checks, dict)
    local_judge = checks["local_judge"]
    assert isinstance(local_judge, dict)
    assert local_judge["model"] == "local-judge-model"
    assert local_judge["status"] == "unversioned-local"


def test_local_judge_revision_health_falls_back_to_default_model() -> None:
    """Local judge health uses the provider model when no local model is set."""
    config = DirectorConfig(llm_judge_provider="local", llm_judge_model="judge-model")

    health = config.model_revision_health()
    checks = health["checks"]
    assert isinstance(checks, dict)
    local_judge = checks["local_judge"]
    assert isinstance(local_judge, dict)
    assert local_judge["model"] == "judge-model"
