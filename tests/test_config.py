# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Configuration Manager Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import json
import os
import tempfile

import pytest

from director_ai.core.config import DirectorConfig


class TestDirectorConfig:
    """Tests for DirectorConfig dataclass."""

    def test_default_values(self):
        cfg = DirectorConfig()
        assert cfg.coherence_threshold == 0.6
        assert cfg.hard_limit == 0.5
        assert cfg.use_nli is False
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

    def test_to_dict_redacts_api_key(self):
        cfg = DirectorConfig(llm_api_key="sk-secret-123")
        d = cfg.to_dict()
        assert d["llm_api_key"] == "***"

    def test_to_dict_empty_key_not_redacted(self):
        cfg = DirectorConfig(llm_api_key="")
        d = cfg.to_dict()
        assert d["llm_api_key"] == ""


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

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            DirectorConfig.from_profile("nonexistent")


class TestEnvLoading:
    """Tests for from_env()."""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_COHERENCE_THRESHOLD", "0.8")
        monkeypatch.setenv("DIRECTOR_USE_NLI", "true")
        monkeypatch.setenv("DIRECTOR_SERVER_PORT", "9999")
        cfg = DirectorConfig.from_env()
        assert cfg.coherence_threshold == 0.8
        assert cfg.use_nli is True
        assert cfg.server_port == 9999

    def test_env_ignores_unknown(self, monkeypatch):
        monkeypatch.setenv("DIRECTOR_TOTALLY_UNKNOWN", "value")
        cfg = DirectorConfig.from_env()
        assert cfg.coherence_threshold == 0.6  # default unchanged

    def test_custom_prefix(self, monkeypatch):
        monkeypatch.setenv("DAI_COHERENCE_THRESHOLD", "0.9")
        cfg = DirectorConfig.from_env(prefix="DAI_")
        assert cfg.coherence_threshold == 0.9


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
