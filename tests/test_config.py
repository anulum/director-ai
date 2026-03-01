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

    @pytest.mark.parametrize(
        "name,threshold,hard,soft,nli,reranker,wl,wf",
        [
            ("medical", 0.75, 0.55, 0.75, True, True, 0.5, 0.5),
            ("finance", 0.70, 0.50, 0.70, True, True, 0.4, 0.6),
            ("legal", 0.68, 0.45, 0.68, True, False, 0.6, 0.4),
            ("creative", 0.40, 0.30, 0.45, False, False, 0.7, 0.3),
            ("customer_support", 0.55, 0.40, 0.60, False, False, 0.5, 0.5),
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
