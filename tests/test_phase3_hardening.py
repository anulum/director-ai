# ─────────────────────────────────────────────────────────────────────
# Phase 3 Hardening Tests (H28-H44, core only)
# ─────────────────────────────────────────────────────────────────────
"""
Tests for Phase 3 hardening fixes (consumer core):
  H28  ROB-5: NLI assert → RuntimeError
  H29  CON-1: batch asyncio.get_running_loop()
  H30  ROB-1: batch coherence None guard
  H34  SEC-3: actor response.text truncation
  H35  SEC-5: config _coerce error message
  H36  API-2: config server_port/workers validation
  H37  RES-3: config from_yaml UTF-8
  H39  API-1: cli --port safety
  H42  CON-2: scorer history thread lock
  H44  ROB-8: scorer setLevel removed
"""

import json
import os
import tempfile
import threading

import pytest

# ── H28: NLI assert → RuntimeError ──────────────────────────────────


class TestH28NLIAssert:
    """NLI scorer should raise RuntimeError (not AssertionError) when model is None."""

    def test_nli_scorer_no_model_raises_runtime_error(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer(use_model=False)
        with pytest.raises(RuntimeError, match="NLI model not loaded"):
            scorer._model_score("premise", "hypothesis")

    def test_nli_scorer_heuristic_fallback(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer(use_model=False)
        result = scorer.score("The sky is blue", "consistent with reality")
        assert 0.0 <= result <= 1.0

    def test_nli_scorer_batch(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer(use_model=False)
        results = scorer.score_batch(
            [
                ("a", "consistent with reality"),
                ("b", "opposite is true"),
            ]
        )
        assert len(results) == 2
        assert results[0] < results[1]


# ── H29: batch asyncio.get_running_loop ──────────────────────────────


class TestH29AsyncLoop:
    """Batch async should use get_running_loop (not deprecated get_event_loop)."""

    def test_process_batch_async_uses_running_loop(self):
        import inspect

        from director_ai.core.batch import BatchProcessor

        source = inspect.getsource(BatchProcessor.process_batch_async)
        assert "get_running_loop" in source
        assert "get_event_loop" not in source


# ── H30: batch coherence None guard ──────────────────────────────────


class TestH30CoherenceNoneGuard:
    """Batch _process_one should not crash when coherence is None."""

    def test_process_one_none_coherence(self):
        from unittest.mock import MagicMock

        from director_ai.core.batch import BatchProcessor
        from director_ai.core.types import ReviewResult

        mock_backend = MagicMock()
        mock_backend.process.return_value = ReviewResult(
            output="test", halted=True, candidates_evaluated=1, coherence=None
        )
        proc = BatchProcessor(mock_backend)
        result = proc._process_one(0, "test")
        assert result.halted is True
        assert result.coherence is None


# ── H34: actor response.text truncation ─────────────────────────────


class TestH34ResponseTruncation:
    """LLMGenerator error log should truncate response.text to 500 chars."""

    def test_log_truncation_in_source(self):
        import inspect

        from director_ai.core.actor import LLMGenerator

        source = inspect.getsource(LLMGenerator.generate_candidates)
        assert "response.text[:500]" in source


# ── H35: config _coerce error message ───────────────────────────────


class TestH35CoerceError:
    """_coerce ValueError should name the offending env var."""

    def test_invalid_env_var_reports_key(self):
        from director_ai.core.config import DirectorConfig

        env = {"DIRECTOR_COHERENCE_THRESHOLD": "not_a_float"}
        original = os.environ.copy()
        try:
            os.environ.update(env)
            with pytest.raises(ValueError, match="DIRECTOR_COHERENCE_THRESHOLD"):
                DirectorConfig.from_env()
        finally:
            os.environ.clear()
            os.environ.update(original)


# ── H36: config server_port / server_workers validation ──────────────


class TestH36ServerValidation:
    """DirectorConfig should reject invalid server_port and server_workers."""

    def test_port_zero_rejected(self):
        from director_ai.core.config import DirectorConfig

        with pytest.raises(ValueError, match="server_port"):
            DirectorConfig(server_port=0)

    def test_port_65536_rejected(self):
        from director_ai.core.config import DirectorConfig

        with pytest.raises(ValueError, match="server_port"):
            DirectorConfig(server_port=65536)

    def test_valid_port_accepted(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(server_port=8080)
        assert cfg.server_port == 8080

    def test_workers_zero_rejected(self):
        from director_ai.core.config import DirectorConfig

        with pytest.raises(ValueError, match="server_workers"):
            DirectorConfig(server_workers=0)

    def test_workers_positive_accepted(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(server_workers=4)
        assert cfg.server_workers == 4


# ── H37: config from_yaml UTF-8 ─────────────────────────────────────


class TestH37YamlUtf8:
    """from_yaml should open files with encoding='utf-8'."""

    def test_yaml_utf8_encoding_in_source(self):
        import inspect

        from director_ai.core.config import DirectorConfig

        source = inspect.getsource(DirectorConfig.from_yaml)
        assert 'encoding="utf-8"' in source or "encoding='utf-8'" in source

    def test_yaml_with_unicode(self):
        from director_ai.core.config import DirectorConfig

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"profile": "default", "log_level": "DEBUG"}, f)
            f.flush()
            path = f.name

        try:
            cfg = DirectorConfig.from_yaml(path)
            assert cfg.log_level == "DEBUG"
        finally:
            os.unlink(path)


# ── H39: CLI --port safety ──────────────────────────────────────────


class TestH39CLIPort:
    """CLI --port should handle non-integer gracefully."""

    def test_cli_port_in_source(self):
        import inspect

        from director_ai.cli import _cmd_serve

        source = inspect.getsource(_cmd_serve)
        assert "ValueError" in source

    def test_cli_batch_utf8(self):
        import inspect

        from director_ai.cli import _cmd_batch

        source = inspect.getsource(_cmd_batch)
        assert 'encoding="utf-8"' in source or "encoding='utf-8'" in source


# ── H42: scorer history thread lock ─────────────────────────────────


class TestH42ScorerThreadLock:
    """CoherenceScorer should protect history mutations with a lock."""

    def test_scorer_has_lock(self):
        from director_ai.core.scorer import CoherenceScorer

        s = CoherenceScorer(use_nli=False)
        assert hasattr(s, "_history_lock")
        assert isinstance(s._history_lock, type(threading.Lock()))

    def test_concurrent_reviews(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        errors = []

        def review_many():
            try:
                for i in range(50):
                    scorer.review(f"prompt {i}", "consistent with reality")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=review_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(scorer.history) <= scorer.window


# ── H44: scorer setLevel removed ────────────────────────────────────


class TestH44ScorerSetLevel:
    """CoherenceScorer should not call setLevel on its logger."""

    def test_no_set_level_in_init(self):
        import inspect

        from director_ai.core.scorer import CoherenceScorer

        source = inspect.getsource(CoherenceScorer.__init__)
        assert "setLevel" not in source
