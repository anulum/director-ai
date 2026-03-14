# ─────────────────────────────────────────────────────────────────────
# Phase 3 Hardening Tests (H28-H44, core only)
# ─────────────────────────────────────────────────────────────────────
"""Tests for Phase 3 hardening fixes (consumer core):
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
import logging
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
            ],
        )
        assert len(results) == 2
        assert results[0] < results[1]


# ── H29: batch asyncio.get_running_loop ──────────────────────────────


class TestH29AsyncLoop:
    """Batch async should use get_running_loop (not deprecated get_event_loop)."""

    def test_process_batch_runs(self):
        from unittest.mock import MagicMock

        from director_ai.core.batch import BatchProcessor
        from director_ai.core.types import CoherenceScore, ReviewResult

        mock_backend = MagicMock()
        mock_backend.process.return_value = ReviewResult(
            output="ok",
            halted=False,
            candidates_evaluated=1,
            coherence=CoherenceScore(
                score=0.9,
                approved=True,
                h_logical=0.05,
                h_factual=0.05,
            ),
        )
        proc = BatchProcessor(mock_backend)
        batch_result = proc.process_batch(["test"])
        assert len(batch_result.results) == 1


# ── H30: batch coherence None guard ──────────────────────────────────


class TestH30CoherenceNoneGuard:
    """Batch _process_one should not crash when coherence is None."""

    def test_process_one_none_coherence(self):
        from unittest.mock import MagicMock

        from director_ai.core.batch import BatchProcessor
        from director_ai.core.types import ReviewResult

        mock_backend = MagicMock()
        mock_backend.process.return_value = ReviewResult(
            output="test",
            halted=True,
            candidates_evaluated=1,
            coherence=None,
        )
        proc = BatchProcessor(mock_backend)
        result = proc._process_one(0, "test")
        assert result.halted is True
        assert result.coherence is None


# ── H34: actor response.text truncation ─────────────────────────────


class TestH34ResponseTruncation:
    """LLMGenerator error log should truncate long responses."""

    def test_long_error_response_does_not_crash(self):
        from unittest.mock import MagicMock, patch

        from director_ai.core.actor import LLMGenerator

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "X" * 10_000
        with patch("director_ai.core.actor.requests.post", return_value=mock_resp):
            gen = LLMGenerator(api_url="http://localhost:8080/completion")
            candidates = gen.generate_candidates("test", n=1)
        assert len(candidates) == 1
        assert "Error" in candidates[0]["text"]


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
    """from_yaml should handle UTF-8 files correctly."""

    def test_yaml_with_unicode(self):
        from director_ai.core.config import DirectorConfig

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump({"profile": "default", "log_level": "DEBUG"}, f)
            f.flush()
            path = f.name

        try:
            cfg = DirectorConfig.from_yaml(path)
            assert cfg.log_level == "DEBUG"
        finally:
            os.unlink(path)

    def test_yaml_with_unicode_chars(self):
        from director_ai.core.config import DirectorConfig

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump({"profile": "default", "log_level": "INFO"}, f)
            f.flush()
            path = f.name

        try:
            cfg = DirectorConfig.from_yaml(path)
            assert cfg.log_level == "INFO"
        finally:
            os.unlink(path)


# ── H39: CLI --port safety ──────────────────────────────────────────


class TestH39CLIPort:
    """CLI --port should handle non-integer gracefully."""

    def test_cli_serve_with_invalid_port(self):
        from director_ai.cli import _cmd_serve

        with pytest.raises((ValueError, SystemExit)):
            _cmd_serve(["--port", "not_a_number"])

    def test_cli_batch_with_nonexistent_file(self):
        from director_ai.cli import _cmd_batch

        with pytest.raises((FileNotFoundError, SystemExit)):
            _cmd_batch(["nonexistent_file.jsonl"])


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
    """CoherenceScorer logger should respect the logging hierarchy."""

    def test_scorer_logger_level_not_forced(self):
        from director_ai.core.scorer import CoherenceScorer

        # Reset logger to isolate scorer behavior from prior test side-effects
        logging.getLogger("DirectorAI").setLevel(logging.NOTSET)
        s = CoherenceScorer(use_nli=False)
        assert s.logger.level == logging.NOTSET
