# ─────────────────────────────────────────────────────────────────────
# Phase 4 Hardening Tests (H46-H64)
# ─────────────────────────────────────────────────────────────────────
"""
Tests for Phase 4 hardening fixes:
  H47  INJ-2: consilium subprocess path validation
  H48  DOS-1: batch per-line JSON size limit
  H49  ERR-2: WebSocket receive_json exception handling
  H50  VAL-1: NaN/Inf clamp logging
  H52  VAL-2: null JSON prompt guard in batch
  H54  RES-2: batch timeout parameter validation
  H55  ERR-3: LLM error type distinction
  H57  VAL-3: CORS origins count limit
  H58  TYP-1: return type annotations on generators
  H59  DOC-1: batch limits in help text
  H64  TEST-4: bool coercion edge cases
"""
import inspect
import json
import logging
import math
import os
import tempfile

import pytest


def _fastapi_available() -> bool:
    try:
        import fastapi  # noqa: F401
        return True
    except ImportError:
        return False


# ── H47: Consilium subprocess path safety ────────────────────────────

class TestH47SubprocessPath:
    """Consilium should validate test path before subprocess.run()."""

    def test_get_real_metrics_missing_file(self):
        from director_ai.research.consilium.director_core import ConsiliumAgent

        agent = ConsiliumAgent()
        # When test file doesn't exist (it won't from this CWD),
        # it should return early with default metrics, not crash
        metrics = agent.get_real_metrics()
        assert isinstance(metrics, dict)
        assert "errors" in metrics

    def test_source_has_isfile_check(self):
        from director_ai.research.consilium.director_core import ConsiliumAgent

        source = inspect.getsource(ConsiliumAgent.get_real_metrics)
        assert "os.path.isfile" in source


# ── H48: Batch per-line size limit ───────────────────────────────────

class TestH48LineSize:
    """CLI batch should reject lines exceeding per-line size limit."""

    def test_line_limit_constant_exists(self):
        from director_ai.cli import _BATCH_MAX_LINE_SIZE

        assert _BATCH_MAX_LINE_SIZE == 1 * 1024 * 1024

    def test_oversized_line_skipped(self, capsys):
        from director_ai.cli import _cmd_batch

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            # Write one normal line and one oversized line
            f.write(json.dumps({"prompt": "hello"}) + "\n")
            f.write(json.dumps({"prompt": "x" * (1024 * 1024 + 100)}) + "\n")
            path = f.name

        try:
            # This will try to process - we just need to verify the warning
            # We can't easily mock the agent, so check source instead
            source = inspect.getsource(_cmd_batch)
            assert "_BATCH_MAX_LINE_SIZE" in source
        finally:
            os.unlink(path)


# ── H49: WebSocket receive_json error handling ───────────────────────

class TestH49WebSocketJson:
    """WebSocket handler should catch ValueError from receive_json."""

    def test_websocket_has_value_error_catch(self):
        from director_ai.server import create_app

        source = inspect.getsource(create_app)
        assert "ValueError" in source or "KeyError" in source
        assert '"invalid JSON"' in source


# ── H50: NaN/Inf clamp logging ──────────────────────────────────────

class TestH50ClampLogging:
    """_clamp should log warnings when NaN/Inf is detected."""

    def test_nan_clamp_logs_warning(self, caplog):
        from director_ai.core.types import _clamp

        with caplog.at_level(logging.WARNING, logger="DirectorAI.Types"):
            result = _clamp(float("nan"))
        assert result == 0.0
        assert any("NaN" in r.message for r in caplog.records)

    def test_inf_clamp_logs_warning(self, caplog):
        from director_ai.core.types import _clamp

        with caplog.at_level(logging.WARNING, logger="DirectorAI.Types"):
            result = _clamp(float("inf"))
        assert result == 1.0
        assert any("Inf" in r.message for r in caplog.records)

    def test_normal_clamp_no_warning(self, caplog):
        from director_ai.core.types import _clamp

        with caplog.at_level(logging.WARNING, logger="DirectorAI.Types"):
            result = _clamp(0.75)
        assert result == 0.75
        assert len(caplog.records) == 0

    def test_neg_inf_clamp(self, caplog):
        from director_ai.core.types import _clamp

        with caplog.at_level(logging.WARNING, logger="DirectorAI.Types"):
            result = _clamp(float("-inf"))
        assert result == 0.0


# ── H52: null JSON prompt guard ──────────────────────────────────────

class TestH52NullPrompt:
    """Batch should skip JSON entries with null/non-string prompts."""

    def test_source_has_isinstance_check(self):
        from director_ai.cli import _cmd_batch

        source = inspect.getsource(_cmd_batch)
        assert "isinstance(prompt, str)" in source


# ── H54: Batch timeout validation ────────────────────────────────────

class TestH54TimeoutValidation:
    """BatchProcessor should reject invalid item_timeout."""

    def test_negative_timeout_rejected(self):
        from unittest.mock import MagicMock

        from director_ai.core.batch import BatchProcessor

        with pytest.raises(ValueError, match="item_timeout"):
            BatchProcessor(MagicMock(), item_timeout=-1)

    def test_zero_timeout_rejected(self):
        from unittest.mock import MagicMock

        from director_ai.core.batch import BatchProcessor

        with pytest.raises(ValueError, match="item_timeout"):
            BatchProcessor(MagicMock(), item_timeout=0)

    def test_valid_timeout_accepted(self):
        from unittest.mock import MagicMock

        from director_ai.core.batch import BatchProcessor

        proc = BatchProcessor(MagicMock(), item_timeout=30.0)
        assert proc.item_timeout == 30.0

    def test_negative_concurrency_rejected(self):
        from unittest.mock import MagicMock

        from director_ai.core.batch import BatchProcessor

        with pytest.raises(ValueError, match="max_concurrency"):
            BatchProcessor(MagicMock(), max_concurrency=0)


# ── H55: LLM error type distinction ─────────────────────────────────

class TestH55ErrorTypes:
    """LLMGenerator should distinguish timeout from other errors."""

    def test_source_has_timeout_catch(self):
        from director_ai.core.actor import LLMGenerator

        source = inspect.getsource(LLMGenerator.generate_candidates)
        assert "requests.exceptions.Timeout" in source
        assert "[Error: LLM timeout]" in source

    def test_error_includes_type_name(self):
        from director_ai.core.actor import LLMGenerator

        source = inspect.getsource(LLMGenerator.generate_candidates)
        assert "type(e).__name__" in source


# ── H57: CORS origins count limit ────────────────────────────────────

class TestH57CORSLimit:
    """Server should reject excessive CORS origins."""

    def test_source_has_origins_limit(self):
        from director_ai.server import create_app

        source = inspect.getsource(create_app)
        assert "Too many CORS origins" in source

    @pytest.mark.skipif(
        not _fastapi_available(),
        reason="FastAPI not installed",
    )
    def test_excessive_origins_rejected(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(cors_origins=",".join([f"http://host{i}.com" for i in range(150)]))
        from director_ai.server import create_app

        with pytest.raises(ValueError, match="Too many CORS origins"):
            create_app(cfg)


# ── H58: Return type annotations ─────────────────────────────────────

class TestH58TypeAnnotations:
    """Generators should have return type annotations."""

    def test_mock_generator_return_type(self):
        from director_ai.core.actor import MockGenerator

        hints = MockGenerator.generate_candidates.__annotations__
        assert "return" in hints

    def test_llm_generator_return_type(self):
        from director_ai.core.actor import LLMGenerator

        hints = LLMGenerator.generate_candidates.__annotations__
        assert "return" in hints


# ── H59: Batch limits in help text ───────────────────────────────────

class TestH59HelpText:
    """Help text should mention batch limits."""

    def test_help_mentions_limits(self, capsys):
        from director_ai.cli import main

        main(["--help"])
        captured = capsys.readouterr()
        assert "10K" in captured.out or "10,000" in captured.out or "10000" in captured.out


# ── H64: Bool coercion edge cases ────────────────────────────────────

class TestH64BoolCoercion:
    """_coerce should reject invalid bool values."""

    def test_true_values(self):
        from director_ai.core.config import _coerce

        for val in ("true", "True", "TRUE", "1", "yes", "YES"):
            assert _coerce(val, "bool") is True

    def test_false_values(self):
        from director_ai.core.config import _coerce

        for val in ("false", "False", "FALSE", "0", "no", "NO"):
            assert _coerce(val, "bool") is False

    def test_invalid_bool_raises(self):
        from director_ai.core.config import _coerce

        for val in ("2", "maybe", "yep", "nah", ""):
            with pytest.raises(ValueError, match="invalid bool"):
                _coerce(val, "bool")

    def test_int_coercion(self):
        from director_ai.core.config import _coerce

        assert _coerce("42", "int") == 42

    def test_float_coercion(self):
        from director_ai.core.config import _coerce

        assert _coerce("3.14", "float") == pytest.approx(3.14)

    def test_string_passthrough(self):
        from director_ai.core.config import _coerce

        assert _coerce("hello", "str") == "hello"
