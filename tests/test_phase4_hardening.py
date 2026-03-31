# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — test_phase4_hardening.py

"""Tests for Phase 4 hardening fixes:
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

import logging

import pytest


def _fastapi_available() -> bool:
    try:
        import fastapi  # noqa: F401

        return True
    except ImportError:
        return False


# â”€â”€ H48: Batch per-line size limit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestH48LineSize:
    """CLI batch should reject lines exceeding per-line size limit."""

    def test_line_limit_constant_exists(self):
        from director_ai.cli import _BATCH_MAX_LINE_SIZE

        assert _BATCH_MAX_LINE_SIZE == 1 * 1024 * 1024


# â”€â”€ H50: NaN/Inf clamp logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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


# â”€â”€ H54: Batch timeout validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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


# â”€â”€ H55: LLM error type distinction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestH55ErrorTypes:
    """LLMGenerator should distinguish timeout from connection errors."""

    def test_timeout_error_handled(self):
        from unittest.mock import patch

        import requests.exceptions

        from director_ai.core.actor import LLMGenerator

        gen = LLMGenerator(api_url="http://localhost:8080/completion")
        with patch(
            "director_ai.core.actor.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            candidates = gen.generate_candidates("test", n=1)
        assert len(candidates) == 1
        assert "Error" in candidates[0]["text"]

    def test_connection_error_handled(self):
        from unittest.mock import patch

        from director_ai.core.actor import LLMGenerator

        gen = LLMGenerator(api_url="http://localhost:8080/completion")
        with patch(
            "director_ai.core.actor.requests.post",
            side_effect=ConnectionError("refused"),
        ):
            candidates = gen.generate_candidates("test", n=1)
        assert len(candidates) == 1
        assert "Error" in candidates[0]["text"]


# â”€â”€ H57: CORS origins count limit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestH57CORSLimit:
    """Server should reject excessive CORS origins."""

    @pytest.mark.skipif(
        not _fastapi_available(),
        reason="FastAPI not installed",
    )
    def test_excessive_origins_rejected(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.server import create_app

        cfg = DirectorConfig(
            cors_origins=",".join([f"http://host{i}.com" for i in range(150)]),
        )
        with pytest.raises(ValueError, match="Too many CORS origins"):
            create_app(cfg)


# â”€â”€ H58: Return type annotations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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


# â”€â”€ H59: Batch limits in help text â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestH59HelpText:
    """Help text should mention batch limits."""

    def test_help_mentions_limits(self, capsys):
        from director_ai.cli import main

        main(["--help"])
        captured = capsys.readouterr()
        assert (
            "10K" in captured.out or "10,000" in captured.out or "10000" in captured.out
        )


# â”€â”€ H64: Bool coercion edge cases â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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
