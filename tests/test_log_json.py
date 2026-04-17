# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — JSON Logging Tests
"""Multi-angle tests for JSON structured logging.

Covers: JSON format validity, field presence (level, logger, msg, ts),
request_id inclusion/exclusion, log level configuration, handler setup,
parametrised log levels, various message formats, and pipeline
performance documentation.
"""

from __future__ import annotations

import json
import logging

import pytest

from director_ai.core.config import DirectorConfig, _JsonFormatter


@pytest.fixture(autouse=True)
def _cleanup_logger():
    yield
    logger = logging.getLogger("DirectorAI")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)


# ── JSON formatter ───────────────────────────────────────────────


class TestJsonFormatter:
    """_JsonFormatter must produce valid, complete JSON lines."""

    def _make_record(
        self,
        *,
        name="DirectorAI.Test",
        level=logging.INFO,
        msg="hello %s",
        args=("world",),
        **kwargs,
    ):
        record = logging.LogRecord(
            name=name,
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=args,
            exc_info=None,
        )
        for k, v in kwargs.items():
            setattr(record, k, v)
        return record

    def test_produces_valid_json(self):
        fmt = _JsonFormatter()
        output = fmt.format(self._make_record())
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_required_fields_present(self):
        fmt = _JsonFormatter()
        parsed = json.loads(fmt.format(self._make_record()))
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "DirectorAI.Test"
        assert parsed["msg"] == "hello world"
        assert "ts" in parsed

    @pytest.mark.parametrize(
        "level,expected",
        [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ],
    )
    def test_various_log_levels(self, level, expected):
        fmt = _JsonFormatter()
        parsed = json.loads(fmt.format(self._make_record(level=level)))
        assert parsed["level"] == expected

    def test_includes_request_id(self):
        fmt = _JsonFormatter()
        parsed = json.loads(fmt.format(self._make_record(request_id="abc-123")))
        assert parsed["request_id"] == "abc-123"

    def test_omits_request_id_when_absent(self):
        fmt = _JsonFormatter()
        parsed = json.loads(fmt.format(self._make_record()))
        assert "request_id" not in parsed

    @pytest.mark.parametrize(
        "msg,args,expected",
        [
            ("simple", (), "simple"),
            ("hello %s", ("world",), "hello world"),
            ("count: %d", (42,), "count: 42"),
            ("", (), ""),
        ],
    )
    def test_various_message_formats(self, msg, args, expected):
        fmt = _JsonFormatter()
        parsed = json.loads(fmt.format(self._make_record(msg=msg, args=args)))
        assert parsed["msg"] == expected

    def test_unicode_message(self):
        fmt = _JsonFormatter()
        parsed = json.loads(
            fmt.format(self._make_record(msg="日本語テスト %s", args=("🎉",)))
        )
        assert "日本語" in parsed["msg"]

    def test_timestamp_present(self):
        fmt = _JsonFormatter()
        parsed = json.loads(fmt.format(self._make_record()))
        assert "ts" in parsed


# ── Configure logging ────────────────────────────────────────────


class TestConfigureLogging:
    """DirectorConfig.configure_logging() must set up handlers correctly."""

    def test_log_json_sets_handler(self):
        cfg = DirectorConfig(log_json=True)
        cfg.configure_logging()
        logger = logging.getLogger("DirectorAI")
        assert len(logger.handlers) >= 1
        assert isinstance(logger.handlers[0].formatter, _JsonFormatter)

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_log_level_applied(self, level):
        cfg = DirectorConfig(log_level=level)
        cfg.configure_logging()
        logger = logging.getLogger("DirectorAI")
        assert logger.level == getattr(logging, level)

    def test_default_config_no_json(self):
        cfg = DirectorConfig(log_json=False)
        cfg.configure_logging()
        logger = logging.getLogger("DirectorAI")
        json_handlers = [
            h for h in logger.handlers if isinstance(h.formatter, _JsonFormatter)
        ]
        assert len(json_handlers) == 0


# ── Pipeline performance ────────────────────────────────────────


class TestLoggingPerformance:
    """Document logging pipeline characteristics."""

    def test_json_formatter_fast(self):
        import time

        fmt = _JsonFormatter()
        record = logging.LogRecord(
            "DirectorAI",
            logging.INFO,
            "test.py",
            1,
            "perf test",
            (),
            None,
        )
        t0 = time.perf_counter()
        for _ in range(10000):
            fmt.format(record)
        per_call_us = (time.perf_counter() - t0) / 10000 * 1_000_000
        assert per_call_us < 100, (
            f"JSON format took {per_call_us:.1f}µs (expected <100µs)"
        )
