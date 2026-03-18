# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” JSON Logging Tests

import json
import logging

from director_ai.core.config import DirectorConfig, _JsonFormatter


class TestJsonFormatter:
    def test_format_produces_valid_json(self):
        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="DirectorAI.Test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "DirectorAI.Test"
        assert parsed["msg"] == "hello world"
        assert "ts" in parsed

    def test_format_includes_request_id(self):
        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="DirectorAI",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.request_id = "abc-123"
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "abc-123"

    def test_format_omits_request_id_when_absent(self):
        fmt = _JsonFormatter()
        record = logging.LogRecord(
            name="DirectorAI",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "request_id" not in parsed


class TestConfigureLogging:
    def test_log_json_sets_handler(self):
        cfg = DirectorConfig(log_json=True)
        cfg.configure_logging()
        logger = logging.getLogger("DirectorAI")
        assert len(logger.handlers) >= 1
        assert isinstance(logger.handlers[0].formatter, _JsonFormatter)
        # Cleanup
        logger.handlers.clear()

    def test_log_level_applied(self):
        cfg = DirectorConfig(log_level="DEBUG")
        cfg.configure_logging()
        logger = logging.getLogger("DirectorAI")
        assert logger.level == logging.DEBUG
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def test_default_config_no_json(self):
        cfg = DirectorConfig(log_json=False)
        logger = logging.getLogger("DirectorAI")
        cfg.configure_logging()
        # Should not add JSON handler
        json_handlers = [
            h for h in logger.handlers if isinstance(h.formatter, _JsonFormatter)
        ]
        assert len(json_handlers) == 0
        logger.handlers.clear()
