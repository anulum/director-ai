# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — structured logging formatter for configuration

"""Structured JSON log formatter applied by ``DirectorConfig.configure_logging``.

Split out of ``config.py``: the log record serialisation is a distinct concern
from the configuration dataclass. ``DirectorConfig.configure_logging`` installs
``JsonLogFormatter`` on the ``DirectorAI`` logger hierarchy when ``log_json`` is
set.
"""

from __future__ import annotations

import json
import logging
import time

__all__ = ["JsonLogFormatter"]


class JsonLogFormatter(logging.Formatter):
    """Structured JSON log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a single-line JSON object."""
        entry = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            entry["request_id"] = request_id
        return json.dumps(entry)
