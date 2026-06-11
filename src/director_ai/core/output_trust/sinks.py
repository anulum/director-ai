# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Zero-trust output sinks

"""The downstream contexts an LLM output may flow into.

Zero-trust output handling treats every model output as untrusted until it has
been encoded for the specific context it is about to enter. The danger is not the
text itself but the *sink*: the same string is harmless as JSON, an XSS vector in
HTML, a command-injection vector in a shell, and a path-traversal vector on a
filesystem. :class:`OutputSink` enumerates the sinks the encoder knows how to
neutralise so a caller names the destination explicitly rather than guessing.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["OutputSink"]


class OutputSink(StrEnum):
    """A downstream context an LLM output is about to be placed into."""

    HTML_TEXT = "html_text"
    """Rendered as HTML body text (escape ``& < > " '``)."""

    HTML_ATTRIBUTE = "html_attribute"
    """Interpolated inside a quoted HTML attribute value."""

    SHELL_ARGUMENT = "shell_argument"
    """A single argument in a shell command line (POSIX quoting)."""

    SQL_IDENTIFIER = "sql_identifier"
    """A table/column identifier — validated against an allowlist, never escaped."""

    SQL_STRING_LITERAL = "sql_string_literal"
    """A string literal in SQL — parameterised queries are still strongly preferred."""

    FILESYSTEM_PATH = "filesystem_path"
    """A relative path under a base directory (traversal/absolute rejected)."""

    JSON_VALUE = "json_value"
    """A value serialised into a JSON document."""

    URL_QUERY = "url_query"
    """A component of a URL query string."""

    EMAIL_HEADER = "email_header"
    """A value placed in an email header (CR/LF header injection neutralised)."""

    LOG_LINE = "log_line"
    """A field written to a line-oriented log (CR/LF + control chars neutralised)."""
