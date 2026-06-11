# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Zero-trust output encoders

"""Per-sink encoders that neutralise untrusted output for one destination.

Each encoder returns ``(encoded, modified, note)``: the safe rendering, whether
anything had to be neutralised, and a short tenant-safe note describing what was
neutralised (or ``None``). Encoders that cannot make the input safe — an invalid
SQL identifier, a path that escapes its base — raise :class:`UnsafeOutputError`
rather than emit something that only looks safe.

All encoders use the standard library (``html``, ``shlex``, ``json``,
``urllib.parse``, ``posixpath``); there is no dependency on the data being valid
UTF-8 beyond what those modules already guarantee.
"""

from __future__ import annotations

import html
import json
import posixpath
import re
import shlex
import urllib.parse

__all__ = ["EncodeResult", "UnsafeOutputError", "encode_for_sink"]

# (encoded, modified, note)
EncodeResult = tuple[str, bool, str | None]

_SQL_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CONTROL_CHARS = {c for c in range(0x20)} | {0x7F}


class UnsafeOutputError(ValueError):
    """Raised when an output cannot be made safe for the requested sink.

    Distinct from neutralisation: an HTML or shell value can always be encoded,
    but an arbitrary SQL identifier or a traversing path cannot — surfacing the
    refusal is safer than emitting a plausible-looking but unsafe rendering.
    """


def _encode_html_text(text: str) -> EncodeResult:
    encoded = html.escape(text, quote=True)
    return encoded, encoded != text, "HTML metacharacters escaped"


def _encode_html_attribute(text: str) -> EncodeResult:
    # ``quote=True`` escapes both quote styles; the caller still wraps the value
    # in quotes. Without quoting an attribute is exploitable even when escaped.
    encoded = html.escape(text, quote=True)
    return encoded, encoded != text, "HTML attribute metacharacters escaped"


def _encode_shell_argument(text: str) -> EncodeResult:
    if "\x00" in text:
        raise UnsafeOutputError("shell arguments cannot contain a NUL byte")
    encoded = shlex.quote(text)
    return encoded, encoded != text, "wrapped as a single POSIX shell argument"


def _encode_sql_identifier(text: str) -> EncodeResult:
    if not _SQL_IDENTIFIER_RE.fullmatch(text):
        raise UnsafeOutputError(
            "SQL identifier must match [A-Za-z_][A-Za-z0-9_]* — validate, do not escape"
        )
    return text, False, None


def _encode_sql_string_literal(text: str) -> EncodeResult:
    if "\x00" in text:
        raise UnsafeOutputError("SQL string literals cannot contain a NUL byte")
    # ANSI single-quote doubling. Parameterised queries remain the correct
    # control; this is a defence-in-depth fallback for unavoidable interpolation.
    encoded = text.replace("'", "''")
    return encoded, encoded != text, "single quotes doubled — prefer a bound parameter"


def _encode_filesystem_path(text: str) -> EncodeResult:
    if "\x00" in text:
        raise UnsafeOutputError("paths cannot contain a NUL byte")
    if text.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", text):
        raise UnsafeOutputError("absolute paths are rejected; supply a relative path")
    normalised = posixpath.normpath(text.replace("\\", "/"))
    segments = normalised.split("/")
    if ".." in segments or normalised.startswith("../") or normalised == "..":
        raise UnsafeOutputError("path escapes its base directory ('..' segment)")
    if normalised == ".":
        raise UnsafeOutputError("path resolves to the base directory itself")
    return normalised, normalised != text, "normalised to a safe relative path"


def _encode_json_value(text: str) -> EncodeResult:
    encoded = json.dumps(text)
    # json.dumps always quotes/escapes; the result differs from the raw text.
    return encoded, True, "serialised as a JSON string literal"


def _encode_url_query(text: str) -> EncodeResult:
    encoded = urllib.parse.quote(text, safe="")
    return encoded, encoded != text, "percent-encoded for a URL query component"


def _strip_crlf(
    text: str, *, label: str, extra: set[int] | None = None
) -> EncodeResult:
    drop = {0x0D, 0x0A} | (extra or set())
    encoded = "".join(ch for ch in text if ord(ch) not in drop)
    modified = encoded != text
    return encoded, modified, (f"{label} neutralised" if modified else None)


def _encode_email_header(text: str) -> EncodeResult:
    return _strip_crlf(text, label="CR/LF header injection")


def _encode_log_line(text: str) -> EncodeResult:
    return _strip_crlf(text, label="control/CR-LF log injection", extra=_CONTROL_CHARS)


_ENCODERS = {
    "html_text": _encode_html_text,
    "html_attribute": _encode_html_attribute,
    "shell_argument": _encode_shell_argument,
    "sql_identifier": _encode_sql_identifier,
    "sql_string_literal": _encode_sql_string_literal,
    "filesystem_path": _encode_filesystem_path,
    "json_value": _encode_json_value,
    "url_query": _encode_url_query,
    "email_header": _encode_email_header,
    "log_line": _encode_log_line,
}


def encode_for_sink(text: str, sink: str) -> EncodeResult:
    """Encode ``text`` for ``sink`` (an :class:`~.sinks.OutputSink` value).

    Raises :class:`UnsafeOutputError` when the value cannot be neutralised for
    the sink, and :class:`KeyError`-free :class:`ValueError` for an unknown sink.
    """
    encoder = _ENCODERS.get(str(sink))
    if encoder is None:
        raise ValueError(f"unknown output sink: {sink!r}")
    return encoder(text)
