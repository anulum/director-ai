# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — zero-trust output handling

"""Zero-trust handling of untrusted LLM output (OWASP LLM05).

Model output is encoded for the specific :class:`OutputSink` it is about to enter
— HTML, a shell argument, a SQL identifier, a filesystem path, JSON, a URL query,
an email header, a log line — so the same string cannot be an XSS payload in one
context and a command-injection payload in another. :class:`ZeroTrustOutputGuard`
also flags constructs that must never be executed or deserialised, keeping
generated text as data by default.
"""

from .encoders import EncodeResult, UnsafeOutputError, encode_for_sink
from .guard import EncodedOutput, OutputExecutionRisk, ZeroTrustOutputGuard
from .sinks import OutputSink

__all__ = [
    "EncodeResult",
    "EncodedOutput",
    "OutputExecutionRisk",
    "OutputSink",
    "UnsafeOutputError",
    "ZeroTrustOutputGuard",
    "encode_for_sink",
]
