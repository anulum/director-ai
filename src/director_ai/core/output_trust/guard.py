# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Zero-trust output guard

"""Encode untrusted LLM output for its destination and refuse code execution.

:class:`ZeroTrustOutputGuard` is the single entry point for the OWASP-LLM05
posture: nothing the model produces is rendered, executed, or persisted raw.
``encode`` neutralises a value for a named :class:`~.sinks.OutputSink`; ``assess``
flags constructs that must never be handed to ``exec``/``eval`` or an unsandboxed
deserialiser, so a caller cannot accidentally treat generated text as code.

Both results are tenant-safe — they carry the sink, what was neutralised, and the
flagged construct categories, never the surrounding prompt or retrieval context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .encoders import UnsafeOutputError, encode_for_sink
from .sinks import OutputSink

__all__ = ["EncodedOutput", "OutputExecutionRisk", "ZeroTrustOutputGuard"]

# Constructs that should never reach a code/deserialisation sink unsandboxed.
_DANGEROUS_CONSTRUCTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dynamic_import", re.compile(r"\b__import__\s*\(")),
    ("eval_exec", re.compile(r"\b(?:eval|exec)\s*\(")),
    ("os_command", re.compile(r"\b(?:os\.system|os\.popen|subprocess\.\w+)\s*\(")),
    ("pickle_deserialise", re.compile(r"\b(?:pickle|cPickle)\.(?:loads?|Unpickler)\b")),
    ("yaml_unsafe_load", re.compile(r"\byaml\.(?:unsafe_load|load)\s*\(")),
    ("dunder_globals", re.compile(r"\b(?:globals|locals|getattr|setattr)\s*\(")),
    ("shell_pipe_redirect", re.compile(r"[;&|`$]\s*\w|\$\(")),
)


@dataclass(frozen=True)
class EncodedOutput:
    """The result of encoding one output for one sink."""

    sink: OutputSink
    encoded: str
    modified: bool
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialise to a tenant-safe JSON dict (no surrounding context)."""
        return {
            "sink": str(self.sink),
            "encoded": self.encoded,
            "modified": self.modified,
            "note": self.note,
        }


@dataclass(frozen=True)
class OutputExecutionRisk:
    """Whether an output is safe to hand to a code/deserialisation sink."""

    safe_to_execute: bool
    constructs: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a tenant-safe JSON dict (construct categories only)."""
        return {
            "safe_to_execute": self.safe_to_execute,
            "constructs": list(self.constructs),
        }


class ZeroTrustOutputGuard:
    """Encode output per destination and gate code execution of generated text."""

    def encode(self, text: str, sink: OutputSink | str) -> EncodedOutput:
        """Neutralise ``text`` for ``sink``.

        Raises :class:`~director_ai.core.output_trust.encoders.UnsafeOutputError`
        when the value cannot be made safe (invalid SQL identifier, traversing or
        absolute path, NUL byte), and ``ValueError`` for an unknown sink.
        """
        if not isinstance(text, str):
            raise TypeError("output to encode must be a string")
        sink_enum = sink if isinstance(sink, OutputSink) else OutputSink(str(sink))
        encoded, modified, note = encode_for_sink(text, sink_enum.value)
        return EncodedOutput(
            sink=sink_enum, encoded=encoded, modified=modified, note=note
        )

    def assess(self, text: str) -> OutputExecutionRisk:
        """Flag constructs that bar handing ``text`` to a code/deserialise sink.

        This never executes or transforms the text; it reports which dangerous
        construct categories appear. ``safe_to_execute`` is ``True`` only when no
        category matches — the default posture is that generated text is data,
        not code.
        """
        if not isinstance(text, str):
            raise TypeError("output to assess must be a string")
        found = tuple(
            name for name, pattern in _DANGEROUS_CONSTRUCTS if pattern.search(text)
        )
        return OutputExecutionRisk(safe_to_execute=not found, constructs=found)

    def encode_or_none(self, text: str, sink: OutputSink | str) -> EncodedOutput | None:
        """Return :meth:`encode` or ``None`` when the value cannot be made safe.

        For callers that prefer to drop an unsafe field rather than handle an
        :class:`UnsafeOutputError` per call site.
        """
        try:
            return self.encode(text, sink)
        except UnsafeOutputError:
            return None
