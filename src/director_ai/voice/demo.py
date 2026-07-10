# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Credential-Free Voice Pipeline Demo
"""Deterministic voice-pipeline demo for local validation and docs.

The demo uses the production :func:`voice_pipeline` with a dry-run TTS
adapter, so it exercises the guard, sentence buffering, halt callback,
recovery path, and adapter lifecycle without external services or
credentials.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from typing import cast

from director_ai.integrations.voice import VoiceToken

from .adapters import TTSAdapter
from .pipeline import voice_pipeline

__all__ = [
    "DEFAULT_FACTS",
    "DEFAULT_PROMPT",
    "DEFAULT_RESPONSE",
    "DryRunTTSAdapter",
    "VoiceDemoResult",
    "main",
    "run_voice_demo",
    "scripted_tokens",
]

DEFAULT_FACTS = {
    "pricing": "The Team plan costs CHF 19 per seat per month.",
    "trial": "The Team plan includes a 14-day trial.",
    "support": "Phone support is available only on the Enterprise plan.",
}
DEFAULT_PROMPT = "What does the Team plan include?"
DEFAULT_RESPONSE = (
    "The Team plan costs CHF 19 per seat per month. It includes a 14-day trial."
)


@dataclass(frozen=True)
class VoiceDemoResult:
    """Summary returned by :func:`run_voice_demo`."""

    audio_chunks: list[bytes]
    tts_texts: list[str]
    halted: bool
    halt_reason: str = ""
    recovery_text: str = ""

    @property
    def total_audio_bytes(self) -> int:
        """Total number of dry-run audio bytes emitted by the demo."""
        return sum(len(chunk) for chunk in self.audio_chunks)


@dataclass
class DryRunTTSAdapter(TTSAdapter):
    """TTS adapter that records text and yields deterministic audio bytes."""

    prefix: bytes = b"audio:"
    texts: list[str] = field(default_factory=list)
    closed: bool = False

    async def synthesise(self, text: str) -> AsyncIterator[bytes]:
        """Record text and emit deterministic dry-run audio bytes."""
        self.texts.append(text)
        yield self.prefix + text.encode("utf-8")

    async def close(self) -> None:
        """Mark the dry-run adapter as closed."""
        self.closed = True


async def scripted_tokens(text: str, *, separator: str = " ") -> AsyncIterator[str]:
    """Yield deterministic token-like fragments from text.

    The default keeps whitespace attached to each token so sentence buffering
    sees realistic fragments while remaining stable across Python versions.
    """
    parts = text.split(separator)
    for index, part in enumerate(parts):
        suffix = separator if index < len(parts) - 1 else ""
        yield part + suffix


async def run_voice_demo(
    *,
    tokens: AsyncIterator[str] | Iterable[str] | None = None,
    facts: dict[str, str] | None = None,
    prompt: str = DEFAULT_PROMPT,
    tts: DryRunTTSAdapter | None = None,
    threshold: float = 0.3,
    hard_limit: float = 0.25,
    score_every: int = 4,
    soft_halt: bool = True,
    recovery: str = "Let me verify that before speaking.",
    use_nli: bool = False,
) -> VoiceDemoResult:
    """Run the package voice demo against the production async pipeline.

    Parameters mirror :func:`voice_pipeline`; defaults are credential-free and
    deterministic so the demo is suitable for CI and quick local validation.
    """
    adapter = tts or DryRunTTSAdapter()
    source_input = tokens if tokens is not None else scripted_tokens(DEFAULT_RESPONSE)
    source: AsyncIterator[str] | Iterator[str]
    if hasattr(source_input, "__aiter__"):
        source = cast(AsyncIterator[str], source_input)
    else:
        source = iter(source_input)
    halt_event: VoiceToken | None = None

    def on_halt(vtoken: VoiceToken) -> None:
        """Capture the first guard halt event for the demo result."""
        nonlocal halt_event
        halt_event = vtoken

    audio_chunks = [
        chunk
        async for chunk in voice_pipeline(
            source,
            adapter,
            facts=DEFAULT_FACTS if facts is None else facts,
            prompt=prompt,
            threshold=threshold,
            hard_limit=hard_limit,
            score_every=score_every,
            soft_halt=soft_halt,
            recovery=recovery,
            use_nli=use_nli,
            on_halt=on_halt,
        )
    ]

    return VoiceDemoResult(
        audio_chunks=audio_chunks,
        tts_texts=list(adapter.texts),
        halted=halt_event is not None,
        halt_reason=halt_event.halt_reason if halt_event else "",
        recovery_text=halt_event.recovery_text if halt_event else "",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the credential-free demo CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run a credential-free Director-AI voice pipeline demo.",
    )
    parser.add_argument(
        "--response",
        default=DEFAULT_RESPONSE,
        help="Scripted response text to stream through the guard.",
    )
    parser.add_argument(
        "--score-every",
        type=int,
        default=4,
        help="Run the guard every N token fragments.",
    )
    parser.add_argument(
        "--hard-limit",
        type=float,
        default=0.25,
        help="Immediate halt threshold.",
    )
    return parser


async def _amain(argv: list[str] | None = None) -> int:
    """Run the async implementation for the voice demo CLI."""
    args = _build_parser().parse_args(argv)
    result = await run_voice_demo(
        tokens=scripted_tokens(args.response),
        score_every=args.score_every,
        hard_limit=args.hard_limit,
    )
    status = "halted" if result.halted else "completed"
    print(f"status={status}")
    if result.halt_reason:
        print(f"halt_reason={result.halt_reason}")
    print(f"tts_calls={len(result.tts_texts)}")
    print(f"audio_chunks={len(result.audio_chunks)}")
    print(f"audio_bytes={result.total_audio_bytes}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m director_ai.voice.demo``."""
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
