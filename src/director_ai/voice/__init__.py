# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Voice AI Subpackage
"""Async voice AI streaming guard with TTS adapter integration.

::

    from director_ai.voice import AsyncVoiceGuard, voice_pipeline, ElevenLabsAdapter
"""

from director_ai.integrations.voice import VoiceToken

from .adapters import DeepgramAdapter, ElevenLabsAdapter, OpenAITTSAdapter, TTSAdapter
from .guard import AsyncVoiceGuard
from .pipeline import voice_pipeline

__all__ = [
    "AsyncVoiceGuard",
    "DryRunTTSAdapter",
    "DeepgramAdapter",
    "ElevenLabsAdapter",
    "OpenAITTSAdapter",
    "TTSAdapter",
    "VoiceToken",
    "VoiceDemoResult",
    "run_voice_demo",
    "scripted_tokens",
    "voice_pipeline",
]

_DEMO_EXPORTS = {
    "DryRunTTSAdapter",
    "VoiceDemoResult",
    "run_voice_demo",
    "scripted_tokens",
}


def __getattr__(name: str) -> object:
    if name in _DEMO_EXPORTS:
        from . import demo

        return getattr(demo, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
