# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
    "DeepgramAdapter",
    "ElevenLabsAdapter",
    "OpenAITTSAdapter",
    "TTSAdapter",
    "VoiceToken",
    "voice_pipeline",
]
