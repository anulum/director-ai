# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# safety subpackage

"""Safety primitives for prompt screening, injection detection, and harm mapping."""

from .harm_taxonomy import HarmCategory, to_harm_category
from .injection import InjectionDetector
from .prompt_guard import (
    DEFAULT_PROMPT_GUARD_MODEL,
    LayeredPromptGuard,
    PromptInjectionModel,
    PromptScreenResult,
)
from .sanitizer import InputSanitizer, SanitizeResult

__all__ = [
    "DEFAULT_PROMPT_GUARD_MODEL",
    "HarmCategory",
    "InjectionDetector",
    "InputSanitizer",
    "LayeredPromptGuard",
    "PromptInjectionModel",
    "PromptScreenResult",
    "SanitizeResult",
    "to_harm_category",
]
