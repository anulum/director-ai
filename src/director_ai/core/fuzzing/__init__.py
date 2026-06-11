# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — continuous guard fuzzing

"""Continuous, mutation-based fuzzing of guard logic.

Where the static adversarial suite checks a fixed list, :class:`ContinuousFuzzer`
mutates a seed corpus of attacks round after round and surfaces the obfuscation a
guard fails to flag — a homoglyph, zero-width, leetspeak, or delimiter variant
that slips past keyword/pattern matching while keeping the malicious intent. The
seeded RNG makes every bypass replayable as a regression case.
"""

from .corpus import DEFAULT_ATTACK_CORPUS
from .engine import Bypass, ContinuousFuzzer, FuzzReport
from .mutators import MUTATORS, Mutator

__all__ = [
    "DEFAULT_ATTACK_CORPUS",
    "MUTATORS",
    "Bypass",
    "ContinuousFuzzer",
    "FuzzReport",
    "Mutator",
]
