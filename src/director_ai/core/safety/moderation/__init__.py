# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — safety moderation package

"""Additional PII and toxicity detectors beyond regex-only
:class:`director_ai.enterprise.redactor.PIIRedactor`.

Two detector families live here:

* :mod:`.pii` — :class:`PresidioPIIDetector` (Microsoft Presidio NER
  plus recogniser rules) and :class:`RegexPIIDetector` (the
  dependency-free baseline, extended from the enterprise redactor's
  pattern list).
* :mod:`.toxicity` — :class:`DetoxifyDetector` (Unitary
  ``detoxify`` multilingual classifier) and
  :class:`KeywordToxicityDetector` (zero-dep seed list for smoke
  tests and environments where the model is not allowed).

All detectors implement :class:`ModerationDetector`; Policy accepts
an iterable of detectors and raises structured violations for any
match.
"""

from .detectors import (
    ModerationDetector,
    ModerationMatch,
    ModerationResult,
)
from .pii import PresidioPIIDetector, RegexPIIDetector
from .toxicity import DetoxifyDetector, KeywordToxicityDetector

__all__ = [
    "DetoxifyDetector",
    "KeywordToxicityDetector",
    "ModerationDetector",
    "ModerationMatch",
    "ModerationResult",
    "PresidioPIIDetector",
    "RegexPIIDetector",
]
