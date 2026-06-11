# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# citation grounding subpackage

"""Citation extraction and grounding for HalluHard-style answer verification.

Splits an answer into its factual assertions and the sources they cite, so the
NLI scorer can check whether the cited material actually supports each claim —
the operationalisation of groundedness HalluHard uses.
"""

from .citations import (
    Citation,
    CitationKind,
    extract_inline_citations,
    parse_reference_section,
    reference_section_start,
    resolve_citations,
)
from .fetch import FetchedSource, HttpGetter, SourceFetcher
from .judge import (
    CitationGroundingJudge,
    ClaimGrounding,
    GroundingReport,
    Scorer,
)
from .transcript import (
    DEFAULT_SYSTEM_PROMPT,
    ExchangeTurn,
    Generator,
    MultiTurnRunner,
    Transcript,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "Citation",
    "CitationGroundingJudge",
    "CitationKind",
    "ClaimGrounding",
    "ExchangeTurn",
    "FetchedSource",
    "Generator",
    "GroundingReport",
    "HttpGetter",
    "MultiTurnRunner",
    "Scorer",
    "SourceFetcher",
    "Transcript",
    "extract_inline_citations",
    "parse_reference_section",
    "reference_section_start",
    "resolve_citations",
]
