# SPDX-License-Identifier: AGPL-3.0-or-later
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
    resolve_citations,
)

__all__ = [
    "Citation",
    "CitationKind",
    "extract_inline_citations",
    "parse_reference_section",
    "resolve_citations",
]
