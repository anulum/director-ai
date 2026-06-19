# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# retrieval subpackage

"""Grounded retrieval: vector stores, ingestion plugins, chunking, and guards."""

from .conflict_guard import (
    ConflictAwareKnowledgeGuard,
    KnowledgeConflict,
    KnowledgeConflictCheck,
    KnowledgeFact,
)

__all__ = [
    "ConflictAwareKnowledgeGuard",
    "KnowledgeConflict",
    "KnowledgeConflictCheck",
    "KnowledgeFact",
]
