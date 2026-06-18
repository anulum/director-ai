# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Differentially Private RAG
"""Expose differentially private RAG retrieval, decoding, and accounting.

The package provides retrieval ranking, token decoding, and a unified
per-tenant privacy budget across the pipeline.
"""

from __future__ import annotations

from .decoding import DPTokenChoice, DPTokenDecoder
from .pipeline import DPRagPipeline, PipelineRanking, StageCharge
from .retrieval import (
    DifferentiallyPrivateRetrieval,
    DPBudgetExceededError,
    PrivateRanking,
    ScoredItem,
)

__all__ = [
    "DPBudgetExceededError",
    "DPRagPipeline",
    "DPTokenChoice",
    "DPTokenDecoder",
    "DifferentiallyPrivateRetrieval",
    "PipelineRanking",
    "PrivateRanking",
    "ScoredItem",
    "StageCharge",
]
