# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Director-Lite package facade
"""Typed facade for the ``director-ai-lite`` distribution.

The separate PyPI package keeps the first-run install surface easy to find while
delegating all runtime behavior to :mod:`director_ai.lite`. That preserves one
canonical implementation for token-level streaming halt: the full
``director-ai`` package, the package-root exports, and this Lite package all run
the same ``StreamGuard`` and ``StreamingKernel`` code path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from director_ai.core import StreamSession
from director_ai.lite import StreamGuard, streaming_guard

__version__ = "3.15.3"


def guard(
    tokens: Iterable[str],
    *,
    facts: Mapping[str, str] | None = None,
    prompt: str = "",
    threshold: float = 0.5,
    scorer: Any = None,
) -> StreamSession:
    """Guard ``tokens`` with the canonical Director-Lite streaming halt.

    Parameters
    ----------
    tokens:
        Iterable of string tokens from an LLM streaming response.
    facts:
        Optional mapping of source identifiers to grounded statements. These are
        loaded into the heuristic scorer when no explicit scorer is supplied.
    prompt:
        User prompt or task context attached to the stream session.
    threshold:
        Coherence floor. Generation halts when a reviewed prefix scores below
        this value.
    scorer:
        Optional preconfigured Director-AI scorer. Passing one upgrades the same
        API to model-backed scoring without changing the call site.
    """
    return streaming_guard(
        tokens,
        facts=facts,
        prompt=prompt,
        threshold=threshold,
        scorer=scorer,
    )


__all__ = ["StreamGuard", "__version__", "guard", "streaming_guard"]
