# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Semantic Kernel real-surface tests
"""Real async filter coverage for the Semantic Kernel adapter."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from director_ai.integrations.semantic_kernel import DirectorAIFilter


@dataclass
class _KernelInvocationContext:
    """Small Semantic-Kernel-shaped context used by the public filter hook."""

    result: object | None = None
    arguments: dict[str, str] = field(default_factory=dict)


@pytest.mark.asyncio
async def test_filter_allows_real_scorer_approved_result() -> None:
    """The public filter hook should preserve an approved invocation result."""
    context = _KernelInvocationContext(
        arguments={"input": "What does the team plan cost?"}
    )
    calls: list[str] = []

    async def next_function(invocation: _KernelInvocationContext) -> None:
        calls.append("next")
        invocation.result = "Team plan costs CHF 19 per user per month."

    filter_hook = DirectorAIFilter(
        facts={"pricing": "Team plan costs CHF 19 per user per month."},
        threshold=0.2,
        use_nli=False,
        raise_on_fail=False,
    )

    await filter_hook(context, next_function)

    assert calls == ["next"]
    assert context.result == "Team plan costs CHF 19 per user per month."


@pytest.mark.asyncio
async def test_filter_rewrites_rejected_result_without_raising() -> None:
    """The public filter hook should return the non-raising rejection payload."""
    context = _KernelInvocationContext(
        arguments={"input": "What does the team plan cost?"}
    )

    async def next_function(invocation: _KernelInvocationContext) -> None:
        invocation.result = "The team plan is free and includes legal advice."

    filter_hook = DirectorAIFilter(
        facts={"pricing": "Team plan costs CHF 19 per user per month."},
        threshold=0.95,
        use_nli=False,
        raise_on_fail=False,
    )

    await filter_hook(context, next_function)

    assert isinstance(context.result, dict)
    assert context.result["approved"] is False
    assert context.result["original"] == (
        "The team plan is free and includes legal advice."
    )
    assert 0.0 <= context.result["score"] < 0.95
