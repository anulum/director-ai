# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — shared transport-level streaming oversight

"""Apply one pre-egress decision policy across the server transports."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .core.agent import AgentStreamEvent, CoherenceAgent
from .core.runtime.streaming import StreamingKernel

if TYPE_CHECKING:
    from .core.config import DirectorConfig


async def _agent_events(
    agent: Any, prompt: str, tenant_id: str
) -> AsyncGenerator[AgentStreamEvent, None]:
    """Adapt tuple-only agents while retaining native agent decisions."""
    if isinstance(agent, CoherenceAgent):
        async with aclosing(agent.stream_events(prompt, tenant_id)) as events:
            async for event in events:
                yield event
        return
    async for token, score in agent.stream(prompt, tenant_id=tenant_id):
        yield AgentStreamEvent(token, score)


async def stream_for_transport(
    agent: Any,
    cfg: DirectorConfig,
    prompt: str,
    tenant_id: str,
) -> AsyncGenerator[AgentStreamEvent, None]:
    """Yield approved tokens and the first halt decision with its trigger.

    Opt-in contradiction decisions use the configured contradiction
    threshold. The generic coherence policy also checks the server's
    request-local egress thresholds.
    """
    kernel = None
    if (
        not isinstance(agent, CoherenceAgent)
        or agent.contradiction_halt is None
        or not hasattr(agent.generator, "stream_tokens")
    ):
        kernel = StreamingKernel(
            hard_limit=cfg.hard_limit,
            window_size=getattr(cfg, "window_size", 5),
            window_threshold=getattr(cfg, "window_threshold", 0.5),
        )

    async with aclosing(_agent_events(agent, prompt, tenant_id)) as events:
        async for event in events:
            if event.halt_reason:
                yield event
                return
            if kernel is not None and kernel.check_halt(event.coherence):
                yield replace(event, halt_reason="coherence_halt")
                return
            yield event
