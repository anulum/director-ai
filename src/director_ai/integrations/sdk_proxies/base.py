# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Shared state and gating helpers for the SDK guard proxies."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar, copy_context
from typing import Any

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.exceptions import HallucinationError, InjectionDetectedError
from director_ai.core.types import CoherenceScore

_log = logging.getLogger("DirectorAI.guard")
_score_var: ContextVar[CoherenceScore | None] = ContextVar(
    "director_ai_score",
    default=None,
)

STREAM_CHECK_INTERVAL = 8


def get_score() -> CoherenceScore | None:
    """Retrieve the last score stored by ``on_fail="metadata"``."""
    return _score_var.get()


def score(
    prompt: str,
    response: str,
    *,
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    threshold: float = 0.3,
    use_nli: bool | None = None,
    profile: str | None = None,
    injection_detection: bool = False,
    injection_threshold: float = 0.7,
    require_model_backed_nli: bool = False,
    injection_require_model_backed_nli: bool = False,
    injection_fail_closed_on_error: bool = False,
) -> CoherenceScore:
    """Score a single prompt/response pair for hallucination.

    Returns a ``CoherenceScore`` without requiring an SDK client.
    When *injection_detection* is enabled, ``CoherenceScore.injection_risk``
    is populated with the intent-grounded injection risk score.
    """
    if profile is not None:
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig.from_profile(profile)
        if use_nli is not None:
            cfg.use_nli = use_nli
        if require_model_backed_nli:
            cfg.coherence_require_model_backed_nli = True
        if cfg.coherence_require_model_backed_nli and not cfg.use_nli:
            raise ValueError(
                "require_model_backed_nli=True requires use_nli=True "
                "when scoring with a profile",
            )
        gts = store or GroundTruthStore()
        if facts:
            for k, v in facts.items():
                gts.add(k, v)
        scorer = cfg.build_scorer(store=gts)
    else:
        gts = store or GroundTruthStore()
        if facts:
            for k, v in facts.items():
                gts.add(k, v)
        scorer = CoherenceScorer(
            threshold=threshold,
            ground_truth_store=gts,
            use_nli=use_nli,
            require_model_backed_nli=require_model_backed_nli,
        )
    if injection_detection:
        scorer.enable_injection_detection(
            injection_threshold=injection_threshold,
            require_model_backed_nli=injection_require_model_backed_nli,
            fail_closed_on_error=injection_fail_closed_on_error,
        )
    _approved, cs = scorer.review(prompt, response)
    return cs


def _extract_prompt(messages: list[dict[str, Any]]) -> str:
    """Pull the user prompt from a messages array."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return str(block.get("text", ""))
            return str(content)
    return " ".join(str(m.get("content", "")) for m in messages)


def _handle_failure(on_fail: str, query: str, response_text: str, score: Any) -> None:
    """Apply the configured hallucination failure policy."""
    if on_fail == "raise":
        raise HallucinationError(query, response_text, score)
    if on_fail == "log":
        _log.warning(
            "Hallucination detected (coherence=%.3f): %.100s",
            score.score,
            response_text,
        )
    elif on_fail == "metadata":  # pragma: no branch
        _score_var.set(score)


def _handle_injection_failure(
    on_fail: str, query: str, response_text: str, score: Any
) -> None:
    """Handle a detected injection — mirrors _handle_failure semantics."""
    if on_fail == "raise":
        raise InjectionDetectedError(query, response_text, score)
    if on_fail == "log":
        risk = getattr(score, "injection_risk", None) or 0.0
        _log.warning(
            "Injection detected (risk=%.3f): %.100s",
            risk,
            response_text,
        )
    elif on_fail == "metadata":  # pragma: no branch
        _score_var.set(score)


def _check_injection(
    on_fail: str,
    query: str,
    response_text: str,
    cs: Any,
    injection_threshold: float | None,
) -> None:
    """Check injection risk on a scored response and handle failure."""
    if injection_threshold is None:
        return
    risk = cs.injection_risk
    if risk is not None and risk >= injection_threshold:
        _handle_injection_failure(on_fail, query, response_text, cs)


def _score_and_gate(
    scorer: Any,
    on_fail: str,
    query: str,
    response_text: str,
    *,
    injection_threshold: float | None = None,
) -> Any:
    """Score a response synchronously and enforce hallucination/injection gates."""
    result = scorer.review(query, response_text)
    if asyncio.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            ctx = copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                approved, cs = pool.submit(ctx.run, asyncio.run, result).result()
        else:
            approved, cs = asyncio.run(result)
    else:
        approved, cs = result
    if on_fail == "metadata":
        _score_var.set(cs)
    if not approved:
        _handle_failure(on_fail, query, response_text, cs)
    _check_injection(on_fail, query, response_text, cs, injection_threshold)
    return cs


async def _ascore_and_gate(
    scorer: Any,
    on_fail: str,
    query: str,
    response_text: str,
    *,
    injection_threshold: float | None = None,
) -> Any:
    """Asynchronously score a response and enforce hallucination/injection gates."""
    result = scorer.review(query, response_text)
    if asyncio.iscoroutine(result):
        approved, cs = await result
    else:
        approved, cs = result
    if on_fail == "metadata":
        _score_var.set(cs)
    if not approved:
        _handle_failure(on_fail, query, response_text, cs)
    _check_injection(on_fail, query, response_text, cs, injection_threshold)
    return cs
