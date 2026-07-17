# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Server Lifecycle (startup state, hot-swap, shutdown)

"""Server lifecycle: startup state assembly, scorer hot-swap, shutdown.

Split out of :mod:`director_ai.server` (WCB-8). ``server_lifespan`` is the
FastAPI lifespan context manager — it builds every ``app.state._state``
entry (sanitizer, redactor, scorer/agent/batch bundle, stats, sessions,
review queue, audit, compliance, feedback, tenants, multimodal) and tears
them down on shutdown. The scorer hot-swap trio
(:func:`_build_coherence_agent` / :func:`_swap_scorer` /
:func:`_activate_scorer`) lives here because the startup bootstrap and the
live fine-tune activation must wire the bundle identically.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from .core.config import DirectorConfig
from .core.metrics import metrics

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger("DirectorAI.Server")

__all__ = [
    "server_lifespan",
]


def _build_coherence_agent(cfg: DirectorConfig, scorer: Any, store: Any) -> Any:
    """Assemble a ``CoherenceAgent`` wired to a scorer, store, and config.

    Shared by the startup bootstrap and the live scorer hot-swap so both paths
    wire the agent identically (LLM provider, contradiction-driven halt, and
    REMANENTIA correctness feedback). Keeping one builder prevents the
    hot-swapped agent from silently drifting from the one built at startup.
    """
    from .core.agent import CoherenceAgent

    agent_kwargs: dict[str, Any] = {
        "_scorer": scorer,
        "_store": store,
        "production_mode": cfg.production_mode,
        "llm_max_tokens": cfg.llm_max_tokens,
        "llm_temperature": cfg.llm_temperature,
        "max_candidates": cfg.max_candidates,
    }
    if cfg.llm_provider == "local":
        agent_kwargs["llm_api_url"] = cfg.llm_api_url
    elif cfg.llm_provider in ("openai", "anthropic"):
        agent_kwargs["provider"] = cfg.llm_provider
        if cfg.llm_api_key:
            agent_kwargs["api_key"] = cfg.llm_api_key
        logger.info("LLM provider: %s", cfg.llm_provider)
    contradiction_halt = cfg.build_contradiction_halt(store)
    if contradiction_halt is not None:
        agent_kwargs["contradiction_halt"] = contradiction_halt
        logger.info(
            "Contradiction-driven streaming halt enabled (threshold=%.2f)",
            cfg.streaming_contradiction_threshold,
        )
    correctness_feedback = cfg.build_correctness_feedback()
    if correctness_feedback is not None:
        agent_kwargs["correctness_feedback"] = correctness_feedback
        logger.info(
            "REMANENTIA recall-correctness feedback enabled (%s)",
            cfg.remanentia_base_url,
        )
    return CoherenceAgent(**agent_kwargs)


def _swap_scorer(state: dict[str, Any], model_path: str) -> None:
    """Rebuild the scoring bundle around ``model_path`` and swap it into ``state``.

    Runs synchronously — the NLI load is the heavy step, so the async caller
    offloads it to a worker thread. Route handlers read ``scorer``/``agent``/
    ``batch`` fresh from ``state`` on each request, so replacing the entries
    hot-swaps the live model with no restart. The ground-truth ``store`` is
    reused, so runtime-added facts survive the swap, and ``config`` is updated
    so health/introspection endpoints report the now-active model.
    """
    from .core.runtime.batch import BatchProcessor

    cfg: DirectorConfig = state["config"]
    store = state["store"]
    new_cfg = dataclasses.replace(cfg, nli_model=model_path)
    new_scorer = new_cfg.build_scorer(store=store)
    new_agent = _build_coherence_agent(new_cfg, new_scorer, store)
    new_batch = BatchProcessor(new_agent, max_concurrency=new_cfg.batch_max_concurrency)
    state["config"] = new_cfg
    state["scorer"] = new_scorer
    state["agent"] = new_agent
    state["batch"] = new_batch


async def _activate_scorer(state: dict[str, Any], model_path: str) -> None:
    """Hot-swap the live scorer to the fine-tuned model at ``model_path``.

    Serialised by ``scorer_swap_lock`` so two activations cannot interleave, with
    the heavy rebuild offloaded off the event loop. A running review queue is
    rebuilt around the new scorer and the superseded one drained.
    """
    lock: asyncio.Lock = state["scorer_swap_lock"]
    async with lock:
        old_queue = state.get("review_queue")
        await asyncio.to_thread(_swap_scorer, state, model_path)
        if old_queue is not None:
            from .core.runtime.review_queue import ReviewQueue

            new_cfg: DirectorConfig = state["config"]
            new_queue = ReviewQueue(
                state["scorer"],
                max_batch=new_cfg.review_queue_max_batch,
                flush_timeout_ms=new_cfg.review_queue_flush_timeout_ms,
            )
            await new_queue.start()
            state["review_queue"] = new_queue
            await old_queue.stop()


@asynccontextmanager
async def server_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifecycle events for the FastAPI server."""
    cfg = app.state.config

    from .core.license import load_license

    lic = load_license()
    app.state._license = lic
    if lic.is_commercial:
        logger.info(
            "Director-AI v%s — Licensed to %s (%s tier)",
            __import__("director_ai").__version__,
            lic.licensee or lic.key[:20],
            lic.tier,
        )
    elif lic.is_trial:
        logger.info("Director-AI — Trial license (expires %s)", lic.expires)
    else:
        logger.info("Director-AI — open core: Apache-2.0 + BUSL-1.1")

    logger.info("Starting Director-AI server")

    app.state._state = {}  # Initialize _state on app.state

    from .core.runtime.batch import BatchProcessor
    from .core.safety.audit import AuditLogger
    from .core.safety.sanitizer import InputSanitizer
    from .core.tenant import TenantRouter

    if cfg.sanitize_inputs:
        app.state._state["sanitizer"] = InputSanitizer(
            block_threshold=cfg.sanitizer_block_threshold,
        )

    if cfg.prompt_guard_model_enabled:
        from .core.safety.prompt_guard import (
            LayeredPromptGuard,
            PromptInjectionModel,
        )

        base = app.state._state.get("sanitizer") or InputSanitizer(
            block_threshold=cfg.sanitizer_block_threshold,
        )
        try:
            model = PromptInjectionModel.from_pretrained(
                cfg.prompt_guard_model_id,
                revision=cfg.prompt_guard_model_revision or None,
                threshold=cfg.prompt_guard_threshold,
            )
            app.state._state["sanitizer"] = LayeredPromptGuard(base, model)
            logger.info(
                "Model-backed prompt-injection screen enabled: %s",
                cfg.prompt_guard_model_id,
            )
        except Exception as exc:  # noqa: BLE001 — degrade, never crash startup
            app.state._state["sanitizer"] = base
            logger.warning(
                "prompt_guard_model_enabled but the classifier could not "
                "load (%s); falling back to the pattern sanitizer.",
                exc,
            )

    from .core.redactor import PIIRedactor

    app.state._state["redactor"] = PIIRedactor(enabled=cfg.redact_pii)
    if cfg.redact_pii:
        logger.info("Enterprise PII Redaction enabled")

    store = cfg.build_store()
    scorer = cfg.build_scorer(store=store)
    agent = _build_coherence_agent(cfg, scorer, store)
    batch_proc = BatchProcessor(agent, max_concurrency=cfg.batch_max_concurrency)

    stats = None
    if cfg.stats_backend == "sqlite":
        from .core.stats import StatsStore

        stats = StatsStore(db_path=cfg.stats_db_path)
        logger.info("SQLite stats backend: %s", cfg.stats_db_path)

    app.state._state["agent"] = agent
    app.state._state["scorer"] = scorer
    app.state._state["batch"] = batch_proc
    app.state._state["config"] = cfg
    app.state._state["stats"] = stats
    app.state._state["sessions"] = {}
    app.state._state["session_owners"] = {}
    app.state._state["sessions_lock"] = asyncio.Lock()
    app.state._state["max_sessions"] = getattr(cfg, "max_sessions", 10000)

    review_queue = None
    if cfg.review_queue_enabled:
        from .core.runtime.review_queue import ReviewQueue

        review_queue = ReviewQueue(
            scorer,
            max_batch=cfg.review_queue_max_batch,
            flush_timeout_ms=cfg.review_queue_flush_timeout_ms,
        )
        await review_queue.start()
    app.state._state["review_queue"] = review_queue
    # Live scorer hot-swap wiring: the fine-tune activation route calls this
    # activator (via app.state) to rebuild the scoring bundle around a newly
    # activated model with no restart. The store is reused so runtime-added
    # facts survive; the lock serialises concurrent activations.
    app.state._state["store"] = store
    app.state._state["scorer_swap_lock"] = asyncio.Lock()
    app.state._state["scorer_activator"] = functools.partial(
        _activate_scorer, app.state._state
    )

    if cfg.audit_log_path or cfg.audit_postgres_url:
        audit_logger = AuditLogger(path=cfg.audit_log_path)
        if cfg.audit_postgres_url:
            from .enterprise.audit_pg import PostgresAuditSink

            audit_logger.add_sink(PostgresAuditSink(db_url=cfg.audit_postgres_url))

        app.state._state["audit"] = audit_logger
        logger.info(
            "Audit logging initialized (path: %s, db: %s)",
            bool(cfg.audit_log_path),
            bool(cfg.audit_postgres_url),
        )

    if cfg.compliance_db_path:
        from .compliance.audit_log import AuditLog as ComplianceAuditLog
        from .compliance.drift_detector import DriftDetector
        from .compliance.reporter import ComplianceReporter

        # SEC-2: the compliance trail is durable and sealed, so raw PII in
        # prompt/response would persist forever. Reuse the single pipeline
        # redactor (enabled by redact_pii) so the sealed content is masked
        # at the sink; a disabled redactor is a passthrough (raw retained).
        # KIMI2-C: audit_strict_mode (on in the production profile) makes
        # that posture fail-closed — construction raises without a redactor
        # or a durable HMAC secret instead of warning.
        c_log = ComplianceAuditLog(
            cfg.compliance_db_path,
            redactor=app.state._state.get("redactor"),
            strict_mode=getattr(cfg, "audit_strict_mode", False),
        )
        app.state._state["compliance_log"] = c_log
        app.state._state["compliance_reporter"] = ComplianceReporter(c_log)
        app.state._state["compliance_drift"] = DriftDetector(c_log)
        logger.info("Compliance audit log: %s", cfg.compliance_db_path)

    if cfg.feedback_db_path:
        from .core.calibration.feedback_store import FeedbackStore

        feedback_store = FeedbackStore(cfg.feedback_db_path)
        app.state._state["feedback_store"] = feedback_store
        logger.info("Feedback store: %s", cfg.feedback_db_path)

    if cfg.tenant_routing:
        app.state._state["tenant_router"] = TenantRouter()
        logger.info("Tenant routing enabled")

    from .core.retrieval.doc_registry import DocRegistry

    app.state._state["doc_registry"] = DocRegistry()

    # Multi-modal hallucination guard: opt-in and isolated. Only stood up
    # when the experimental hooks flag is set AND a modality is configured,
    # so the default safety posture is unchanged.
    if cfg.multimodal_enabled_modalities:
        from .experimental import experimental_hooks_enabled

        if experimental_hooks_enabled():
            from .core.multimodal_guard import build_hashbag_adapter

            app.state._state["multimodal_adapter"] = build_hashbag_adapter(
                enabled_modalities=cfg.multimodal_enabled_modalities,
                benchmarked_modalities=cfg.multimodal_benchmarked_modalities,
                dim=cfg.multimodal_embedding_dim,
                hallucination_threshold=cfg.multimodal_hallucination_threshold,
                consistency_threshold=cfg.multimodal_consistency_threshold,
                temporal_alpha=cfg.multimodal_temporal_alpha,
                temporal_floor=cfg.multimodal_temporal_floor,
                grounding_floor=cfg.multimodal_grounding_floor,
                grounding_allow_threshold=cfg.multimodal_grounding_allow_threshold,
            )
            logger.info(
                "Multimodal guard enabled (modalities=%s)",
                cfg.multimodal_enabled_modalities,
            )

    cfg.configure_logging()

    if cfg.otel_enabled:
        from .core.otel import setup_otel

        setup_otel()

    if cfg.use_nli:  # pragma: no cover — lifespan only runs under ASGI
        metrics.gauge_set("nli_model_loaded", 1.0)

    logger.info(
        "Director AI server started (profile=%s, nli=%s)",
        cfg.profile,
        cfg.use_nli,
    )
    yield
    logger.info("Director AI server shutting down")
    if review_queue:
        await review_queue.stop()
    if stats:
        try:
            stats.close()
        except Exception:  # pragma: no cover — defensive
            logger.warning("Failed to close stats database")
    c_log_shutdown = app.state._state.get("compliance_log")
    if c_log_shutdown is not None:
        c_log_shutdown.close()
    feedback_shutdown = app.state._state.get("feedback_store")
    if feedback_shutdown is not None:
        feedback_shutdown.close()
