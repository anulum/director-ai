# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — runtime object builders for configuration

"""Construct runtime objects (store, scorer, contradiction halt) from config.

Split out of ``config.py``: assembling the retrieval store, the coherence
scorer, and the optional contradiction-driven streaming halt from configuration
fields is a distinct responsibility from holding and validating those fields.
``DirectorConfig`` keeps thin ``build_store`` / ``build_scorer`` /
``build_contradiction_halt`` methods that delegate here, so the public
``cfg.build_*()`` surface is unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .calibration.recall_correctness_client import RemanentiaCorrectnessClient
    from .config import DirectorConfig
    from .retrieval.knowledge import GroundTruthStore
    from .runtime.contradiction_halt import ContradictionHalt
    from .scoring.scorer import CoherenceScorer

logger = logging.getLogger("DirectorAI.Config")


def build_store(cfg: DirectorConfig) -> GroundTruthStore:
    """Construct a VectorGroundTruthStore from config fields.

    In ``general`` mode, returns a bare GroundTruthStore (no vector backend).
    In ``grounded`` and ``auto`` modes, builds the full vector pipeline.
    """
    if cfg.mode == "general":
        from .retrieval.knowledge import GroundTruthStore

        logger.info("Mode 'general': no vector store (NLI-only scoring)")
        return GroundTruthStore()
    if cfg.redis_url:
        try:
            from director_ai.enterprise.redis import RedisGroundTruthStore

            return RedisGroundTruthStore(
                redis_url=cfg.redis_url,
                prefix=cfg.redis_prefix + "facts:",
            )
        except ImportError:
            logger.warning(
                "director-ai[enterprise] not installed, "
                "falling back to local vector store",
            )

    from .retrieval.vector_store import (
        InMemoryBackend,
        VectorBackend,
        VectorGroundTruthStore,
    )

    backend: VectorBackend
    if cfg.vector_backend == "chroma":
        try:
            from .retrieval.vector_store import ChromaBackend

            backend = ChromaBackend(
                collection_name=cfg.chroma_collection,
                persist_directory=cfg.chroma_persist_dir or None,
            )
        except ImportError:
            logger.warning("chromadb not installed, falling back to memory backend")
            backend = InMemoryBackend()
    elif cfg.vector_backend == "sentence-transformer":
        try:
            from .retrieval.vector_store import SentenceTransformerBackend

            backend = SentenceTransformerBackend(
                model_name=cfg.embedding_model,
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, falling back to memory",
            )
            backend = InMemoryBackend()
    elif cfg.vector_backend == "http-faiss":
        try:
            from .retrieval.vector_store import FAISSBackend, HttpEmbeddingFunction

            embed_fn = HttpEmbeddingFunction(
                base_url=cfg.embedding_base_url,
                model=cfg.embedding_model,
                api_key=cfg.embedding_api_key,
                timeout_s=cfg.embedding_timeout_s,
                vector_size=cfg.embedding_vector_size,
            )
            backend = FAISSBackend(
                embed_fn=embed_fn,
                vector_size=cfg.embedding_vector_size,
            )
        except ImportError:
            logger.warning("faiss not installed, falling back to memory")
            backend = InMemoryBackend()
    elif cfg.vector_backend == "remanentia":
        from .retrieval.vector_store import RemanentiaVectorBackend

        backend = RemanentiaVectorBackend(
            base_url=cfg.remanentia_base_url,
            timeout_s=cfg.remanentia_timeout_s,
            source=cfg.remanentia_source,
        )
    else:
        # Try vector backend registry for third-party / unrecognized names
        try:
            from .retrieval.vector_store import get_vector_backend

            backend_cls = get_vector_backend(cfg.vector_backend)
            backend = backend_cls()
        except (KeyError, TypeError):
            logger.warning(
                "Unknown vector_backend %r, falling back to memory",
                cfg.vector_backend,
            )
            backend = InMemoryBackend()

    local_decorators_enabled = cfg.vector_backend != "remanentia"

    if cfg.hybrid_retrieval and local_decorators_enabled:
        try:
            from .retrieval.vector_store import HybridBackend

            backend = HybridBackend(
                base=backend,
                rrf_k=cfg.hybrid_rrf_k,
                sparse_weight=cfg.hybrid_sparse_weight,
                dense_weight=cfg.hybrid_dense_weight,
                fusion_method=cfg.hybrid_fusion_method,
            )
            logger.info(
                "Hybrid retrieval enabled (BM25 + dense, fusion=%s, k=%s)",
                cfg.hybrid_fusion_method,
                cfg.hybrid_rrf_k,
            )
        except ImportError:
            logger.warning("HybridBackend unavailable, using dense-only retrieval")

    if cfg.reranker_enabled and local_decorators_enabled:
        try:
            from .retrieval.vector_store import RerankedBackend

            backend = RerankedBackend(
                base=backend,
                reranker_model=cfg.reranker_model,
                reranker_revision=cfg.reranker_model_revision or None,
                top_k_multiplier=cfg.reranker_top_k_multiplier,
            )
            logger.info("Cross-encoder reranker enabled: %s", cfg.reranker_model)
        except (ImportError, OSError) as exc:
            if cfg.production_mode or cfg.hardened:
                raise RuntimeError(
                    "reranker_enabled=True but the reranker model could not load",
                ) from exc
            logger.warning("Reranker unavailable, skipping reranker: %s", exc)

    if cfg.parent_child_enabled:
        from .retrieval.parent_child import ParentChildBackend

        backend = ParentChildBackend(
            base=backend,
            parent_size=cfg.parent_chunk_size,
            child_size=cfg.child_chunk_size,
        )
        logger.info(
            "Parent-child chunking enabled (parent=%d, child=%d)",
            cfg.parent_chunk_size,
            cfg.child_chunk_size,
        )

    if cfg.hyde_enabled:
        from .retrieval.hyde import HyDEBackend

        if cfg.hyde_prompt_template:
            backend = HyDEBackend(base=backend, template=cfg.hyde_prompt_template)
        else:
            # Generator will be injected by build_scorer() or by the user.
            backend = HyDEBackend(base=backend)
        logger.info("HyDE retrieval enabled")

    if cfg.query_decomposition_enabled:
        from .retrieval.query_decomposition import QueryDecompositionBackend

        backend = QueryDecompositionBackend(
            base=backend,
            strategy=cfg.query_decomposition_strategy,
        )
        logger.info(
            "Query decomposition enabled (strategy=%s)",
            cfg.query_decomposition_strategy,
        )

    if cfg.contextual_compression_enabled:
        from .retrieval.contextual_compression import (
            ContextualCompressionBackend,
        )

        backend = ContextualCompressionBackend(
            base=backend,
            strategy=cfg.contextual_compression_strategy,
        )
        logger.info("Contextual compression enabled")

    if cfg.multi_vector_enabled:
        from .retrieval.multi_vector import MultiVectorBackend

        reps = [
            r.strip() for r in cfg.multi_vector_representations.split(",") if r.strip()
        ]
        backend = MultiVectorBackend(base=backend, representations=reps)
        logger.info(
            "Multi-vector retrieval enabled (representations=%s)",
            reps,
        )

    from .evidence_firewall import build_evidence_firewall

    return VectorGroundTruthStore(
        backend=backend,
        evidence_firewall=build_evidence_firewall(cfg),
    )


def resolve_scorer_backend(cfg: DirectorConfig) -> str:
    """Resolve 'auto' scorer backend to best available."""
    if cfg.scorer_backend != "auto":
        return cfg.scorer_backend

    # Priority: rust > onnx > deberta > nli-lite > lite
    import importlib.util

    if importlib.util.find_spec("backfire_kernel") is not None:
        logger.info("Auto scorer: selected 'rust' (backfire_kernel available)")
        return "rust"

    if cfg.onnx_path:
        logger.info("Auto scorer: selected 'onnx' (onnx_path configured)")
        return "onnx"

    if cfg.use_nli:
        logger.info("Auto scorer: selected 'deberta' (NLI enabled)")
        return "deberta"

    logger.info("Auto scorer: selected 'lite' (no NLI available)")
    return "lite"


def build_scorer(
    cfg: DirectorConfig, store: GroundTruthStore | None = None
) -> CoherenceScorer:
    """Construct a CoherenceScorer wired to all relevant config fields."""
    from .metrics import metrics
    from .scoring.scorer import CoherenceScorer

    if store is None:
        # Call through the method so a patched build_store is honoured.
        store = cfg.build_store()

    judge_model = cfg.llm_judge_model
    if cfg.llm_judge_provider == "local" and cfg.llm_judge_local_model:
        judge_model = cfg.llm_judge_local_model

    resolved_backend = cfg._resolve_scorer_backend()

    nli_model = cfg.nli_model
    nli_revision = cfg.nli_model_revision or None
    if cfg.model_fallback_enabled:
        from .model_registry import FallbackModelRegistry

        resolved = FallbackModelRegistry().resolve(
            "nli", nli_model, primary_revision=nli_revision
        )
        nli_model, nli_revision = resolved.model_id, resolved.revision

    kw: dict[str, Any] = {
        "threshold": cfg.coherence_threshold,
        "use_nli": cfg.use_nli,
        "require_model_backed_nli": cfg.coherence_require_model_backed_nli,
        "strict_mode": cfg.strict_mode,
        "scorer_backend": resolved_backend,
        "soft_limit": cfg.soft_limit,
        "nli_model": nli_model,
        "nli_revision": nli_revision,
        "nli_max_length": cfg.nli_max_length,
        "llm_judge_enabled": cfg.llm_judge_enabled,
        "llm_judge_confidence_threshold": cfg.llm_judge_confidence_threshold,
        "llm_judge_provider": cfg.llm_judge_provider,
        "llm_judge_model": judge_model,
        "llm_judge_model_revision": cfg.llm_judge_model_revision or None,
        "llm_judge_rubric": cfg.llm_judge_rubric,
        "llm_judge_ensemble": cfg.llm_judge_ensemble,
        "reasoning_enabled": cfg.reasoning_enabled,
        "reasoning_provider": cfg.reasoning_provider,
        "reasoning_model": cfg.reasoning_model,
        "reasoning_model_revision": cfg.reasoning_model_revision or None,
        "reasoning_escalation_margin": cfg.reasoning_escalation_margin,
        "minicheck_variant": cfg.minicheck_variant,
        "nli_quantize_8bit": cfg.nli_quantize_8bit,
        "nli_torch_dtype": cfg.nli_torch_dtype or None,
        "nli_device": cfg.nli_device or None,
        "privacy_mode": cfg.privacy_mode,
        "ground_truth_store": store,
        "onnx_batch_size": cfg.onnx_batch_size,
        "onnx_flush_timeout_ms": cfg.onnx_flush_timeout_ms,
    }
    if cfg.redis_url:
        try:
            from director_ai.enterprise.redis import RedisScoreCache

            kw["cache"] = RedisScoreCache(
                redis_url=cfg.redis_url,
                prefix=cfg.redis_prefix + "cache:",
                ttl_seconds=cfg.cache_ttl,
            )
        except ImportError:
            pass
    else:
        kw["cache_size"] = cfg.cache_size
        kw["cache_ttl"] = cfg.cache_ttl

    if cfg.onnx_path:
        kw["onnx_path"] = cfg.onnx_path
    if cfg.w_logic != 0.0 or cfg.w_fact != 0.0:
        kw["w_logic"] = cfg.w_logic
        kw["w_fact"] = cfg.w_fact
    if cfg.nli_devices:
        kw["nli_devices"] = [d.strip() for d in cfg.nli_devices.split(",") if d.strip()]
    scorer = CoherenceScorer(**kw)
    scorer._fact_inner_agg = cfg.nli_fact_inner_agg
    scorer._fact_outer_agg = cfg.nli_fact_outer_agg
    scorer._logic_inner_agg = cfg.nli_logic_inner_agg
    scorer._logic_outer_agg = cfg.nli_logic_outer_agg
    scorer._premise_ratio = cfg.nli_premise_ratio
    scorer._fact_retrieval_top_k = cfg.nli_fact_retrieval_top_k
    scorer._use_prompt_as_premise = cfg.nli_use_prompt_as_premise
    scorer._summarization_nli_baseline = cfg.nli_summarization_baseline
    scorer._claim_coverage_enabled = cfg.nli_claim_coverage_enabled
    scorer._claim_support_threshold = cfg.nli_claim_support_threshold
    scorer._claim_coverage_alpha = cfg.nli_claim_coverage_alpha
    scorer._verified_scorer_enabled = cfg.verified_scorer_enabled
    scorer._verified_scorer_atomic = cfg.verified_scorer_atomic
    scorer._verified_scorer_evidence_top_k = cfg.verified_scorer_evidence_top_k
    scorer._verified_scorer_low_confidence_margin = (
        cfg.verified_scorer_low_confidence_margin
    )
    scorer._verified_scorer_min_coverage = cfg.verified_scorer_min_coverage
    scorer._adaptive_threshold_enabled = cfg.adaptive_threshold_enabled
    scorer._adaptive_threshold_fail_closed = cfg.adaptive_threshold_fail_closed
    scorer._task_type_thresholds = {
        "summarization": cfg.threshold_summarization,
        "qa": cfg.threshold_qa,
        "fact_check": cfg.threshold_fact_check,
        "rag": cfg.threshold_rag,
        "dialogue": cfg.threshold_dialogue,
    }
    scorer._chunk_overlap_ratio = cfg.nli_chunk_overlap_ratio
    scorer._qa_premise_ratio = cfg.nli_qa_premise_ratio
    scorer._confidence_weighted_agg = cfg.nli_confidence_weighted_agg
    scorer._retrieval_abstention_threshold = cfg.retrieval_abstention_threshold
    if cfg.injection_detection_enabled:
        try:
            scorer.enable_injection_detection(
                injection_threshold=cfg.injection_threshold,
                drift_threshold=cfg.injection_drift_threshold,
                injection_claim_threshold=cfg.injection_claim_threshold,
                baseline_divergence=cfg.injection_baseline_divergence,
                stage1_weight=cfg.injection_stage1_weight,
                require_model_backed_nli=cfg.injection_require_model_backed_nli,
                fail_closed_on_error=cfg.injection_fail_closed_on_error,
            )
        except RuntimeError as exc:
            metrics.inc_labeled(
                "scorer_startup_failures_total",
                labels={
                    "component": "injection",
                    "reason": "detector_init_runtime_error",
                },
            )
            metrics.inc_labeled(
                "injection_startup_failures_total",
                labels={"reason": type(exc).__name__},
            )
            if cfg.injection_require_model_backed_nli:
                metrics.inc_labeled(
                    "model_backed_nli_startup_failures_total",
                    labels={"stage": "injection"},
                )
            raise
    if cfg.lora_adapter_path and scorer._nli is not None:
        if hasattr(scorer._nli, "_load_lora_adapter"):
            scorer._nli._load_lora_adapter(cfg.lora_adapter_path)
        else:
            logger.warning(
                "LoRA adapter not supported on %s",
                type(scorer._nli).__name__,
            )
    if cfg.meta_classifier_path:
        scorer._meta_classifier_path = cfg.meta_classifier_path
    if cfg.adaptive_retrieval_enabled:
        scorer.enable_adaptive_retrieval(
            threshold=cfg.adaptive_retrieval_threshold,
        )
    if scorer._adaptive_threshold_enabled and scorer._adaptive_threshold_fail_closed:
        try:
            meta_classifier = scorer._get_meta_classifier()
        except RuntimeError:
            metrics.inc_labeled(
                "scorer_startup_failures_total",
                labels={
                    "component": "adaptive_threshold",
                    "reason": "classifier_init_runtime_error",
                },
            )
            metrics.inc_labeled(
                "adaptive_threshold_startup_failures_total",
                labels={"reason": "exception_during_classifier_init"},
            )
            raise
        if meta_classifier is None:
            metrics.inc_labeled(
                "scorer_startup_failures_total",
                labels={
                    "component": "adaptive_threshold",
                    "reason": "classifier_missing_or_unloadable",
                },
            )
            metrics.inc_labeled(
                "adaptive_threshold_startup_failures_total",
                labels={"reason": "missing_or_unloadable_classifier"},
            )
            raise RuntimeError(
                "adaptive_threshold_fail_closed=True requires a loadable meta-classifier "
                "(set meta_classifier_path or provide a compatible bundled artefact)",
            )
    if cfg.dry_run:
        scorer._dry_run = True
        logger.info("Dry-run mode: scoring but never rejecting")
    if cfg.cost_tracking_enabled:
        from director_ai.compliance.cost_analyser import CostAnalyser

        analyser = CostAnalyser()
        scorer._cost_analyser = analyser

        def _cost_cb(model: str, inp: int, out: int) -> None:
            analyser.record(model, input_tokens=inp, output_tokens=out)

        scorer._judge._cost_callback = _cost_cb
        logger.info("Cost tracking enabled on scorer")
    if cfg.coherence_require_model_backed_nli and not scorer._has_model_backed_nli():
        metrics.inc_labeled(
            "scorer_startup_failures_total",
            labels={
                "component": "coherence",
                "reason": "model_backed_nli_unavailable",
            },
        )
        metrics.inc_labeled(
            "model_backed_nli_startup_failures_total",
            labels={"stage": "coherence"},
        )
        raise RuntimeError(
            "Configured coherence_require_model_backed_nli=True, but scorer could not initialize a model-backed NLI backend",
        )
    if (
        cfg.injection_detection_enabled
        and cfg.injection_require_model_backed_nli
        and not scorer._has_model_backed_nli()
    ):
        metrics.inc_labeled(
            "scorer_startup_failures_total",
            labels={
                "component": "injection",
                "reason": "model_backed_nli_unavailable",
            },
        )
        metrics.inc_labeled(
            "model_backed_nli_startup_failures_total",
            labels={"stage": "injection"},
        )
        raise RuntimeError(
            "Configured injection_require_model_backed_nli=True, but scorer could not initialize a model-backed NLI backend for injection detection",
        )
    return scorer


def build_contradiction_halt(
    cfg: DirectorConfig,
    store: GroundTruthStore | None = None,
) -> ContradictionHalt | None:
    """Build the opt-in contradiction-driven streaming halt, or ``None``.

    Returns ``None`` when ``streaming_contradiction_halt`` is off, or when
    the NLI extra is missing / the model cannot load — the caller then keeps
    the coherence halt instead of failing startup, matching the prompt-guard
    degrade path. The halt scores each completed streamed claim against the
    store's retrieved grounding and stops on a contradiction.
    """
    if not cfg.streaming_contradiction_halt:
        return None
    from .runtime.contradiction_halt import ContradictionHalt
    from .scoring.contradiction import ContradictionScorer

    if store is None:
        # Call through the method so a patched build_store is honoured.
        store = cfg.build_store()
    try:
        scorer = ContradictionScorer.from_pretrained(
            cfg.streaming_contradiction_model,
            revision=cfg.streaming_contradiction_revision or None,
            device=cfg.streaming_contradiction_device,
            threshold=cfg.streaming_contradiction_threshold,
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never crash startup
        logger.warning(
            "streaming_contradiction_halt enabled but the contradiction "
            "model could not load (%s); keeping the coherence halt.",
            exc,
        )
        return None
    return ContradictionHalt(scorer, store.retrieve_context)


def build_correctness_feedback(
    cfg: DirectorConfig,
) -> RemanentiaCorrectnessClient | None:
    """Build the opt-in REMANENTIA recall-correctness client, or ``None``.

    Returns ``None`` unless ``remanentia_correctness_feedback`` is on and the
    grounding actually comes from the REMANENTIA vector backend — there is no
    recall ledger to label otherwise. When engaged, the agent posts each
    verification verdict back as the ``was_correct`` label for the recall that
    grounded the answer, closing REMANENTIA's two-label loop. The client reuses
    the configured REMANENTIA base URL and timeout and carries the bearer token
    the authenticated ``/recall`` family requires.
    """
    if not cfg.remanentia_correctness_feedback:
        return None
    if cfg.vector_backend != "remanentia":
        logger.warning(
            "remanentia_correctness_feedback is on but vector_backend is %r, "
            "not 'remanentia'; no recall ledger to label, skipping.",
            cfg.vector_backend,
        )
        return None
    from .calibration.recall_correctness_client import RemanentiaCorrectnessClient

    return RemanentiaCorrectnessClient(
        base_url=cfg.remanentia_base_url,
        token=cfg.remanentia_token,
        timeout_s=cfg.remanentia_timeout_s,
    )
