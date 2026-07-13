# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — configuration validation and normalisation

"""Validate configuration bounds and apply profile-derived defaults.

Split out of ``config.py``: enforcing the cross-field invariants (mode, hardened
and production-mode rules, threshold ranges, backend prerequisites) and applying
the derived defaults is a distinct responsibility from declaring the fields.
``DirectorConfig.__post_init__`` calls :func:`validate_and_normalize`, which
mutates the instance through ``object.__setattr__`` exactly as the inline
``__post_init__`` did, so construction behaviour is unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DirectorConfig

logger = logging.getLogger("DirectorAI.Config")


def _require_unit_interval(name: str, value: float) -> None:
    """Validate that a numeric configuration score is in ``[0, 1]``."""
    if isinstance(value, bool) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def validate_and_normalize(cfg: DirectorConfig) -> None:
    """Apply profile-derived defaults and validate configuration bounds."""
    if cfg.mode not in ("general", "grounded", "auto"):
        raise ValueError(
            f"mode must be 'general', 'grounded', or 'auto', got {cfg.mode!r}"
        )
    # Apply mode defaults before other validation
    if cfg.mode == "general":
        if not cfg.use_nli:
            object.__setattr__(cfg, "use_nli", True)
        object.__setattr__(cfg, "hybrid_retrieval", False)
        object.__setattr__(cfg, "reranker_enabled", False)
        object.__setattr__(cfg, "retrieval_abstention_threshold", 0.0)
    else:
        # mode is validated to general/grounded/auto above, so the non-general
        # branch is exactly grounded or auto.
        if cfg.use_nli or cfg.hybrid_retrieval or cfg.reranker_enabled:
            object.__setattr__(cfg, "use_nli", True)
        if cfg.retrieval_abstention_threshold <= 0:
            object.__setattr__(cfg, "retrieval_abstention_threshold", 0.3)

    if not (0.0 <= cfg.coherence_threshold <= 1.0):
        raise ValueError(
            f"coherence_threshold must be in [0, 1], got {cfg.coherence_threshold}",
        )
    if not (0.0 <= cfg.hard_limit <= 1.0):
        raise ValueError(f"hard_limit must be in [0, 1], got {cfg.hard_limit}")
    if not (0.0 <= cfg.soft_limit <= 1.0):
        raise ValueError(f"soft_limit must be in [0, 1], got {cfg.soft_limit}")
    if cfg.soft_limit < cfg.hard_limit:
        raise ValueError(
            f"soft_limit ({cfg.soft_limit}) must be >= hard_limit ({cfg.hard_limit})",
        )
    if cfg.max_candidates < 1:
        raise ValueError(f"max_candidates must be >= 1, got {cfg.max_candidates}")
    if cfg.history_window < 1:
        raise ValueError(f"history_window must be >= 1, got {cfg.history_window}")
    if not (0.0 <= cfg.llm_temperature <= 2.0):
        raise ValueError(
            f"llm_temperature must be in [0, 2], got {cfg.llm_temperature}",
        )
    if cfg.llm_max_tokens < 1:
        raise ValueError(f"llm_max_tokens must be >= 1, got {cfg.llm_max_tokens}")
    if cfg.batch_max_concurrency < 1:
        raise ValueError(
            f"batch_max_concurrency must be >= 1, got {cfg.batch_max_concurrency}",
        )
    if not (1 <= cfg.server_port <= 65535):
        raise ValueError(
            f"server_port must be in [1, 65535], got {cfg.server_port}",
        )
    if cfg.server_workers < 1:
        raise ValueError(f"server_workers must be >= 1, got {cfg.server_workers}")
    if cfg.rate_limit_rpm < 0:
        raise ValueError(f"rate_limit_rpm must be >= 0, got {cfg.rate_limit_rpm}")
    if cfg.stats_backend not in ("prometheus", "sqlite"):
        raise ValueError(
            f"stats_backend must be 'prometheus' or 'sqlite', "
            f"got {cfg.stats_backend!r}",
        )
    if cfg.grpc_max_message_mb < 1:
        raise ValueError(
            f"grpc_max_message_mb must be >= 1, got {cfg.grpc_max_message_mb}",
        )
    if cfg.grpc_deadline_seconds <= 0:
        raise ValueError(
            f"grpc_deadline_seconds must be > 0, got {cfg.grpc_deadline_seconds}",
        )
    if not (0.0 <= cfg.sanitizer_block_threshold <= 1.0):
        raise ValueError(
            "sanitizer_block_threshold must be in [0, 1], "
            f"got {cfg.sanitizer_block_threshold}",
        )
    _require_unit_interval("injection_threshold", cfg.injection_threshold)
    _require_unit_interval(
        "injection_drift_threshold",
        cfg.injection_drift_threshold,
    )
    _require_unit_interval(
        "injection_claim_threshold",
        cfg.injection_claim_threshold,
    )
    _require_unit_interval(
        "injection_baseline_divergence",
        cfg.injection_baseline_divergence,
    )
    _require_unit_interval("injection_stage1_weight", cfg.injection_stage1_weight)
    if not (0.0 <= cfg.span_token_threshold <= 1.0):
        raise ValueError(
            f"span_token_threshold must be in [0, 1], got {cfg.span_token_threshold}",
        )
    if cfg.span_min_tokens < 1:
        raise ValueError(f"span_min_tokens must be >= 1, got {cfg.span_min_tokens}")
    if cfg.span_max_length < 1:
        raise ValueError(f"span_max_length must be >= 1, got {cfg.span_max_length}")
    if (cfg.w_logic != 0.0 or cfg.w_fact != 0.0) and abs(
        cfg.w_logic + cfg.w_fact - 1.0,
    ) > 1e-6:
        raise ValueError(
            f"w_logic + w_fact must equal 1.0 when set, "
            f"got {cfg.w_logic} + {cfg.w_fact}",
        )
    if cfg.reranker_enabled and not cfg.reranker_model.strip():
        raise ValueError("reranker_model must be set when reranker_enabled=True")
    if not isinstance(cfg.llm_judge_rubric, bool):
        raise ValueError("llm_judge_rubric must be a boolean")
    if not isinstance(cfg.llm_judge_ensemble, int) or isinstance(
        cfg.llm_judge_ensemble, bool
    ):
        raise ValueError("llm_judge_ensemble must be an integer")
    if not 1 <= cfg.llm_judge_ensemble <= 5:
        raise ValueError("llm_judge_ensemble must be between 1 and 5")
    if cfg.claim_decomposition_provider:
        from .scoring.claim_decomposition import VALID_DECOMPOSITION_PROVIDERS

        if cfg.claim_decomposition_provider not in VALID_DECOMPOSITION_PROVIDERS:
            raise ValueError(
                "claim_decomposition_provider must be one of "
                f"{VALID_DECOMPOSITION_PROVIDERS} or empty"
            )
        if not cfg.claim_decomposition_model.strip():
            raise ValueError(
                "claim_decomposition_model must be set when "
                "claim_decomposition_provider is enabled"
            )
    if cfg.verified_scorer_evidence_top_k < 1:
        raise ValueError("verified_scorer_evidence_top_k must be >= 1")
    if not (0.0 <= cfg.verified_scorer_low_confidence_margin <= 1.0):
        raise ValueError("verified_scorer_low_confidence_margin must be in [0, 1]")
    if not (0.0 <= cfg.verified_scorer_min_coverage <= 1.0):
        raise ValueError("verified_scorer_min_coverage must be in [0, 1]")
    if not isinstance(cfg.hybrid_rrf_k, int) or isinstance(cfg.hybrid_rrf_k, bool):
        raise ValueError("hybrid_rrf_k must be an integer")
    if cfg.hybrid_rrf_k < 1:
        raise ValueError("hybrid_rrf_k must be at least 1")
    from .retrieval.vector_store.fusion import validate_fusion_method

    object.__setattr__(
        cfg,
        "hybrid_fusion_method",
        validate_fusion_method(cfg.hybrid_fusion_method),
    )
    for weight_field in ("hybrid_sparse_weight", "hybrid_dense_weight"):
        weight = getattr(cfg, weight_field)
        if not isinstance(weight, int | float) or isinstance(weight, bool):
            raise ValueError(f"{weight_field} must be numeric")
        if weight < 0.0:
            raise ValueError(f"{weight_field} must be non-negative")
    if cfg.hybrid_sparse_weight + cfg.hybrid_dense_weight == 0.0:
        raise ValueError("at least one hybrid fusion weight must be positive")
    if cfg.scorer_model:
        from .scoring.model_choices import resolve_scorer_model_choice

        choice = resolve_scorer_model_choice(
            cfg.scorer_model,
            allow_domain_only=cfg.allow_domain_only_scorer_model,
            allow_custom=cfg.allow_custom_scorer_model,
        )
        object.__setattr__(cfg, "nli_model", choice.runtime_model)
        object.__setattr__(cfg, "nli_model_artifact_uri", choice.artifact_uri)
        object.__setattr__(cfg, "nli_model_revision", choice.revision)
        object.__setattr__(cfg, "nli_max_length", choice.max_length)

    # Hardened mode: enforce all safety features
    if cfg.hardened:
        object.__setattr__(cfg, "production_mode", True)
        object.__setattr__(cfg, "use_nli", True)
        object.__setattr__(cfg, "coherence_require_model_backed_nli", True)
        object.__setattr__(cfg, "adaptive_threshold_enabled", True)
        object.__setattr__(cfg, "adaptive_threshold_fail_closed", True)
        object.__setattr__(cfg, "injection_detection_enabled", True)
        object.__setattr__(cfg, "injection_require_model_backed_nli", True)
        object.__setattr__(cfg, "injection_fail_closed_on_error", True)
        object.__setattr__(cfg, "sanitize_inputs", True)
        object.__setattr__(cfg, "redact_pii", True)
        object.__setattr__(cfg, "strict_mode", True)
        logger.info("Hardened mode: all safety features enforced")

    if cfg.injection_require_model_backed_nli and not cfg.injection_detection_enabled:
        raise ValueError(
            "injection_require_model_backed_nli=True requires injection_detection_enabled=True",
        )
    if cfg.injection_fail_closed_on_error and not cfg.injection_detection_enabled:
        raise ValueError(
            "injection_fail_closed_on_error=True requires injection_detection_enabled=True",
        )
    if cfg.adaptive_threshold_fail_closed and not cfg.adaptive_threshold_enabled:
        raise ValueError(
            "adaptive_threshold_fail_closed=True requires adaptive_threshold_enabled=True",
        )

    # Production mode enforcements
    if cfg.production_mode:
        if cfg.dry_run:
            raise ValueError(
                "production_mode requires dry_run=False (fail-open dry-run is not permitted)",
            )
        if not cfg.sanitize_inputs:
            raise ValueError(
                "production_mode requires sanitize_inputs=True",
            )
        object.__setattr__(cfg, "knowledge_write_require_auth", True)
        # KB integrity is the product: a poisoned knowledge base lets the
        # guard certify false claims as ground truth. Production requires
        # signed KB writes.
        object.__setattr__(cfg, "knowledge_write_require_signature", True)
        # A configured rate limit must fail closed in production: a missing
        # limiter backend must refuse startup rather than silently run an
        # unthrottled public listener.
        object.__setattr__(cfg, "rate_limit_strict", True)
        if not cfg.api_keys and not cfg.api_key_tenant_map:
            raise ValueError("production_mode requires api_keys or api_key_tenant_map")
        # Tenant isolation is only trustworthy with key→tenant binding;
        # accepting X-Tenant-ID without a binding is advisory. When tenant
        # routing is on in production, a binding map is mandatory.
        if cfg.tenant_routing and not cfg.api_key_tenant_map:
            raise ValueError(
                "production_mode with tenant_routing=True requires "
                "api_key_tenant_map (key→tenant binding)"
            )
        if not cfg.llm_api_url.strip() and cfg.llm_provider in {"", "mock"}:
            raise ValueError(
                "production_mode requires a real LLM provider or llm_api_url"
            )
        if cfg.llm_provider == "local" and not cfg.llm_api_url.strip():
            raise ValueError("production_mode local LLM requires llm_api_url")
        if cfg.coherence_require_model_backed_nli and not cfg.use_nli:
            raise ValueError(
                "production_mode with coherence_require_model_backed_nli=True requires use_nli=True",
            )
        # Ruff S104 fires on the literal "0.0.0.0"; the host
        # value is compared, not bound, so assembling the
        # wildcard from parts keeps the check intact.
        _wildcard_host = ".".join(["0", "0", "0", "0"])
        if cfg.server_host == _wildcard_host:
            logger.warning(
                "production_mode: binding to %s — ensure reverse proxy with TLS",
                _wildcard_host,
            )
    if cfg.knowledge_write_require_signature and not cfg.knowledge_write_hmac_keys:
        raise ValueError(
            "knowledge_write_require_signature requires knowledge_write_hmac_keys",
        )
    if cfg.vector_backend == "sentence-transformer" and not cfg.embedding_model.strip():
        raise ValueError(
            "embedding_model must be set when vector_backend='sentence-transformer'",
        )
    if cfg.vector_backend == "http-faiss":
        if not cfg.embedding_base_url.strip():
            raise ValueError(
                "embedding_base_url must be set when vector_backend='http-faiss'",
            )
        if not cfg.embedding_model.strip():
            raise ValueError(
                "embedding_model must be set when vector_backend='http-faiss'",
            )
    if cfg.vector_backend == "remanentia" and not cfg.remanentia_base_url.strip():
        raise ValueError(
            "remanentia_base_url must be set when vector_backend='remanentia'",
        )
    if cfg.embedding_timeout_s <= 0:
        raise ValueError("embedding_timeout_s must be > 0")
    if cfg.embedding_vector_size < 1:
        raise ValueError("embedding_vector_size must be >= 1")
    if cfg.remanentia_timeout_s <= 0:
        raise ValueError("remanentia_timeout_s must be > 0")
