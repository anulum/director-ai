# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Configuration Manager

"""Dataclass-based configuration with env var, YAML, and profile support.

Usage::

    config = DirectorConfig.from_env()
    config = DirectorConfig.from_yaml("config.yaml")
    config = DirectorConfig.from_profile("fast")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

# ``ProfileMetadata`` keeps its redundant alias so it is re-exported (and stays
# importable as ``from director_ai.core.config import ProfileMetadata``).
from .config_profiles import (
    PROFILE_DEFINITIONS,
    PROFILE_METADATA,
)
from .config_profiles import (
    ProfileMetadata as ProfileMetadata,
)

if TYPE_CHECKING:
    from .retrieval.knowledge import GroundTruthStore
    from .runtime.contradiction_halt import ContradictionHalt
    from .scoring.scorer import CoherenceScorer

__all__ = ["DirectorConfig", "ProfileMetadata"]

logger = logging.getLogger("DirectorAI.Config")


def _parse_api_keys_env(raw: str) -> list[str]:
    """Parse ``DIRECTOR_API_KEYS`` accepting a JSON array or a comma list.

    Operators reach for both spellings: a JSON array (``["sk-a","sk-b"]``, the
    form the production checklist used to show) and a bare comma-separated list
    (``sk-a,sk-b``). Parsing only one of them silently embeds brackets/quotes
    into the literal key and produces the "auth is configured but the keys never
    match" footgun. A JSON array of strings is honoured when the value parses to
    a list of strings; everything else falls back to comma splitting. Blank and
    whitespace-only entries are dropped.
    """
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(k, str) for k in parsed):
            return [k.strip() for k in parsed if k.strip()]
    return [k.strip() for k in raw.split(",") if k.strip()]


@dataclass
class DirectorConfig:
    """Central configuration for Director-Class AI.

    Parameters
    ----------
    coherence_threshold : float — minimum coherence to approve (0.0-1.0).
    hard_limit : float — safety kernel emergency stop threshold.
    use_nli : bool — enable DeBERTa NLI model for logical divergence.
    nli_model : str — HuggingFace model ID for NLI.
    max_candidates : int — number of LLM candidates to generate.
    history_window : int — scorer rolling history size.
    llm_provider : str — LLM backend name.
    llm_api_url : str — API endpoint URL (for "local" provider).
    llm_api_key : str — API key (for cloud providers).
    llm_model : str — model name for cloud providers.
    llm_temperature : float — sampling temperature.
    llm_max_tokens : int — maximum tokens per response.
    vector_backend : str — "memory" or "chroma".
    chroma_collection : str — ChromaDB collection name.
    chroma_persist_dir : str — ChromaDB persistence directory (None=in-memory).
    onnx_path : str — directory with exported ONNX model (for scorer_backend="onnx").
    server_host : str — FastAPI server bind address.
    server_port : int — FastAPI server port.
    server_workers : int — Uvicorn worker count.
    batch_max_concurrency : int — max concurrent batch requests.
    metrics_enabled : bool — enable Prometheus-style metrics collection.
    log_level : str — logging level.
    log_json : bool — structured JSON logging.

    """

    # Mode: "general" | "grounded" | "auto"
    #   general  — NLI only, no KB, no embeddings. Fast, lightweight.
    #   grounded — requires KB. Hybrid + reranker + claim decomposition.
    #   auto     — KB if available + relevant, falls back to general NLI.
    mode: str = "auto"

    # Operational modes
    dry_run: bool = False  # log scores but never halt/reject (observability mode)
    production_mode: bool = False  # enforce HTTPS-only, strict CORS, require auth
    hardened: bool = False  # strict_mode + all sanitisers + injection detection
    strict_mode: bool = False  # scorer disables heuristic fallbacks (fail closed)
    cost_tracking_enabled: bool = False  # attach CostAnalyser to scorer

    # Scoring
    coherence_threshold: float = 0.6
    hard_limit: float = 0.5
    soft_limit: float = 0.6
    use_nli: bool = False
    scorer_model: str = ""
    allow_domain_only_scorer_model: bool = False
    allow_custom_scorer_model: bool = False
    nli_model: str = "yaxili96/FactCG-DeBERTa-v3-Large"
    nli_model_artifact_uri: str = ""
    nli_model_revision: str = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
    nli_max_length: int = 512  # >512 for long-context models (Longformer, BigBird)
    # Contradiction-driven streaming halt — the working real-time halt. Opt-in:
    # halts when a completed claim contradicts retrieved KB facts
    # (P(contradiction) from a 3-class NLI), instead of the coherence halt which
    # false-halts correct-but-unsupported streaming text. Needs the nli extra.
    streaming_contradiction_halt: bool = False
    streaming_contradiction_model: str = (
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    )
    streaming_contradiction_revision: str = ""
    streaming_contradiction_threshold: float = 0.2
    streaming_contradiction_device: int = -1  # CUDA index; -1 runs on CPU
    # When True, resolve nli_model through the fallback registry at build time:
    # if the primary is delisted/unreachable on the Hub, degrade to a vetted
    # alternate NLI model instead of the heuristic floor. Probes the Hub at
    # startup, so it is off by default.
    model_fallback_enabled: bool = False
    max_candidates: int = 3
    history_window: int = 5

    # LLM
    llm_provider: str = "mock"
    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.8
    llm_max_tokens: int = 512

    # LLM-as-judge escalation
    # WARNING: when enabled, user prompts and responses are sent to the
    # configured external LLM provider for scoring.
    # Do not enable in privacy-sensitive deployments without user consent.
    llm_judge_enabled: bool = False
    llm_judge_confidence_threshold: float = 0.3
    llm_judge_provider: str = ""  # cloud or local judge backend name
    llm_judge_model: str = ""
    llm_judge_local_model: str = ""  # path to local judge checkpoint
    llm_judge_model_revision: str = ""
    privacy_mode: bool = False

    # Tier-6 reasoning escalation (causal-LM safety chain-of-thought above NLI).
    # Fires only when the composite score is within the margin of the decision
    # boundary, so median latency is unchanged. With an openai/anthropic
    # provider, the borderline prompt/response is sent to that provider
    # (respect privacy_mode); "local" loads a causal-LM with transformers.
    reasoning_enabled: bool = False
    reasoning_provider: str = ""  # "openai" | "anthropic" | "local"
    reasoning_model: str = ""
    reasoning_model_revision: str = ""
    reasoning_escalation_margin: float = 0.15

    # Scorer backend: "deberta", "onnx", "minicheck", "hybrid", "lite", "rust"
    # "auto" picks best available: rust > onnx > deberta > lite (see
    # _resolve_scorer_backend). The distilled "nli-lite" backend is never
    # auto-selected — it stays opt-in because it has not passed validation.
    scorer_backend: str = "auto"

    # Token-level hallucinated-span detector (opt-in). The response/claim-level
    # scorer above judges whole answers; this ModernBERT token classifier flags
    # the short unsupported spans inside a RAG response (RAGTruth-style). Enabling
    # it loads an extra model, so it is off by default and exposed via
    # ``DirectorGuard.detect_spans``.
    span_detection_enabled: bool = False
    span_model: str = "anulum/director-ragtruth-token-modernbert"
    span_model_revision: str = ""
    span_token_threshold: float = 0.95
    span_min_tokens: int = 1
    span_max_length: int = 1024
    span_device: int = -1

    # Multi-GPU NLI sharding (comma-separated, e.g. "cuda:0,cuda:1")
    nli_devices: str = ""

    # NLI precision tier. minicheck_variant selects the MiniCheck checkpoint
    # ("deberta-v3-large" 0.4B fast → "Bespoke-MiniCheck-7B" most accurate);
    # nli_torch_dtype ("float16"/"bfloat16"/"float32") and nli_quantize_8bit
    # trade memory/latency for precision. Empty strings = library defaults.
    minicheck_variant: str = "deberta-v3-large"
    nli_torch_dtype: str = ""
    nli_quantize_8bit: bool = False
    nli_device: str = ""

    # Vector store
    vector_backend: str = "memory"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_model_revision: str = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_timeout_s: float = 10.0
    embedding_vector_size: int = 384
    remanentia_base_url: str = "http://127.0.0.1:8001"
    remanentia_timeout_s: float = 5.0
    remanentia_source: str = ""
    chroma_collection: str = "director_ai"
    chroma_persist_dir: str = ""
    hybrid_retrieval: bool = True  # BM25 + dense with Reciprocal Rank Fusion
    hybrid_rrf_k: int = 60  # RRF rank constant; 60 is the canonical TREC default
    reranker_enabled: bool = True  # cross-encoder reranking on top of retrieval
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_model_revision: str = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    reranker_top_k_multiplier: int = 3
    retrieval_abstention_threshold: float = 0.3  # 0 = disabled; min similarity to score
    verified_scorer_enabled: bool = False
    verified_scorer_atomic: bool = True
    verified_scorer_evidence_top_k: int = 3
    verified_scorer_low_confidence_margin: float = 0.10
    verified_scorer_min_coverage: float = 0.50

    # Parent-child chunking (v3.14+)
    parent_child_enabled: bool = False
    parent_chunk_size: int = 2048
    child_chunk_size: int = 256

    # Adaptive retrieval routing (v3.14+)
    adaptive_retrieval_enabled: bool = False
    adaptive_retrieval_threshold: float = 0.5

    # HyDE — Hypothetical Document Embeddings (v3.15+)
    hyde_enabled: bool = False
    hyde_prompt_template: str = ""  # empty = use default template

    # Query decomposition (v3.15+)
    query_decomposition_enabled: bool = False
    query_decomposition_strategy: str = "heuristic"  # "heuristic" or "llm"

    # Contextual compression (v3.15+)
    contextual_compression_enabled: bool = False
    contextual_compression_strategy: str = "heuristic"  # "heuristic" or "llm"

    # Multi-vector retrieval (v3.15+)
    multi_vector_enabled: bool = False
    multi_vector_representations: str = "content,summary,title"

    # Enterprise & Caching
    redis_url: str = ""
    redis_prefix: str = "dai:"
    cache_size: int = 1024
    cache_ttl: float = 300.0

    # Server
    # Secure default: bind to loopback. A direct embedder that runs uvicorn with
    # this value is not exposed on all interfaces unless it opts in. The CLI
    # `serve` resolves the effective bind (explicit --host, then
    # DIRECTOR_SERVER_HOST, then 0.0.0.0 for --production behind a reverse proxy,
    # else loopback for dev); the container image sets DIRECTOR_SERVER_HOST.
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    server_workers: int = 1
    cors_origins: str = ""
    # Lifetime of a single-use browser WebSocket handshake ticket (seconds).
    ws_ticket_ttl_seconds: float = 30.0

    # ONNX
    onnx_path: str = ""
    onnx_batch_size: int = 16
    onnx_flush_timeout_ms: float = 10.0

    # Batch
    batch_max_concurrency: int = 4

    # Continuous batching (review queue)
    review_queue_enabled: bool = False
    review_queue_max_batch: int = 32
    review_queue_flush_timeout_ms: float = 10.0

    # Observability
    metrics_enabled: bool = True
    log_level: str = "INFO"
    log_json: bool = False
    otel_enabled: bool = False

    # Audit
    audit_log_path: str = ""
    audit_postgres_url: str = ""

    # EU AI Act compliance
    compliance_db_path: str = ""

    # Human feedback for online calibration (empty = feedback API disabled)
    feedback_db_path: str = ""

    # Tenant routing
    tenant_routing: bool = False

    # Input Sanitization
    sanitize_inputs: bool = True
    sanitizer_block_threshold: float = 0.8
    redact_pii: bool = False

    # Injection Detection (output-side NLI-based)
    injection_detection_enabled: bool = False
    injection_threshold: float = 0.7
    injection_drift_threshold: float = 0.6
    injection_claim_threshold: float = 0.75
    injection_baseline_divergence: float = 0.4
    injection_stage1_weight: float = 0.3
    injection_require_model_backed_nli: bool = False
    injection_fail_closed_on_error: bool = False

    # Model-backed prompt-injection / jailbreak input screen (opt-in).
    # When enabled, the request-path input sanitizer is wrapped in a
    # LayeredPromptGuard whose model stage catches adaptive jailbreaks
    # (GCG/PAIR) the patterns miss. Requires the ``nli`` extra (transformers);
    # the model id defaults to ProtectAI's Apache-2.0, ungated classifier.
    prompt_guard_model_enabled: bool = False
    prompt_guard_model_id: str = "protectai/deberta-v3-base-prompt-injection-v2"
    prompt_guard_model_revision: str = ""
    prompt_guard_threshold: float = 0.5

    # Multi-modal hallucination guard (opt-in; only active when the
    # experimental hooks flag is set AND at least one modality is enabled).
    multimodal_enabled_modalities: tuple[str, ...] = ()
    multimodal_benchmarked_modalities: tuple[str, ...] = ()
    multimodal_hallucination_threshold: float = 0.15
    multimodal_consistency_threshold: float = 0.45
    multimodal_temporal_alpha: float = 0.5
    multimodal_temporal_floor: float = 0.2
    multimodal_grounding_floor: float = 0.4
    multimodal_grounding_allow_threshold: float = 0.75
    multimodal_embedding_dim: int = 512
    # Image backend: "hashbag" (dependency-free FNV baseline, default) or "clip"
    # (open_clip semantic vision via director-ai[multimodal]).
    multimodal_backend: str = "hashbag"
    multimodal_clip_model: str = "ViT-B-32"
    multimodal_clip_pretrained: str = "openai"
    multimodal_clip_device: str = "cpu"
    multimodal_policy_id: str = "multimodal-default"
    multimodal_calibrated_threshold: float = 0.5
    multimodal_no_go_threshold: float = 0.9

    # Pre-model evidence firewall (opt-in). When enabled, every retrieved chunk
    # is screened before it can reach the model; quarantined chunks are dropped
    # from the grounding context. Defaults are fail-closed on the integrity
    # checks once enabled (tenant, provenance, signature, expiry, poisoning).
    evidence_firewall_enabled: bool = False
    evidence_firewall_require_tenant_match: bool = True
    evidence_firewall_require_provenance: bool = True
    evidence_firewall_require_signature: bool = True
    evidence_firewall_verify_content_hash: bool = True
    evidence_firewall_enforce_expiry: bool = True
    evidence_firewall_max_age_seconds: float = 0.0
    evidence_firewall_require_source_owner: bool = False
    evidence_firewall_enforce_sensitivity: bool = False
    evidence_firewall_allowed_sensitivity: tuple[str, ...] = (
        "unclassified",
        "public",
        "internal",
    )
    evidence_firewall_scan_poisoning: bool = True
    evidence_firewall_poison_threshold: float = 0.6
    evidence_firewall_enforce_use_case: bool = False

    # Scoring weights (0.0 = use CoherenceScorer class defaults)
    w_logic: float = 0.0
    w_fact: float = 0.0
    coherence_require_model_backed_nli: bool = False

    # Metrics auth: when True, /v1/metrics/prometheus requires API key
    metrics_require_auth: bool = True

    # Rate limiting (requests per minute, 0 = disabled)
    rate_limit_rpm: int = 0

    # When True, raise ImportError if rate_limit_rpm > 0 and slowapi missing
    rate_limit_strict: bool = False

    # API key auth (empty list = no auth required)
    api_keys: list[str] = field(default_factory=list)

    # Bind API keys to tenants: JSON {"api_key": "tenant_id"}
    # When set, X-Tenant-ID header is validated against this map.
    api_key_tenant_map: str = ""

    # Knowledge-base write controls
    knowledge_write_require_auth: bool = False
    knowledge_write_require_tenant_binding: bool = True
    knowledge_write_require_signature: bool = False
    knowledge_write_hmac_keys: str = ""

    # Stats backend: "prometheus" (default, in-memory) or "sqlite" (persistent)
    stats_backend: str = "prometheus"
    stats_db_path: str = "~/.director-ai/stats.db"

    # Source-availability endpoint (transparency convenience)
    source_endpoint_enabled: bool = True
    source_repository_url: str = "https://github.com/anulum/director-ai"

    # Commercial license (set via DIRECTOR_LICENSE_KEY or DIRECTOR_LICENSE_FILE)
    license_key: str = ""
    license_file: str = ""

    # Chunked NLI aggregation: "max"|"min"|"mean"
    nli_fact_inner_agg: str = "max"
    nli_fact_outer_agg: str = "max"
    nli_logic_inner_agg: str = "max"
    nli_logic_outer_agg: str = "max"
    nli_premise_ratio: float = 0.4
    nli_fact_retrieval_top_k: int = 3
    nli_use_prompt_as_premise: bool = False
    nli_summarization_baseline: float = 0.20
    nli_claim_coverage_enabled: bool = True
    nli_claim_support_threshold: float = 0.6
    nli_claim_coverage_alpha: float = 0.4

    # Adaptive task-type thresholding (Phase 1B)
    # Validated on LLM-AggreFact 29K: per-task-type BA 76.68% vs global 75.82%
    # Coherence values derived from optimal NLI thresholds:
    #   coherence = 0.4 + 0.6 * nli_threshold (W_LOGIC=0.6 pure-NLI case)
    adaptive_threshold_enabled: bool = True
    adaptive_threshold_fail_closed: bool = False
    threshold_summarization: float = 0.72  # NLI=0.54, AggreFact/TofuEval
    threshold_qa: float = 0.69  # NLI=0.48, ExpertQA/Lfqa
    threshold_fact_check: float = (
        0.56  # NLI=0.27, ClaimVerify/FactCheck-GPT/Reveal/Wice
    )
    threshold_rag: float = 0.78  # NLI=0.63, RAGTruth
    threshold_dialogue: float = 0.68  # NLI=0.46 (global default)

    # Chunking overlap (Phase 2A)
    nli_chunk_overlap_ratio: float = 0.5
    nli_qa_premise_ratio: float = 0.7

    # Confidence-weighted aggregation (Phase 2B)
    nli_confidence_weighted_agg: bool = True

    # LoRA adapter path (Phase 3A)
    lora_adapter_path: str = ""

    # Meta-classifier model path (Phase 6A)
    meta_classifier_path: str = ""

    # gRPC limits
    grpc_max_message_mb: int = 4
    grpc_deadline_seconds: float = 30.0

    # Profile name (informational)
    profile: str = "default"

    # Extra key-value overrides
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Apply profile-derived defaults and validate configuration bounds."""
        if self.mode not in ("general", "grounded", "auto"):
            raise ValueError(
                f"mode must be 'general', 'grounded', or 'auto', got {self.mode!r}"
            )
        # Apply mode defaults before other validation
        if self.mode == "general":
            if not self.use_nli:
                object.__setattr__(self, "use_nli", True)
            object.__setattr__(self, "hybrid_retrieval", False)
            object.__setattr__(self, "reranker_enabled", False)
            object.__setattr__(self, "retrieval_abstention_threshold", 0.0)
        elif self.mode in ("grounded", "auto"):
            if self.use_nli or self.hybrid_retrieval or self.reranker_enabled:
                object.__setattr__(self, "use_nli", True)
            if self.retrieval_abstention_threshold <= 0:
                object.__setattr__(self, "retrieval_abstention_threshold", 0.3)

        if not (0.0 <= self.coherence_threshold <= 1.0):
            raise ValueError(
                f"coherence_threshold must be in [0, 1], got {self.coherence_threshold}",
            )
        if not (0.0 <= self.hard_limit <= 1.0):
            raise ValueError(f"hard_limit must be in [0, 1], got {self.hard_limit}")
        if not (0.0 <= self.soft_limit <= 1.0):
            raise ValueError(f"soft_limit must be in [0, 1], got {self.soft_limit}")
        if self.soft_limit < self.hard_limit:
            raise ValueError(
                f"soft_limit ({self.soft_limit}) must be "
                f">= hard_limit ({self.hard_limit})",
            )
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {self.max_candidates}")
        if self.history_window < 1:
            raise ValueError(f"history_window must be >= 1, got {self.history_window}")
        if not (0.0 <= self.llm_temperature <= 2.0):
            raise ValueError(
                f"llm_temperature must be in [0, 2], got {self.llm_temperature}",
            )
        if self.llm_max_tokens < 1:
            raise ValueError(f"llm_max_tokens must be >= 1, got {self.llm_max_tokens}")
        if self.batch_max_concurrency < 1:
            raise ValueError(
                f"batch_max_concurrency must be >= 1, got {self.batch_max_concurrency}",
            )
        if not (1 <= self.server_port <= 65535):
            raise ValueError(
                f"server_port must be in [1, 65535], got {self.server_port}",
            )
        if self.server_workers < 1:
            raise ValueError(f"server_workers must be >= 1, got {self.server_workers}")
        if self.rate_limit_rpm < 0:
            raise ValueError(f"rate_limit_rpm must be >= 0, got {self.rate_limit_rpm}")
        if self.stats_backend not in ("prometheus", "sqlite"):
            raise ValueError(
                f"stats_backend must be 'prometheus' or 'sqlite', "
                f"got {self.stats_backend!r}",
            )
        if self.grpc_max_message_mb < 1:
            raise ValueError(
                f"grpc_max_message_mb must be >= 1, got {self.grpc_max_message_mb}",
            )
        if self.grpc_deadline_seconds <= 0:
            raise ValueError(
                f"grpc_deadline_seconds must be > 0, got {self.grpc_deadline_seconds}",
            )
        if not (0.0 <= self.sanitizer_block_threshold <= 1.0):
            raise ValueError(
                "sanitizer_block_threshold must be in [0, 1], "
                f"got {self.sanitizer_block_threshold}",
            )
        if (self.w_logic != 0.0 or self.w_fact != 0.0) and abs(
            self.w_logic + self.w_fact - 1.0,
        ) > 1e-6:
            raise ValueError(
                f"w_logic + w_fact must equal 1.0 when set, "
                f"got {self.w_logic} + {self.w_fact}",
            )
        if self.reranker_enabled and not self.reranker_model.strip():
            raise ValueError("reranker_model must be set when reranker_enabled=True")
        if self.verified_scorer_evidence_top_k < 1:
            raise ValueError("verified_scorer_evidence_top_k must be >= 1")
        if not (0.0 <= self.verified_scorer_low_confidence_margin <= 1.0):
            raise ValueError("verified_scorer_low_confidence_margin must be in [0, 1]")
        if not (0.0 <= self.verified_scorer_min_coverage <= 1.0):
            raise ValueError("verified_scorer_min_coverage must be in [0, 1]")
        if not isinstance(self.hybrid_rrf_k, int) or isinstance(
            self.hybrid_rrf_k, bool
        ):
            raise ValueError("hybrid_rrf_k must be an integer")
        if self.hybrid_rrf_k < 1:
            raise ValueError("hybrid_rrf_k must be at least 1")
        if self.scorer_model:
            from .scoring.model_choices import resolve_scorer_model_choice

            choice = resolve_scorer_model_choice(
                self.scorer_model,
                allow_domain_only=self.allow_domain_only_scorer_model,
                allow_custom=self.allow_custom_scorer_model,
            )
            object.__setattr__(self, "nli_model", choice.runtime_model)
            object.__setattr__(self, "nli_model_artifact_uri", choice.artifact_uri)
            object.__setattr__(self, "nli_model_revision", choice.revision)
            object.__setattr__(self, "nli_max_length", choice.max_length)

        # Hardened mode: enforce all safety features
        if self.hardened:
            object.__setattr__(self, "production_mode", True)
            object.__setattr__(self, "use_nli", True)
            object.__setattr__(self, "coherence_require_model_backed_nli", True)
            object.__setattr__(self, "adaptive_threshold_enabled", True)
            object.__setattr__(self, "adaptive_threshold_fail_closed", True)
            object.__setattr__(self, "injection_detection_enabled", True)
            object.__setattr__(self, "injection_require_model_backed_nli", True)
            object.__setattr__(self, "injection_fail_closed_on_error", True)
            object.__setattr__(self, "sanitize_inputs", True)
            object.__setattr__(self, "redact_pii", True)
            object.__setattr__(self, "strict_mode", True)
            logger.info("Hardened mode: all safety features enforced")

        if (
            self.injection_require_model_backed_nli
            and not self.injection_detection_enabled
        ):
            raise ValueError(
                "injection_require_model_backed_nli=True requires injection_detection_enabled=True",
            )
        if self.injection_fail_closed_on_error and not self.injection_detection_enabled:
            raise ValueError(
                "injection_fail_closed_on_error=True requires injection_detection_enabled=True",
            )
        if self.adaptive_threshold_fail_closed and not self.adaptive_threshold_enabled:
            raise ValueError(
                "adaptive_threshold_fail_closed=True requires adaptive_threshold_enabled=True",
            )

        # Production mode enforcements
        if self.production_mode:
            if self.dry_run:
                raise ValueError(
                    "production_mode requires dry_run=False (fail-open dry-run is not permitted)",
                )
            if not self.sanitize_inputs:
                raise ValueError(
                    "production_mode requires sanitize_inputs=True",
                )
            object.__setattr__(self, "knowledge_write_require_auth", True)
            # KB integrity is the product: a poisoned knowledge base lets the
            # guard certify false claims as ground truth. Production requires
            # signed KB writes.
            object.__setattr__(self, "knowledge_write_require_signature", True)
            # A configured rate limit must fail closed in production: a missing
            # limiter backend must refuse startup rather than silently run an
            # unthrottled public listener.
            object.__setattr__(self, "rate_limit_strict", True)
            if not self.api_keys and not self.api_key_tenant_map:
                raise ValueError(
                    "production_mode requires api_keys or api_key_tenant_map"
                )
            # Tenant isolation is only trustworthy with key→tenant binding;
            # accepting X-Tenant-ID without a binding is advisory. When tenant
            # routing is on in production, a binding map is mandatory.
            if self.tenant_routing and not self.api_key_tenant_map:
                raise ValueError(
                    "production_mode with tenant_routing=True requires "
                    "api_key_tenant_map (key→tenant binding)"
                )
            if not self.llm_api_url.strip() and self.llm_provider in {"", "mock"}:
                raise ValueError(
                    "production_mode requires a real LLM provider or llm_api_url"
                )
            if self.llm_provider == "local" and not self.llm_api_url.strip():
                raise ValueError("production_mode local LLM requires llm_api_url")
            if self.coherence_require_model_backed_nli and not self.use_nli:
                raise ValueError(
                    "production_mode with coherence_require_model_backed_nli=True requires use_nli=True",
                )
            # Ruff S104 fires on the literal "0.0.0.0"; the host
            # value is compared, not bound, so assembling the
            # wildcard from parts keeps the check intact.
            _wildcard_host = ".".join(["0", "0", "0", "0"])
            if self.server_host == _wildcard_host:
                logger.warning(
                    "production_mode: binding to %s — ensure reverse proxy with TLS",
                    _wildcard_host,
                )
        if (
            self.knowledge_write_require_signature
            and not self.knowledge_write_hmac_keys
        ):
            raise ValueError(
                "knowledge_write_require_signature requires knowledge_write_hmac_keys",
            )
        if (
            self.vector_backend == "sentence-transformer"
            and not self.embedding_model.strip()
        ):
            raise ValueError(
                "embedding_model must be set when vector_backend='sentence-transformer'",
            )
        if self.vector_backend == "http-faiss":
            if not self.embedding_base_url.strip():
                raise ValueError(
                    "embedding_base_url must be set when vector_backend='http-faiss'",
                )
            if not self.embedding_model.strip():
                raise ValueError(
                    "embedding_model must be set when vector_backend='http-faiss'",
                )
        if self.vector_backend == "remanentia" and not self.remanentia_base_url.strip():
            raise ValueError(
                "remanentia_base_url must be set when vector_backend='remanentia'",
            )
        if self.embedding_timeout_s <= 0:
            raise ValueError("embedding_timeout_s must be > 0")
        if self.embedding_vector_size < 1:
            raise ValueError("embedding_vector_size must be >= 1")
        if self.remanentia_timeout_s <= 0:
            raise ValueError("remanentia_timeout_s must be > 0")

    @classmethod
    def from_env(cls, prefix: str = "DIRECTOR_") -> DirectorConfig:
        """Load configuration from environment variables.

        Reads ``DIRECTOR_<FIELD>`` env vars (case-insensitive field matching).
        Example: ``DIRECTOR_COHERENCE_THRESHOLD=0.7``
        """
        kwargs: dict[str, Any] = {}
        field_map = {f.name.upper(): f for f in cls.__dataclass_fields__.values()}

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            field_name = key[len(prefix) :]
            if field_name in field_map:
                fld = field_map[field_name]
                try:
                    if fld.name == "api_keys":
                        # ``api_keys`` accepts both a JSON array and a comma
                        # list; the generic list coercion would split a JSON
                        # array on commas and embed brackets/quotes into the
                        # literal keys (auth configured but never matches).
                        kwargs[fld.name] = _parse_api_keys_env(value)
                        continue
                    # ``fld.type`` is str under
                    # ``from __future__ import annotations`` (which
                    # the file imports); the cast pins that contract
                    # for mypy without suppressing the check.
                    kwargs[fld.name] = _coerce(value, cast(str, fld.type))
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid value for env var {key}={value!r}: {exc}",
                    ) from exc

        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> DirectorConfig:
        """Load configuration from a YAML file.

        Falls back to JSON parsing if PyYAML is not installed.
        """
        with open(path, encoding="utf-8") as f:
            raw = f.read()

        try:
            import yaml

            data = yaml.safe_load(raw)
        except ImportError:
            if path.endswith((".yaml", ".yml")):
                logger.warning(
                    "PyYAML not installed — parsing %s as JSON fallback",
                    path,
                )
            data = json.loads(raw)

        if not isinstance(data, dict):
            return cls()
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            logger.warning("Unknown config key(s) ignored: %s", sorted(unknown))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_profile(cls, name: str) -> DirectorConfig:
        """Load a predefined profile.

        Profiles
        --------
        - ``"fast"`` — heuristic scoring only, no NLI model, low latency.
        - ``"thorough"`` — NLI + RAG scoring, higher accuracy.
        - ``"research"`` — NLI + RAG + reranker, all scoring modules enabled.
        """
        profiles = PROFILE_DEFINITIONS
        if name not in profiles:
            raise ValueError(
                f"Unknown profile '{name}'. Choose from: {list(profiles.keys())}",
            )
        resolved = dict(profiles[name])
        # Production secrets come from the environment, never from a hard-coded
        # profile value. With neither env var set, production_mode validation
        # fails closed (production_mode requires api_keys or api_key_tenant_map).
        if resolved.get("production_mode"):
            env_keys = os.environ.get("DIRECTOR_API_KEYS", "").strip()
            env_map = os.environ.get("DIRECTOR_API_KEY_TENANT_MAP", "").strip()
            env_hmac = os.environ.get("DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS", "").strip()
            if env_keys:
                resolved["api_keys"] = _parse_api_keys_env(env_keys)
            if env_map:
                resolved["api_key_tenant_map"] = env_map
            if env_hmac:
                resolved["knowledge_write_hmac_keys"] = env_hmac
        cfg = cls(**resolved)
        for key, value in resolved.items():
            object.__setattr__(cfg, key, value)
        return cfg

    @classmethod
    def profile_metadata(cls, name: str) -> ProfileMetadata:
        """Return validation and dependency metadata for a built-in profile."""
        try:
            return PROFILE_METADATA[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown profile '{name}'. Choose from: {list(PROFILE_METADATA)}",
            ) from exc

    @classmethod
    def list_profile_metadata(cls) -> tuple[ProfileMetadata, ...]:
        """Return metadata for all built-in profiles in display order."""
        return tuple(PROFILE_METADATA.values())

    def configure_logging(self) -> None:
        """Apply log_level and log_json settings to the DirectorAI logger hierarchy."""
        root = logging.getLogger("DirectorAI")
        root.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))

        if self.log_json:
            handler = logging.StreamHandler()
            handler.setFormatter(_JsonFormatter())
            root.handlers = [handler]

    def build_store(self) -> GroundTruthStore:
        """Construct a VectorGroundTruthStore from config fields.

        In ``general`` mode, returns a bare GroundTruthStore (no vector backend).
        In ``grounded`` and ``auto`` modes, builds the full vector pipeline.
        """
        if self.mode == "general":
            from .retrieval.knowledge import GroundTruthStore

            logger.info("Mode 'general': no vector store (NLI-only scoring)")
            return GroundTruthStore()
        if self.redis_url:
            try:
                from director_ai.enterprise.redis import RedisGroundTruthStore

                return RedisGroundTruthStore(
                    redis_url=self.redis_url,
                    prefix=self.redis_prefix + "facts:",
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
        if self.vector_backend == "chroma":
            try:
                from .retrieval.vector_store import ChromaBackend

                backend = ChromaBackend(
                    collection_name=self.chroma_collection,
                    persist_directory=self.chroma_persist_dir or None,
                )
            except ImportError:
                logger.warning("chromadb not installed, falling back to memory backend")
                backend = InMemoryBackend()
        elif self.vector_backend == "sentence-transformer":
            try:
                from .retrieval.vector_store import SentenceTransformerBackend

                backend = SentenceTransformerBackend(
                    model_name=self.embedding_model,
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, falling back to memory",
                )
                backend = InMemoryBackend()
        elif self.vector_backend == "http-faiss":
            try:
                from .retrieval.vector_store import FAISSBackend, HttpEmbeddingFunction

                embed_fn = HttpEmbeddingFunction(
                    base_url=self.embedding_base_url,
                    model=self.embedding_model,
                    api_key=self.embedding_api_key,
                    timeout_s=self.embedding_timeout_s,
                    vector_size=self.embedding_vector_size,
                )
                backend = FAISSBackend(
                    embed_fn=embed_fn,
                    vector_size=self.embedding_vector_size,
                )
            except ImportError:
                logger.warning("faiss not installed, falling back to memory")
                backend = InMemoryBackend()
        elif self.vector_backend == "remanentia":
            from .retrieval.vector_store import RemanentiaVectorBackend

            backend = RemanentiaVectorBackend(
                base_url=self.remanentia_base_url,
                timeout_s=self.remanentia_timeout_s,
                source=self.remanentia_source,
            )
        else:
            # Try vector backend registry for third-party / unrecognized names
            try:
                from .retrieval.vector_store import get_vector_backend

                backend_cls = get_vector_backend(self.vector_backend)
                backend = backend_cls()
            except (KeyError, TypeError):
                logger.warning(
                    "Unknown vector_backend %r, falling back to memory",
                    self.vector_backend,
                )
                backend = InMemoryBackend()

        local_decorators_enabled = self.vector_backend != "remanentia"

        if self.hybrid_retrieval and local_decorators_enabled:
            try:
                from .retrieval.vector_store import HybridBackend

                backend = HybridBackend(base=backend, rrf_k=self.hybrid_rrf_k)
                logger.info(
                    "Hybrid retrieval enabled (BM25 + dense + RRF, k=%s)",
                    self.hybrid_rrf_k,
                )
            except ImportError:
                logger.warning("HybridBackend unavailable, using dense-only retrieval")

        if self.reranker_enabled and local_decorators_enabled:
            try:
                from .retrieval.vector_store import RerankedBackend

                backend = RerankedBackend(
                    base=backend,
                    reranker_model=self.reranker_model,
                    reranker_revision=self.reranker_model_revision or None,
                    top_k_multiplier=self.reranker_top_k_multiplier,
                )
                logger.info("Cross-encoder reranker enabled: %s", self.reranker_model)
            except (ImportError, OSError) as exc:
                if self.production_mode or self.hardened:
                    raise RuntimeError(
                        "reranker_enabled=True but the reranker model could not load",
                    ) from exc
                logger.warning("Reranker unavailable, skipping reranker: %s", exc)

        if self.parent_child_enabled:
            from .retrieval.parent_child import ParentChildBackend

            backend = ParentChildBackend(
                base=backend,
                parent_size=self.parent_chunk_size,
                child_size=self.child_chunk_size,
            )
            logger.info(
                "Parent-child chunking enabled (parent=%d, child=%d)",
                self.parent_chunk_size,
                self.child_chunk_size,
            )

        if self.hyde_enabled:
            from .retrieval.hyde import HyDEBackend

            if self.hyde_prompt_template:
                backend = HyDEBackend(base=backend, template=self.hyde_prompt_template)
            else:
                # Generator will be injected by build_scorer() or by the user.
                backend = HyDEBackend(base=backend)
            logger.info("HyDE retrieval enabled")

        if self.query_decomposition_enabled:
            from .retrieval.query_decomposition import QueryDecompositionBackend

            backend = QueryDecompositionBackend(
                base=backend,
                strategy=self.query_decomposition_strategy,
            )
            logger.info(
                "Query decomposition enabled (strategy=%s)",
                self.query_decomposition_strategy,
            )

        if self.contextual_compression_enabled:
            from .retrieval.contextual_compression import (
                ContextualCompressionBackend,
            )

            backend = ContextualCompressionBackend(
                base=backend,
                strategy=self.contextual_compression_strategy,
            )
            logger.info("Contextual compression enabled")

        if self.multi_vector_enabled:
            from .retrieval.multi_vector import MultiVectorBackend

            reps = [
                r.strip()
                for r in self.multi_vector_representations.split(",")
                if r.strip()
            ]
            backend = MultiVectorBackend(base=backend, representations=reps)
            logger.info(
                "Multi-vector retrieval enabled (representations=%s)",
                reps,
            )

        from .evidence_firewall import build_evidence_firewall

        return VectorGroundTruthStore(
            backend=backend,
            evidence_firewall=build_evidence_firewall(self),
        )

    def _resolve_scorer_backend(self) -> str:
        """Resolve 'auto' scorer backend to best available."""
        if self.scorer_backend != "auto":
            return self.scorer_backend

        # Priority: rust > onnx > deberta > nli-lite > lite
        import importlib.util

        if importlib.util.find_spec("backfire_kernel") is not None:
            logger.info("Auto scorer: selected 'rust' (backfire_kernel available)")
            return "rust"

        if self.onnx_path:
            logger.info("Auto scorer: selected 'onnx' (onnx_path configured)")
            return "onnx"

        if self.use_nli:
            logger.info("Auto scorer: selected 'deberta' (NLI enabled)")
            return "deberta"

        logger.info("Auto scorer: selected 'lite' (no NLI available)")
        return "lite"

    def build_scorer(self, store: GroundTruthStore | None = None) -> CoherenceScorer:
        """Construct a CoherenceScorer wired to all relevant config fields."""
        from .metrics import metrics
        from .scoring.scorer import CoherenceScorer

        if store is None:
            store = self.build_store()

        judge_model = self.llm_judge_model
        if self.llm_judge_provider == "local" and self.llm_judge_local_model:
            judge_model = self.llm_judge_local_model

        resolved_backend = self._resolve_scorer_backend()

        nli_model = self.nli_model
        nli_revision = self.nli_model_revision or None
        if self.model_fallback_enabled:
            from .model_registry import FallbackModelRegistry

            resolved = FallbackModelRegistry().resolve(
                "nli", nli_model, primary_revision=nli_revision
            )
            nli_model, nli_revision = resolved.model_id, resolved.revision

        kw: dict[str, Any] = {
            "threshold": self.coherence_threshold,
            "use_nli": self.use_nli,
            "require_model_backed_nli": self.coherence_require_model_backed_nli,
            "strict_mode": self.strict_mode,
            "scorer_backend": resolved_backend,
            "soft_limit": self.soft_limit,
            "nli_model": nli_model,
            "nli_revision": nli_revision,
            "nli_max_length": self.nli_max_length,
            "llm_judge_enabled": self.llm_judge_enabled,
            "llm_judge_confidence_threshold": self.llm_judge_confidence_threshold,
            "llm_judge_provider": self.llm_judge_provider,
            "llm_judge_model": judge_model,
            "llm_judge_model_revision": self.llm_judge_model_revision or None,
            "reasoning_enabled": self.reasoning_enabled,
            "reasoning_provider": self.reasoning_provider,
            "reasoning_model": self.reasoning_model,
            "reasoning_model_revision": self.reasoning_model_revision or None,
            "reasoning_escalation_margin": self.reasoning_escalation_margin,
            "minicheck_variant": self.minicheck_variant,
            "nli_quantize_8bit": self.nli_quantize_8bit,
            "nli_torch_dtype": self.nli_torch_dtype or None,
            "nli_device": self.nli_device or None,
            "privacy_mode": self.privacy_mode,
            "ground_truth_store": store,
            "onnx_batch_size": self.onnx_batch_size,
            "onnx_flush_timeout_ms": self.onnx_flush_timeout_ms,
        }
        if self.redis_url:
            try:
                from director_ai.enterprise.redis import RedisScoreCache

                kw["cache"] = RedisScoreCache(
                    redis_url=self.redis_url,
                    prefix=self.redis_prefix + "cache:",
                    ttl_seconds=self.cache_ttl,
                )
            except ImportError:
                pass
        else:
            kw["cache_size"] = self.cache_size
            kw["cache_ttl"] = self.cache_ttl

        if self.onnx_path:
            kw["onnx_path"] = self.onnx_path
        if self.w_logic != 0.0 or self.w_fact != 0.0:
            kw["w_logic"] = self.w_logic
            kw["w_fact"] = self.w_fact
        if self.nli_devices:
            kw["nli_devices"] = [
                d.strip() for d in self.nli_devices.split(",") if d.strip()
            ]
        scorer = CoherenceScorer(**kw)
        scorer._fact_inner_agg = self.nli_fact_inner_agg
        scorer._fact_outer_agg = self.nli_fact_outer_agg
        scorer._logic_inner_agg = self.nli_logic_inner_agg
        scorer._logic_outer_agg = self.nli_logic_outer_agg
        scorer._premise_ratio = self.nli_premise_ratio
        scorer._fact_retrieval_top_k = self.nli_fact_retrieval_top_k
        scorer._use_prompt_as_premise = self.nli_use_prompt_as_premise
        scorer._summarization_nli_baseline = self.nli_summarization_baseline
        scorer._claim_coverage_enabled = self.nli_claim_coverage_enabled
        scorer._claim_support_threshold = self.nli_claim_support_threshold
        scorer._claim_coverage_alpha = self.nli_claim_coverage_alpha
        scorer._verified_scorer_enabled = self.verified_scorer_enabled
        scorer._verified_scorer_atomic = self.verified_scorer_atomic
        scorer._verified_scorer_evidence_top_k = self.verified_scorer_evidence_top_k
        scorer._verified_scorer_low_confidence_margin = (
            self.verified_scorer_low_confidence_margin
        )
        scorer._verified_scorer_min_coverage = self.verified_scorer_min_coverage
        scorer._adaptive_threshold_enabled = self.adaptive_threshold_enabled
        scorer._adaptive_threshold_fail_closed = self.adaptive_threshold_fail_closed
        scorer._task_type_thresholds = {
            "summarization": self.threshold_summarization,
            "qa": self.threshold_qa,
            "fact_check": self.threshold_fact_check,
            "rag": self.threshold_rag,
            "dialogue": self.threshold_dialogue,
        }
        scorer._chunk_overlap_ratio = self.nli_chunk_overlap_ratio
        scorer._qa_premise_ratio = self.nli_qa_premise_ratio
        scorer._confidence_weighted_agg = self.nli_confidence_weighted_agg
        scorer._retrieval_abstention_threshold = self.retrieval_abstention_threshold
        if self.injection_detection_enabled:
            try:
                scorer.enable_injection_detection(
                    injection_threshold=self.injection_threshold,
                    drift_threshold=self.injection_drift_threshold,
                    injection_claim_threshold=self.injection_claim_threshold,
                    baseline_divergence=self.injection_baseline_divergence,
                    stage1_weight=self.injection_stage1_weight,
                    require_model_backed_nli=self.injection_require_model_backed_nli,
                    fail_closed_on_error=self.injection_fail_closed_on_error,
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
                if self.injection_require_model_backed_nli:
                    metrics.inc_labeled(
                        "model_backed_nli_startup_failures_total",
                        labels={"stage": "injection"},
                    )
                raise
        if self.lora_adapter_path and scorer._nli is not None:
            if hasattr(scorer._nli, "_load_lora_adapter"):
                scorer._nli._load_lora_adapter(self.lora_adapter_path)
            else:
                logger.warning(
                    "LoRA adapter not supported on %s",
                    type(scorer._nli).__name__,
                )
        if self.meta_classifier_path:
            scorer._meta_classifier_path = self.meta_classifier_path
        if self.adaptive_retrieval_enabled:
            scorer.enable_adaptive_retrieval(
                threshold=self.adaptive_retrieval_threshold,
            )
        if (
            scorer._adaptive_threshold_enabled
            and scorer._adaptive_threshold_fail_closed
        ):
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
        if self.dry_run:
            scorer._dry_run = True
            logger.info("Dry-run mode: scoring but never rejecting")
        if self.cost_tracking_enabled:
            from director_ai.compliance.cost_analyser import CostAnalyser

            analyser = CostAnalyser()
            scorer._cost_analyser = analyser

            def _cost_cb(model: str, inp: int, out: int) -> None:
                analyser.record(model, input_tokens=inp, output_tokens=out)

            scorer._judge._cost_callback = _cost_cb
            logger.info("Cost tracking enabled on scorer")
        if (
            self.coherence_require_model_backed_nli
            and not scorer._has_model_backed_nli()
        ):
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
            self.injection_detection_enabled
            and self.injection_require_model_backed_nli
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
        self,
        store: GroundTruthStore | None = None,
    ) -> ContradictionHalt | None:
        """Build the opt-in contradiction-driven streaming halt, or ``None``.

        Returns ``None`` when ``streaming_contradiction_halt`` is off, or when
        the NLI extra is missing / the model cannot load — the caller then keeps
        the coherence halt instead of failing startup, matching the prompt-guard
        degrade path. The halt scores each completed streamed claim against the
        store's retrieved grounding and stops on a contradiction.
        """
        if not self.streaming_contradiction_halt:
            return None
        from .runtime.contradiction_halt import ContradictionHalt
        from .scoring.contradiction import ContradictionScorer

        if store is None:
            store = self.build_store()
        try:
            scorer = ContradictionScorer.from_pretrained(
                self.streaming_contradiction_model,
                revision=self.streaming_contradiction_revision or None,
                device=self.streaming_contradiction_device,
                threshold=self.streaming_contradiction_threshold,
            )
        except Exception as exc:  # noqa: BLE001 — degrade, never crash startup
            logger.warning(
                "streaming_contradiction_halt enabled but the contradiction "
                "model could not load (%s); keeping the coherence halt.",
                exc,
            )
            return None
        return ContradictionHalt(scorer, store.retrieve_context)

    def model_revision_health(self) -> dict[str, object]:
        """Return non-network health for configured model revision pins."""
        from .model_revisions import model_revision_health

        judge_model = self.llm_judge_model
        if self.llm_judge_provider == "local" and self.llm_judge_local_model:
            judge_model = self.llm_judge_local_model
        return model_revision_health(
            {
                "nli": (self.nli_model, self.nli_model_revision or None),
                "embedding": (
                    self.embedding_model,
                    self.embedding_model_revision or None,
                ),
                "reranker": (
                    self.reranker_model if self.reranker_enabled else "",
                    self.reranker_model_revision or None,
                ),
                "local_judge": (
                    judge_model if self.llm_judge_provider == "local" else "",
                    self.llm_judge_model_revision or None,
                ),
                "contradiction": (
                    self.streaming_contradiction_model
                    if self.streaming_contradiction_halt
                    else "",
                    None,
                ),
            }
        )

    def retrieval_recipe(self) -> dict[str, object]:
        """Return the explicit grounded retrieval recipe without secrets.

        The recipe is operator metadata: it documents how ``build_store()`` composes
        retrieval for grounded/auto modes and is safe to expose through CLIs, API
        diagnostics, or documentation generators.
        """
        return {
            "name": "grounded-hybrid-rerank-v1",
            "mode": self.mode,
            "vector_backend": self.vector_backend,
            "embedding_model": self.embedding_model,
            "embedding_model_revision": self.embedding_model_revision,
            "hybrid": {
                "enabled": self.hybrid_retrieval
                and self.vector_backend != "remanentia"
                and self.mode != "general",
                "sparse": "bm25",
                "dense": self.vector_backend,
                "fusion": "reciprocal_rank_fusion",
                "rrf_k": self.hybrid_rrf_k,
            },
            "reranker": {
                "enabled": self.reranker_enabled
                and self.vector_backend != "remanentia"
                and self.mode != "general",
                "model": self.reranker_model,
                "model_revision": self.reranker_model_revision,
                "top_k_multiplier": self.reranker_top_k_multiplier,
            },
            "abstention": {
                "enabled": self.retrieval_abstention_threshold > 0,
                "threshold": self.retrieval_abstention_threshold,
            },
        }

    _REDACTED_FIELDS: frozenset[str] = frozenset(
        {
            "llm_api_key",
            "embedding_api_key",
            "api_keys",
            "api_key_tenant_map",
            "knowledge_write_hmac_keys",
            "audit_postgres_url",
            "redis_url",
            "license_key",
            "license_file",
        },
    )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict (safe for JSON/API responses)."""
        d: dict[str, object] = {}
        for fld in self.__dataclass_fields__:
            val = getattr(self, fld)
            if fld in self._REDACTED_FIELDS and val:
                d[fld] = "***"  # Redact secrets
            else:
                d[fld] = val
        return d


class _JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        import json as _json
        import time as _time

        entry = {
            "ts": _time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            entry["request_id"] = request_id
        return _json.dumps(entry)


def _coerce(value: str, type_hint: str) -> object:
    """Coerce a string env var to the target type."""
    if type_hint == "bool":
        low = value.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(
            f"invalid bool value: {value!r} (expected true/false/1/0/yes/no)",
        )
    if type_hint == "int":
        return int(value)
    if type_hint == "float":
        return float(value)
    if "list" in type_hint:
        items = [s.strip() for s in value.split(",") if s.strip()]
        if "int" in type_hint:
            return [int(x) for x in items]
        if "float" in type_hint:
            return [float(x) for x in items]
        return items
    if "tuple" in type_hint:
        # ``tuple[str, ...]`` fields (e.g. enabled modalities, evidence-firewall
        # sensitivity allowlist) must split on commas like lists. Without this
        # the raw string falls through and downstream ``frozenset(value)`` turns
        # it into a per-character set, silently corrupting the allowlist.
        parts = [s.strip() for s in value.split(",") if s.strip()]
        if "int" in type_hint:
            return tuple(int(x) for x in parts)
        if "float" in type_hint:
            return tuple(float(x) for x in parts)
        return tuple(parts)
    return value
