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
from .config_builders import build_contradiction_halt as _build_contradiction_halt
from .config_builders import build_correctness_feedback as _build_correctness_feedback
from .config_builders import build_scorer as _build_scorer
from .config_builders import build_store as _build_store
from .config_builders import resolve_scorer_backend as _resolve_scorer_backend
from .config_env import coerce_env_value as _coerce
from .config_env import parse_api_keys_env as _parse_api_keys_env
from .config_logging import JsonLogFormatter
from .config_profiles import (
    PROFILE_DEFINITIONS,
    PROFILE_METADATA,
)
from .config_profiles import (
    ProfileMetadata as ProfileMetadata,
)
from .config_validation import validate_and_normalize

if TYPE_CHECKING:
    from .calibration.recall_correctness_client import RemanentiaCorrectnessClient
    from .retrieval.knowledge import GroundTruthStore
    from .runtime.contradiction_halt import ContradictionHalt
    from .scoring.scorer import CoherenceScorer

__all__ = ["DirectorConfig", "ProfileMetadata"]

logger = logging.getLogger("DirectorAI.Config")


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
    llm_judge_rubric: bool = False  # rubric-scored judge (G-Eval-style dims)
    llm_judge_ensemble: int = 1  # independent judge calls per escalation (1-5)
    # FActScore-style LLM claim decomposition ("" = regex sentence split).
    # Sends passage text to the named provider - respect privacy_mode.
    claim_decomposition_provider: str = ""
    claim_decomposition_model: str = ""
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
    # Opt-in: post each verification verdict back to REMANENTIA as the
    # was_correct label for the recall that grounded the answer (two-label loop).
    # Requires vector_backend == "remanentia"; the token authenticates /recall.
    remanentia_correctness_feedback: bool = False
    remanentia_token: str = ""
    chroma_collection: str = "director_ai"
    chroma_persist_dir: str = ""
    hybrid_retrieval: bool = True  # BM25 + dense with Reciprocal Rank Fusion
    hybrid_rrf_k: int = 60  # RRF rank constant; 60 is the canonical TREC default
    hybrid_fusion_method: str = "rrf"  # rrf | convex | combmnz | zscore
    hybrid_sparse_weight: float = 1.0  # BM25 run weight in hybrid fusion
    hybrid_dense_weight: float = 1.0  # dense run weight in hybrid fusion
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
    # Directory for fine-tuned model artefacts and the persistent job store
    # mounted at /v1/finetune (empty = the router default ./director-models).
    finetune_models_dir: str = ""
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
    # Source-document budget for the summarisation claim-coverage layers.
    # 0 = the WHOLE document (each backend applies its own long-input
    # chunking). The pre-WCS-1 behaviour was a 3000-char truncation, which
    # the 2026-07-15 sweep showed is dominated on both HaluEval and
    # RAGTruth (BENCHMARK_REPORT §16); set 3000 to restore it.
    nli_summarization_premise_chars: int = 0

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
        validate_and_normalize(self)

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
            handler.setFormatter(JsonLogFormatter())
            root.handlers = [handler]

    def build_store(self) -> GroundTruthStore:
        """Construct a vector-backed GroundTruthStore from config fields."""
        return _build_store(self)

    def _resolve_scorer_backend(self) -> str:
        """Resolve the 'auto' scorer backend to the best available option."""
        return _resolve_scorer_backend(self)

    def build_scorer(self, store: GroundTruthStore | None = None) -> CoherenceScorer:
        """Construct a CoherenceScorer wired to all relevant config fields."""
        return _build_scorer(self, store)

    def build_contradiction_halt(
        self,
        store: GroundTruthStore | None = None,
    ) -> ContradictionHalt | None:
        """Build the opt-in contradiction-driven streaming halt, or ``None``."""
        return _build_contradiction_halt(self, store)

    def build_correctness_feedback(self) -> RemanentiaCorrectnessClient | None:
        """Build the opt-in REMANENTIA recall-correctness client, or ``None``."""
        return _build_correctness_feedback(self)

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
                "fusion": (
                    "reciprocal_rank_fusion"
                    if self.hybrid_fusion_method == "rrf"
                    else self.hybrid_fusion_method
                ),
                "rrf_k": self.hybrid_rrf_k,
                "sparse_weight": self.hybrid_sparse_weight,
                "dense_weight": self.hybrid_dense_weight,
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
