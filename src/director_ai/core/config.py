# SPDX-License-Identifier: AGPL-3.0-or-later
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

__all__ = ["DirectorConfig"]

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
    llm_provider : str — LLM backend: mock, openai, anthropic,
        huggingface, or local.
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

    # Scoring
    coherence_threshold: float = 0.6
    hard_limit: float = 0.5
    soft_limit: float = 0.6
    use_nli: bool = False
    nli_model: str = "yaxili96/FactCG-DeBERTa-v3-Large"
    nli_model_revision: str = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
    nli_max_length: int = 512  # >512 for long-context models (Longformer, BigBird)
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
    # configured external LLM provider (OpenAI / Anthropic) for scoring.
    # Do not enable in privacy-sensitive deployments without user consent.
    llm_judge_enabled: bool = False
    llm_judge_confidence_threshold: float = 0.3
    llm_judge_provider: str = ""  # "openai", "anthropic", or "local"
    llm_judge_model: str = ""
    llm_judge_local_model: str = ""  # path to local judge checkpoint
    privacy_mode: bool = False

    # Scorer backend: "deberta", "onnx", "minicheck", "hybrid", "lite", "rust"
    # "auto" picks best available: rust > onnx > deberta > nli-lite > lite
    scorer_backend: str = "auto"

    # Multi-GPU NLI sharding (comma-separated, e.g. "cuda:0,cuda:1")
    nli_devices: str = ""

    # Vector store
    vector_backend: str = "memory"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_model_revision: str = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
    chroma_collection: str = "director_ai"
    chroma_persist_dir: str = ""
    hybrid_retrieval: bool = True  # BM25 + dense with Reciprocal Rank Fusion
    reranker_enabled: bool = True  # cross-encoder reranking on top of retrieval
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_model_revision: str = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    reranker_top_k_multiplier: int = 3
    retrieval_abstention_threshold: float = 0.3  # 0 = disabled; min similarity to score

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
    server_host: str = "0.0.0.0"  # nosec B104 — override via HOST env var; production behind reverse proxy
    server_port: int = 8080
    server_workers: int = 1
    cors_origins: str = ""

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

    # Tenant routing
    tenant_routing: bool = False

    # Rate Limiting
    rate_limit_enabled: bool = False

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

    # Scoring weights (0.0 = use CoherenceScorer class defaults)
    w_logic: float = 0.0
    w_fact: float = 0.0

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

    # Stats backend: "prometheus" (default, in-memory) or "sqlite" (persistent)
    stats_backend: str = "prometheus"
    stats_db_path: str = "~/.director-ai/stats.db"

    # AGPL Â§13 source code endpoint
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

        # Hardened mode: enforce all safety features
        if self.hardened:
            object.__setattr__(self, "production_mode", True)
            object.__setattr__(self, "use_nli", True)
            object.__setattr__(self, "injection_detection_enabled", True)
            object.__setattr__(self, "sanitize_inputs", True)
            object.__setattr__(self, "redact_pii", True)
            object.__setattr__(self, "strict_mode", True)
            logger.info("Hardened mode: all safety features enforced")

        # Production mode enforcements
        if self.production_mode:
            if not self.api_keys and not self.api_key_tenant_map:
                raise ValueError(
                    "production_mode requires api_keys or api_key_tenant_map"
                )
            if self.server_host == "0.0.0.0":  # noqa: S104
                logger.warning(
                    "production_mode: binding to 0.0.0.0 — ensure reverse proxy with TLS"
                )
        if (
            self.vector_backend == "sentence-transformer"
            and not self.embedding_model.strip()
        ):
            raise ValueError(
                "embedding_model must be set when vector_backend='sentence-transformer'",
            )

    @classmethod
    def from_env(cls, prefix: str = "DIRECTOR_") -> DirectorConfig:
        """Load configuration from environment variables.

        Reads ``DIRECTOR_<FIELD>`` env vars (case-insensitive field matching).
        Example: ``DIRECTOR_COHERENCE_THRESHOLD=0.7``
        """
        kwargs: dict = {}
        field_map = {f.name.upper(): f for f in cls.__dataclass_fields__.values()}

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            field_name = key[len(prefix) :]
            if field_name in field_map:
                fld = field_map[field_name]
                try:
                    kwargs[fld.name] = _coerce(
                        value,
                        fld.type,  # type: ignore[arg-type]
                    )
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
            import yaml  # type: ignore[import-untyped]

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
        profiles: dict[str, dict] = {
            "fast": {
                "use_nli": False,
                "coherence_threshold": 0.5,
                "max_candidates": 1,
                "metrics_enabled": False,
                "profile": "fast",
            },
            "thorough": {
                "use_nli": True,
                "coherence_threshold": 0.6,
                "max_candidates": 3,
                "metrics_enabled": True,
                "scorer_backend": "hybrid",
                "llm_judge_enabled": True,
                "llm_judge_provider": "local",
                "profile": "thorough",
            },
            "research": {
                "use_nli": True,
                "coherence_threshold": 0.7,
                "max_candidates": 5,
                "metrics_enabled": True,
                "scorer_backend": "hybrid",
                "llm_judge_enabled": True,
                "llm_judge_provider": "local",
                "profile": "research",
            },
            # PubMedQA (1000 samples, NLI, GTX 1060, 2026-03-20):
            #   F1=61.9% at t=0.30, BUT FPR=100% (all responses flagged).
            #   Precision=44.8%. Needs KB grounding or calibration to be usable.
            "medical": {
                "coherence_threshold": 0.30,
                "hard_limit": 0.20,
                "soft_limit": 0.35,
                "use_nli": True,
                "reranker_enabled": True,
                "scorer_backend": "hybrid",
                "llm_judge_enabled": True,
                "w_logic": 0.5,
                "w_fact": 0.5,
                "profile": "medical",
            },
            # FinanceBench (150 samples, NLI, GTX 1060, 2026-03-20):
            #   All 150 clean samples flagged (FPR=100%, precision=0%).
            #   Threshold not validated — needs KB grounding or recalibration.
            "finance": {
                "coherence_threshold": 0.30,
                "hard_limit": 0.20,
                "soft_limit": 0.35,
                "use_nli": True,
                "reranker_enabled": True,
                "scorer_backend": "hybrid",
                "llm_judge_enabled": True,
                "w_logic": 0.4,
                "w_fact": 0.6,
                "profile": "finance",
            },
            # Not yet measured (CUAD OOM on 6GB VRAM). Threshold aligned
            # with medical/finance pending domain-specific validation.
            "legal": {
                "coherence_threshold": 0.30,
                "hard_limit": 0.20,
                "soft_limit": 0.35,
                "use_nli": True,
                "reranker_enabled": False,
                "scorer_backend": "hybrid",
                "llm_judge_enabled": True,
                "w_logic": 0.6,
                "w_fact": 0.4,
                "profile": "legal",
            },
            "creative": {
                "coherence_threshold": 0.40,
                "hard_limit": 0.30,
                "soft_limit": 0.45,
                "use_nli": False,
                "reranker_enabled": False,
                "w_logic": 0.7,
                "w_fact": 0.3,
                "profile": "creative",
            },
            "customer_support": {
                "coherence_threshold": 0.55,
                "hard_limit": 0.40,
                "soft_limit": 0.60,
                "use_nli": False,
                "reranker_enabled": False,
                "w_logic": 0.5,
                "w_fact": 0.5,
                "profile": "customer_support",
            },
            "summarization": {
                "coherence_threshold": 0.15,
                "hard_limit": 0.08,
                "soft_limit": 0.25,
                "use_nli": True,
                "reranker_enabled": False,
                "scorer_backend": "hybrid",
                "llm_judge_enabled": True,
                "w_logic": 0.0,
                "w_fact": 1.0,
                "nli_fact_inner_agg": "min",
                "nli_fact_outer_agg": "trimmed_mean",
                "nli_logic_inner_agg": "min",
                "nli_logic_outer_agg": "mean",
                "nli_premise_ratio": 0.85,
                "nli_fact_retrieval_top_k": 8,
                "nli_use_prompt_as_premise": True,
                "nli_summarization_baseline": 0.20,
                "nli_claim_coverage_enabled": True,
                "nli_claim_support_threshold": 0.6,
                "nli_claim_coverage_alpha": 0.4,
                "profile": "summarization",
            },
            "lite": {
                "use_nli": False,
                "scorer_backend": "lite",
                "coherence_threshold": 0.5,
                "max_candidates": 1,
                "metrics_enabled": False,
                "profile": "lite",
            },
            "rules": {
                "use_nli": False,
                "scorer_backend": "rules",
                "coherence_threshold": 0.5,
                "max_candidates": 1,
                "metrics_enabled": False,
                "profile": "rules",
            },
            "embed": {
                "use_nli": False,
                "scorer_backend": "embed",
                "coherence_threshold": 0.6,
                "max_candidates": 2,
                "metrics_enabled": False,
                "profile": "embed",
            },
        }
        if name not in profiles:
            raise ValueError(
                f"Unknown profile '{name}'. Choose from: {list(profiles.keys())}",
            )
        cfg = cls(**profiles[name])
        for key, value in profiles[name].items():
            object.__setattr__(cfg, key, value)
        return cfg

    def configure_logging(self) -> None:
        """Apply log_level and log_json settings to the DirectorAI logger hierarchy."""
        root = logging.getLogger("DirectorAI")
        root.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))

        if self.log_json:
            handler = logging.StreamHandler()
            handler.setFormatter(_JsonFormatter())
            root.handlers = [handler]

    def build_store(self):
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

        if self.hybrid_retrieval:
            try:
                from .retrieval.vector_store import HybridBackend

                backend = HybridBackend(base=backend)
                logger.info("Hybrid retrieval enabled (BM25 + dense + RRF)")
            except ImportError:
                logger.warning("HybridBackend unavailable, using dense-only retrieval")

        if self.reranker_enabled:
            try:
                from .retrieval.vector_store import RerankedBackend

                backend = RerankedBackend(
                    base=backend,
                    reranker_model=self.reranker_model,
                    top_k_multiplier=self.reranker_top_k_multiplier,
                )
                logger.info("Cross-encoder reranker enabled: %s", self.reranker_model)
            except ImportError:
                logger.warning("sentence-transformers not installed, skipping reranker")

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

            kw_hyde: dict = {}
            if self.hyde_prompt_template:
                kw_hyde["template"] = self.hyde_prompt_template
            # Generator will be injected by build_scorer() or by the user
            backend = HyDEBackend(base=backend, **kw_hyde)
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

        return VectorGroundTruthStore(backend=backend)

    def _resolve_scorer_backend(self) -> str:
        """Resolve 'auto' scorer backend to best available."""
        if self.scorer_backend != "auto":
            return self.scorer_backend

        # Priority: rust > onnx > deberta > nli-lite > lite
        try:
            import backfire_kernel  # noqa: F401

            logger.info("Auto scorer: selected 'rust' (backfire_kernel available)")
            return "rust"
        except ImportError:
            pass

        if self.onnx_path:
            logger.info("Auto scorer: selected 'onnx' (onnx_path configured)")
            return "onnx"

        if self.use_nli:
            logger.info("Auto scorer: selected 'deberta' (NLI enabled)")
            return "deberta"

        logger.info("Auto scorer: selected 'lite' (no NLI available)")
        return "lite"

    def build_scorer(self, store=None):
        """Construct a CoherenceScorer wired to all relevant config fields."""
        from .scoring.scorer import CoherenceScorer

        if store is None:
            store = self.build_store()

        judge_model = self.llm_judge_model
        if self.llm_judge_provider == "local" and self.llm_judge_local_model:
            judge_model = self.llm_judge_local_model

        resolved_backend = self._resolve_scorer_backend()

        kw: dict = {
            "threshold": self.coherence_threshold,
            "use_nli": self.use_nli,
            "scorer_backend": resolved_backend,
            "soft_limit": self.soft_limit,
            "nli_model": self.nli_model,
            "nli_revision": self.nli_model_revision or None,
            "nli_max_length": self.nli_max_length,
            "llm_judge_enabled": self.llm_judge_enabled,
            "llm_judge_confidence_threshold": self.llm_judge_confidence_threshold,
            "llm_judge_provider": self.llm_judge_provider,
            "llm_judge_model": judge_model,
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
        scorer._adaptive_threshold_enabled = self.adaptive_threshold_enabled
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
        if self.dry_run:
            scorer._dry_run = True
            logger.info("Dry-run mode: scoring but never rejecting")
        return scorer

    _REDACTED_FIELDS: frozenset[str] = frozenset(
        {
            "llm_api_key",
            "api_keys",
            "api_key_tenant_map",
            "audit_postgres_url",
            "redis_url",
        },
    )

    def to_dict(self) -> dict:
        """Serialize to a plain dict (safe for JSON/API responses)."""
        d = {}
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
    return value
