# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Configuration Manager
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Dataclass-based configuration with env var, YAML, and profile support.

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
    server_host : str — FastAPI server bind address.
    server_port : int — FastAPI server port.
    server_workers : int — Uvicorn worker count.
    batch_max_concurrency : int — max concurrent batch requests.
    metrics_enabled : bool — enable Prometheus-style metrics collection.
    log_level : str — logging level.
    log_json : bool — structured JSON logging.
    """

    # Scoring
    coherence_threshold: float = 0.6
    hard_limit: float = 0.5
    soft_limit: float = 0.6
    use_nli: bool = False
    nli_model: str = "yaxili96/FactCG-DeBERTa-v3-Large"
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
    llm_judge_enabled: bool = False
    llm_judge_confidence_threshold: float = 0.3
    llm_judge_provider: str = ""

    # Vector store
    vector_backend: str = "memory"
    chroma_collection: str = "director_ai"
    chroma_persist_dir: str = ""
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k_multiplier: int = 3

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    server_workers: int = 1
    cors_origins: str = "*"

    # Batch
    batch_max_concurrency: int = 4

    # Observability
    metrics_enabled: bool = True
    log_level: str = "INFO"
    log_json: bool = False
    otel_enabled: bool = False

    # Audit
    audit_log_path: str = ""

    # Tenant routing
    tenant_routing: bool = False

    # Scoring weights (0.0 = use CoherenceScorer class defaults)
    w_logic: float = 0.0
    w_fact: float = 0.0

    # Rate limiting (requests per minute, 0 = disabled)
    rate_limit_rpm: int = 0

    # API key auth (empty list = no auth required)
    api_keys: list[str] = field(default_factory=list)

    # Profile name (informational)
    profile: str = "default"

    # Extra key-value overrides
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.coherence_threshold <= 1.0):
            raise ValueError(
                f"coherence_threshold must be in [0, 1], got {self.coherence_threshold}"
            )
        if not (0.0 <= self.hard_limit <= 1.0):
            raise ValueError(f"hard_limit must be in [0, 1], got {self.hard_limit}")
        if not (0.0 <= self.soft_limit <= 1.0):
            raise ValueError(f"soft_limit must be in [0, 1], got {self.soft_limit}")
        if self.soft_limit < self.hard_limit:
            raise ValueError(
                f"soft_limit ({self.soft_limit}) must be "
                f">= hard_limit ({self.hard_limit})"
            )
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {self.max_candidates}")
        if self.history_window < 1:
            raise ValueError(f"history_window must be >= 1, got {self.history_window}")
        if not (0.0 <= self.llm_temperature <= 2.0):
            raise ValueError(
                f"llm_temperature must be in [0, 2], got {self.llm_temperature}"
            )
        if self.llm_max_tokens < 1:
            raise ValueError(f"llm_max_tokens must be >= 1, got {self.llm_max_tokens}")
        if self.batch_max_concurrency < 1:
            raise ValueError(
                f"batch_max_concurrency must be >= 1, got {self.batch_max_concurrency}"
            )
        if not (1 <= self.server_port <= 65535):
            raise ValueError(
                f"server_port must be in [1, 65535], got {self.server_port}"
            )
        if self.server_workers < 1:
            raise ValueError(f"server_workers must be >= 1, got {self.server_workers}")
        if self.rate_limit_rpm < 0:
            raise ValueError(f"rate_limit_rpm must be >= 0, got {self.rate_limit_rpm}")
        if (self.w_logic != 0.0 or self.w_fact != 0.0) and abs(
            self.w_logic + self.w_fact - 1.0
        ) > 1e-6:
            raise ValueError(
                f"w_logic + w_fact must equal 1.0 when set, "
                f"got {self.w_logic} + {self.w_fact}"
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
                    kwargs[fld.name] = _coerce(value, fld.type)  # type: ignore[arg-type]
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid value for env var {key}={value!r}: {exc}"
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
            data = json.loads(raw)

        if not isinstance(data, dict):
            return cls()
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
                "profile": "thorough",
            },
            "research": {
                "use_nli": True,
                "coherence_threshold": 0.7,
                "max_candidates": 5,
                "metrics_enabled": True,
                "profile": "research",
            },
            "medical": {
                "coherence_threshold": 0.75,
                "hard_limit": 0.55,
                "soft_limit": 0.75,
                "use_nli": True,
                "reranker_enabled": True,
                "w_logic": 0.5,
                "w_fact": 0.5,
                "profile": "medical",
            },
            "finance": {
                "coherence_threshold": 0.70,
                "hard_limit": 0.50,
                "soft_limit": 0.70,
                "use_nli": True,
                "reranker_enabled": True,
                "w_logic": 0.4,
                "w_fact": 0.6,
                "profile": "finance",
            },
            "legal": {
                "coherence_threshold": 0.68,
                "hard_limit": 0.45,
                "soft_limit": 0.68,
                "use_nli": True,
                "reranker_enabled": False,
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
        }
        if name not in profiles:
            raise ValueError(
                f"Unknown profile '{name}'. Choose from: {list(profiles.keys())}"
            )
        return cls(**profiles[name])

    def configure_logging(self) -> None:
        """Apply log_level and log_json settings to the DirectorAI logger hierarchy."""
        root = logging.getLogger("DirectorAI")
        root.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))

        if self.log_json:
            handler = logging.StreamHandler()
            handler.setFormatter(_JsonFormatter())
            root.handlers = [handler]

    _REDACTED_FIELDS: frozenset[str] = frozenset({"llm_api_key", "api_keys"})

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
            f"invalid bool value: {value!r} (expected true/false/1/0/yes/no)"
        )
    if type_hint == "int":
        return int(value)
    if type_hint == "float":
        return float(value)
    if "list" in type_hint:
        return [s.strip() for s in value.split(",") if s.strip()]
    return value
