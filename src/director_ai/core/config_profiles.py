# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — built-in configuration profile catalogue

"""Built-in configuration profiles and their operator-facing metadata.

Split out of ``config.py``: the catalogue of named presets is a distinct
responsibility from the ``DirectorConfig`` dataclass that consumes it.
``PROFILE_DEFINITIONS`` holds the field overrides ``DirectorConfig.from_profile``
applies; ``PROFILE_METADATA`` holds the validation/dependency notes
``DirectorConfig.profile_metadata`` returns. The ``ProfileMetadata`` dataclass is
re-exported from ``config`` so ``from director_ai.core.config import
ProfileMetadata`` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["PROFILE_DEFINITIONS", "PROFILE_METADATA", "ProfileMetadata"]


@dataclass(frozen=True)
class ProfileMetadata:
    """Operator-facing metadata for a built-in configuration profile."""

    name: str
    intended_workload: str
    validation_status: str
    expected_false_halt_risk: str
    required_dependencies: tuple[str, ...] = ()
    notes: str = ""
    calibration_required: bool = False
    min_calibration_samples: int = 0
    calibration_command: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize profile metadata for CLIs, APIs, and docs tooling."""
        return {
            "name": self.name,
            "intended_workload": self.intended_workload,
            "validation_status": self.validation_status,
            "expected_false_halt_risk": self.expected_false_halt_risk,
            "required_dependencies": list(self.required_dependencies),
            "notes": self.notes,
            "calibration_required": self.calibration_required,
            "min_calibration_samples": self.min_calibration_samples,
            "calibration_command": self.calibration_command,
        }


# Field overrides applied by ``DirectorConfig.from_profile``. Production secrets
# are never hard-coded here — ``from_profile`` injects api_keys / tenant map /
# HMAC keys from the environment and fails closed when they are absent.
PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
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
    "production": {
        "mode": "grounded",
        "production_mode": True,
        "use_nli": True,
        "coherence_threshold": 0.65,
        "hard_limit": 0.45,
        "soft_limit": 0.70,
        "max_candidates": 1,
        "scorer_backend": "auto",
        "coherence_require_model_backed_nli": True,
        "adaptive_threshold_enabled": True,
        "adaptive_threshold_fail_closed": True,
        "injection_detection_enabled": True,
        "injection_require_model_backed_nli": True,
        "injection_fail_closed_on_error": True,
        "sanitize_inputs": True,
        "redact_pii": True,
        "privacy_mode": True,
        "vector_backend": "chroma",
        "chroma_collection": "director_production",
        "chroma_persist_dir": "chroma",
        "hybrid_retrieval": True,
        "reranker_enabled": False,
        "tenant_routing": True,
        # No hard-coded key: production secrets are injected from the
        # environment (DIRECTOR_API_KEYS / DIRECTOR_API_KEY_TENANT_MAP).
        # Without them the production profile fails closed below.
        "llm_provider": "local",
        "llm_api_url": "http://127.0.0.1:8081/v1",
        "metrics_enabled": True,
        "metrics_require_auth": True,
        "rate_limit_rpm": 120,
        "review_queue_enabled": True,
        "review_queue_max_batch": 32,
        "review_queue_flush_timeout_ms": 10.0,
        "audit_log_path": "audit/audit.jsonl",
        "compliance_db_path": "audit/compliance.sqlite",
        "feedback_db_path": "audit/feedback.sqlite",
        "stats_backend": "sqlite",
        "stats_db_path": "audit/stats.sqlite",
        "log_json": True,
        "otel_enabled": True,
        "profile": "production",
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
        "llm_judge_provider": "local",
        "verified_scorer_enabled": True,
        # Regulated-domain profiles redact PII from logs and review queues
        # by default (KIMI-C 2026-07-15).
        "privacy_mode": True,
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
        "llm_judge_provider": "local",
        "verified_scorer_enabled": True,
        # Regulated-domain profiles redact PII from logs and review queues
        # by default (KIMI-C 2026-07-15).
        "privacy_mode": True,
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
        "llm_judge_provider": "local",
        "verified_scorer_enabled": True,
        # Regulated-domain profiles redact PII from logs and review queues
        # by default (KIMI-C 2026-07-15).
        "privacy_mode": True,
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
        "llm_judge_provider": "local",
        "verified_scorer_enabled": True,
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
    # MiniCheck precision ladder: same fact-grounding backend at three
    # latency/accuracy/memory tiers (fp16 0.4B → fp32 0.4B → 8-bit 7B).
    "minicheck-fast": {
        "use_nli": True,
        "scorer_backend": "minicheck",
        "minicheck_variant": "deberta-v3-large",
        "nli_torch_dtype": "float16",
        "coherence_threshold": 0.5,
        "max_candidates": 1,
        "metrics_enabled": True,
        "profile": "minicheck-fast",
    },
    "minicheck-balanced": {
        "use_nli": True,
        "scorer_backend": "minicheck",
        "minicheck_variant": "deberta-v3-large",
        "coherence_threshold": 0.5,
        "max_candidates": 1,
        "metrics_enabled": True,
        "profile": "minicheck-balanced",
    },
    "minicheck-accurate": {
        "use_nli": True,
        "scorer_backend": "minicheck",
        "minicheck_variant": "Bespoke-MiniCheck-7B",
        "nli_quantize_8bit": True,
        "coherence_threshold": 0.5,
        "max_candidates": 1,
        "metrics_enabled": True,
        "profile": "minicheck-accurate",
    },
}


PROFILE_METADATA: dict[str, ProfileMetadata] = {
    "fast": ProfileMetadata(
        name="fast",
        intended_workload="Development loops and high-throughput heuristic screening.",
        validation_status="smoke-tested heuristic baseline",
        expected_false_halt_risk="low for obvious policy checks, unknown for factual QA",
        required_dependencies=(),
        notes="No model loading; use for latency-first trials.",
    ),
    "lite": ProfileMetadata(
        name="lite",
        intended_workload="Offline and edge-like trials with approximate local scoring.",
        validation_status="smoke-tested lite scorer baseline",
        expected_false_halt_risk="medium without domain calibration",
        required_dependencies=(),
        notes="Uses the lite scorer backend without heavyweight NLI dependencies.",
    ),
    "rules": ProfileMetadata(
        name="rules",
        intended_workload="Deterministic local checks where model downloads are not allowed.",
        validation_status="deterministic baseline",
        expected_false_halt_risk="low for exact rules, high for semantic hallucinations",
        required_dependencies=(),
        notes="Rules-only; does not catch paraphrased contradictions reliably.",
    ),
    "embed": ProfileMetadata(
        name="embed",
        intended_workload="Semantic similarity screening when full NLI is unavailable.",
        validation_status="benchmarked approximate scorer",
        expected_false_halt_risk="medium; tune thresholds for each corpus",
        required_dependencies=("embed",),
        notes="Requires an embedding scorer dependency path.",
    ),
    "thorough": ProfileMetadata(
        name="thorough",
        intended_workload="General production baseline with NLI and local judge escalation.",
        validation_status="standard validated baseline",
        expected_false_halt_risk="medium until tuned on customer data",
        required_dependencies=("nli",),
        notes="Use `director-ai tune` before strict production enforcement.",
    ),
    "research": ProfileMetadata(
        name="research",
        intended_workload="Academic and analytical workloads that prefer precision.",
        validation_status="experimental high-threshold baseline",
        expected_false_halt_risk="high by design",
        required_dependencies=("nli",),
        notes="Higher threshold intentionally rejects more borderline responses.",
    ),
    "production": ProfileMetadata(
        name="production",
        intended_workload="Authenticated multi-tenant service deployment with audit, metrics, and fail-closed scoring.",
        validation_status="operational scaffold; requires customer calibration and external security test before public exposure",
        expected_false_halt_risk="medium until tuned on tenant-specific clean and adversarial traces",
        required_dependencies=("server", "nli", "vector", "otel"),
        notes=(
            "The built-in key is for local validation only; generated Compose "
            "requires operator-provided secrets and upstream URLs."
        ),
        calibration_required=True,
        min_calibration_samples=100,
        calibration_command="director-ai tune --profile production --input labelled_traces.jsonl",
    ),
    "medical": ProfileMetadata(
        name="medical",
        intended_workload="Clinical or biomedical fact-heavy review with curated KB.",
        validation_status="PubMedQA artifact shows FPR=1.0 at t=0.30; calibration required",
        expected_false_halt_risk="very high without KB grounding and calibration",
        required_dependencies=("nli", "vector"),
        notes="Do not deploy strictly until tuned on local clean and adversarial samples.",
        calibration_required=True,
        min_calibration_samples=50,
        calibration_command="director-ai tune --profile medical --input labelled_traces.jsonl",
    ),
    "finance": ProfileMetadata(
        name="finance",
        intended_workload="Financial claims, numeric facts, and regulatory KB review.",
        validation_status="FinanceBench artifact shows FPR=1.0 at t=0.30; calibration required",
        expected_false_halt_risk="very high without KB grounding and calibration",
        required_dependencies=("nli", "vector"),
        notes="Use with retrieval and domain-specific clean-response calibration.",
        calibration_required=True,
        min_calibration_samples=50,
        calibration_command="director-ai tune --profile finance --input labelled_traces.jsonl",
    ),
    "legal": ProfileMetadata(
        name="legal",
        intended_workload="Legal reasoning chains over small curated KBs.",
        validation_status="CUAD artifact shows FPR=1.0 at t=0.30; calibration required",
        expected_false_halt_risk="unknown; treat as high until tuned",
        required_dependencies=("nli",),
        notes="Thresholds are aligned with other high-stakes profiles pending eval.",
        calibration_required=True,
        min_calibration_samples=50,
        calibration_command="director-ai tune --profile legal --input labelled_traces.jsonl",
    ),
    "creative": ProfileMetadata(
        name="creative",
        intended_workload="Drafting, fiction, style exploration, and non-factual generation.",
        validation_status="heuristic permissive preset",
        expected_false_halt_risk="low for creative drift, high for factual safety",
        required_dependencies=(),
        notes="NLI is disabled to avoid penalising metaphor and divergent writing.",
    ),
    "customer_support": ProfileMetadata(
        name="customer_support",
        intended_workload="Policy support bots and troubleshooting assistants.",
        validation_status="latency-first starter preset",
        expected_false_halt_risk="medium; depends on policy KB coverage",
        required_dependencies=(),
        notes="Add vector retrieval when support policy facts are available.",
    ),
    "summarization": ProfileMetadata(
        name="summarization",
        intended_workload="Source-grounded summaries and extractive synthesis.",
        validation_status="validated with summarization FPR diagnostics",
        expected_false_halt_risk="low after v3.6 claim coverage, still tune per corpus",
        required_dependencies=("nli",),
        notes="Uses prompt-as-premise scoring, trimmed mean aggregation, and claim coverage.",
        calibration_required=True,
        min_calibration_samples=20,
        calibration_command=(
            "director-ai tune --profile summarization --input labelled_traces.jsonl"
        ),
    ),
    "minicheck-fast": ProfileMetadata(
        name="minicheck-fast",
        intended_workload="High-throughput fact grounding with the 0.4B MiniCheck "
        "DeBERTa checkpoint in float16.",
        validation_status="MiniCheck fact-grounding backend, fp16 latency tier",
        expected_false_halt_risk="medium; tune thresholds per corpus",
        required_dependencies=("nli", "minicheck"),
        notes="Lowest latency and memory; pip install minicheck.",
    ),
    "minicheck-balanced": ProfileMetadata(
        name="minicheck-balanced",
        intended_workload="Default-precision fact grounding with the 0.4B MiniCheck "
        "DeBERTa checkpoint.",
        validation_status="MiniCheck fact-grounding backend, full-precision tier",
        expected_false_halt_risk="medium; tune thresholds per corpus",
        required_dependencies=("nli", "minicheck"),
        notes="Balances latency and accuracy; pip install minicheck.",
    ),
    "minicheck-accurate": ProfileMetadata(
        name="minicheck-accurate",
        intended_workload="Highest-accuracy fact grounding with the 7B "
        "Bespoke-MiniCheck checkpoint in 8-bit.",
        validation_status="MiniCheck 7B backend, 8-bit accuracy tier",
        expected_false_halt_risk="lower than smaller variants; still tune per corpus",
        required_dependencies=("nli", "minicheck", "bitsandbytes"),
        notes="Needs a GPU with ~8 GB; pip install minicheck bitsandbytes.",
    ),
}
