# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — grouped value-configuration for CoherenceScorer

"""A grouped, validated value-configuration for :class:`CoherenceScorer`.

``CoherenceScorer.__init__`` carries ~35 keyword arguments — thresholds,
weights, NLI backend settings, LLM-judge and reasoning options. Passing them
loose is error-prone and hard to reuse. :class:`ScorerConfig` collects the
*value* settings (not the injected runtime dependencies — the ground-truth
store and cache stay separate) into one immutable, self-validating object that
``CoherenceScorer.from_config`` unpacks.

Field names mirror the scorer's parameters exactly, so the config round-trips
through ``from_config`` without any mapping layer; the existing per-argument
constructor is untouched, so this is purely additive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ScorerConfig"]


@dataclass(frozen=True)
class ScorerConfig:
    """Immutable value-configuration for :class:`CoherenceScorer`.

    Holds only configuration *values*; runtime dependencies (the ground-truth
    store and the cache instance) are passed to :meth:`CoherenceScorer.from_config`
    separately so the config stays serialisable and reusable across scorers.
    """

    # Core thresholds / weights
    threshold: float = 0.5
    history_window: int = 5
    soft_limit: float | None = None
    w_logic: float | None = None
    w_fact: float | None = None
    strict_mode: bool = False

    # NLI backend
    use_nli: bool | None = None
    nli_model: str | None = None
    require_model_backed_nli: bool = False
    nli_quantize_8bit: bool = False
    nli_device: str | None = None
    nli_torch_dtype: str | None = None
    nli_devices: tuple[str, ...] | None = None
    nli_max_length: int = 512
    nli_revision: str | None = None
    scorer_backend: str = "deberta"
    onnx_path: str | None = None
    onnx_batch_size: int = 16
    onnx_flush_timeout_ms: float = 10.0
    minicheck_variant: str = "deberta-v3-large"

    # Cache
    cache_size: int = 0
    cache_ttl: float = 300.0

    # LLM-judge (hybrid backend)
    llm_judge_enabled: bool = False
    llm_judge_confidence_threshold: float = 0.3
    llm_judge_provider: str = ""
    llm_judge_model: str = ""
    llm_judge_model_revision: str | None = None

    # Reasoning escalation
    reasoning_enabled: bool = False
    reasoning_provider: str = ""
    reasoning_model: str = ""
    reasoning_model_revision: str | None = None
    reasoning_escalation_margin: float = 0.15

    privacy_mode: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.soft_limit is not None and not (0.0 <= self.soft_limit <= 1.0):
            raise ValueError(f"soft_limit must be in [0, 1], got {self.soft_limit}")
        if self.soft_limit is not None and self.soft_limit < self.threshold:
            raise ValueError(
                f"soft_limit ({self.soft_limit}) must be >= threshold "
                f"({self.threshold})",
            )
        for name in ("w_logic", "w_fact"):
            v = getattr(self, name)
            if v is not None and not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.history_window < 1:
            raise ValueError(f"history_window must be >= 1, got {self.history_window}")

    def to_kwargs(self) -> dict[str, Any]:
        """Return the config as scorer constructor keyword arguments."""
        from dataclasses import asdict

        return asdict(self)
