# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Token cost estimation and analysis
"""Token cost estimation for LLM guardrail operations.

Estimates per-model and per-agent cost based on token counts and
provider pricing. Useful for compliance reports (EU AI Act Article 15),
budget planning, and swarm cost attribution.

Usage::

    from director_ai.compliance.cost_analyser import CostAnalyser

    analyser = CostAnalyser()
    analyser.add_pricing("gpt-4o", input_per_1k=0.0025, output_per_1k=0.01)
    analyser.record("gpt-4o", input_tokens=500, output_tokens=150)
    analyser.record("gpt-4o", input_tokens=800, output_tokens=200)

    report = analyser.report()
    print(report["total_cost_chf"])  # estimated CHF cost
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

__all__ = ["CostAnalyser", "ModelPricing", "CostRecord"]


@dataclass(frozen=True)
class ModelPricing:
    """Per-model token pricing in CHF per 1K tokens."""

    model: str
    input_per_1k: float  # CHF per 1K input tokens
    output_per_1k: float  # CHF per 1K output tokens


@dataclass
class CostRecord:
    """Accumulated cost data for a single model."""

    model: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


# Default pricing (CHF, approximate as of 2026-04)
_DEFAULT_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing("gpt-4o", 0.0025, 0.01),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.00015, 0.0006),
    "claude-3-5-sonnet": ModelPricing("claude-3-5-sonnet", 0.003, 0.015),
    "claude-3-5-haiku": ModelPricing("claude-3-5-haiku", 0.0008, 0.004),
    "gemini-2.0-flash": ModelPricing("gemini-2.0-flash", 0.0001, 0.0004),
}


class CostAnalyser:
    """Token cost estimation and attribution.

    Thread-safe accumulator for per-model token costs.

    Parameters
    ----------
    currency : str
        Currency label (default ``"CHF"``).
    """

    def __init__(self, currency: str = "CHF") -> None:
        self._currency = currency
        self._pricing: dict[str, ModelPricing] = dict(_DEFAULT_PRICING)
        self._records: dict[str, CostRecord] = {}
        self._lock = threading.Lock()

    def add_pricing(
        self,
        model: str,
        input_per_1k: float,
        output_per_1k: float,
    ) -> None:
        """Add or update pricing for a model."""
        self._pricing[model] = ModelPricing(model, input_per_1k, output_per_1k)

    def record(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        agent_id: str = "",
    ) -> None:
        """Record a token usage event.

        Parameters
        ----------
        model : str
            Model identifier (e.g., ``"gpt-4o"``).
        input_tokens : int
            Number of input/prompt tokens.
        output_tokens : int
            Number of output/completion tokens.
        agent_id : str
            Optional agent ID for per-agent attribution.
        """
        key = f"{model}::{agent_id}" if agent_id else model
        with self._lock:
            rec = self._records.setdefault(key, CostRecord(model=model))
            rec.total_input_tokens += input_tokens
            rec.total_output_tokens += output_tokens
            rec.call_count += 1

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a single call (CHF)."""
        pricing = self._pricing.get(model)
        if pricing is None:
            return 0.0
        return (
            input_tokens / 1000 * pricing.input_per_1k
            + output_tokens / 1000 * pricing.output_per_1k
        )

    def report(self) -> dict:
        """Generate a cost report.

        Returns
        -------
        dict with keys:
            ``"currency"``: currency label
            ``"total_cost"``: total estimated cost
            ``"total_tokens"``: total tokens across all models
            ``"models"``: per-model breakdown
        """
        with self._lock:
            models: dict[str, dict] = {}
            total_cost = 0.0
            total_tokens = 0

            for key, rec in self._records.items():
                cost = self.estimate_cost(
                    rec.model, rec.total_input_tokens, rec.total_output_tokens
                )
                total_cost += cost
                total_tokens += rec.total_tokens

                models[key] = {
                    "model": rec.model,
                    "input_tokens": rec.total_input_tokens,
                    "output_tokens": rec.total_output_tokens,
                    "total_tokens": rec.total_tokens,
                    "call_count": rec.call_count,
                    "estimated_cost": round(cost, 6),
                }

        return {
            "currency": self._currency,
            "total_cost": round(total_cost, 6),
            "total_tokens": total_tokens,
            "models": models,
        }

    def reset(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()
