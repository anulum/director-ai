# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ReDeEP mechanistic interpretability attribution

"""Mechanistic attribution of hallucination to layers and attention heads.

Follows the ReDeEP decoupling (Sun et al., 2025, "ReDeEP: Detecting
Hallucination in RAG via Mechanistic Interpretability"): a model's output is
driven by two separable mechanisms —

* **Knowledge FFNs** — feed-forward layers that inject *parametric* knowledge
  the model learned in training; and
* **Copying Heads** — attention heads that copy from the *external* context the
  retriever supplied.

A response is hallucination-prone when it leans on parametric knowledge (high
FFN injection) while under-using the external context (low copying-head
attention). :class:`MechanisticAttributor` consumes per-layer FFN-knowledge and
external-attention signals plus per-head copying scores and reports an overall
risk together with *which* Knowledge-FFN layers and *which* Copying-Heads drove
it — the per-component explanation regulators increasingly ask for.

The signals are injected through :class:`ActivationProvider`, mirroring the
dependency-injection pattern used across the codebase (``score_fn`` /
``ContradictionScorer``). The core attribution logic is therefore pure and
testable without an ML stack; a real integration extracts the signals from a
transformer's attention maps and MLP activations (HuggingFace
``output_attentions`` / TransformerLens hooks) and feeds them in via
:meth:`MechanisticAttributor.attribute` or the :meth:`from_arrays` helpers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "ActivationProvider",
    "HeadContribution",
    "HeadSignal",
    "LayerContribution",
    "LayerSignal",
    "MechanisticAttributionReport",
    "MechanisticAttributor",
]


def _unit_interval(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]; got {value!r}")
    return float(value)


@dataclass(frozen=True)
class LayerSignal:
    """Decoupled per-layer signals for one analysed response.

    ``ffn_knowledge`` is the normalised magnitude of the layer's FFN update —
    how much *parametric* knowledge it injected. ``external_attention`` is the
    normalised attention mass the layer placed on the external context tokens —
    how much it *used the retrieved evidence*. Both in ``[0, 1]``.
    """

    layer_index: int
    ffn_knowledge: float
    external_attention: float

    def __post_init__(self) -> None:
        """Reject a negative layer index and range-check both unit signals."""
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        _unit_interval("ffn_knowledge", self.ffn_knowledge)
        _unit_interval("external_attention", self.external_attention)


@dataclass(frozen=True)
class HeadSignal:
    """Per-head copying score — attention mass on external context tokens."""

    layer_index: int
    head_index: int
    copying_score: float

    def __post_init__(self) -> None:
        """Reject negative layer/head indices and range-check ``copying_score``."""
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if self.head_index < 0:
            raise ValueError("head_index must be non-negative")
        _unit_interval("copying_score", self.copying_score)


@dataclass(frozen=True)
class LayerContribution:
    """A Knowledge-FFN layer ranked by its hallucination contribution."""

    layer_index: int
    ffn_knowledge: float
    external_attention: float
    decoupled_risk: float


@dataclass(frozen=True)
class HeadContribution:
    """A Copying-Head ranked by how much it grounded the response."""

    layer_index: int
    head_index: int
    copying_score: float


@dataclass(frozen=True)
class MechanisticAttributionReport:
    """Where a hallucination signal came from, mechanistically."""

    hallucination_risk: float
    is_hallucination: bool
    knowledge_ffn_layers: tuple[LayerContribution, ...]
    copying_heads: tuple[HeadContribution, ...]
    reason: str


@runtime_checkable
class ActivationProvider(Protocol):
    """Supplies the per-layer and per-head signals for one response.

    A real implementation extracts these from a transformer's attention maps
    and MLP activations; tests inject fixed signals.
    """

    def layer_signals(self) -> Sequence[LayerSignal]:
        """Return the per-layer FFN-knowledge and external-attention signals."""
        ...

    def head_signals(self) -> Sequence[HeadSignal]:
        """Return the per-head copying scores for the analysed response."""
        ...


class MechanisticAttributor:
    """Attribute a hallucination signal to Knowledge FFNs and Copying Heads.

    Parameters
    ----------
    ffn_weight, attention_weight :
        Relative weight of parametric reliance (FFN knowledge) versus
        external-context neglect in the per-layer risk. Normalised to sum to 1.
    risk_threshold :
        Overall risk at or above which the response is flagged a hallucination.
        Default 0.5.
    top_k :
        Number of top Knowledge-FFN layers and Copying-Heads to report.
    """

    def __init__(
        self,
        *,
        ffn_weight: float = 0.5,
        attention_weight: float = 0.5,
        risk_threshold: float = 0.5,
        top_k: int = 5,
    ) -> None:
        if ffn_weight < 0 or attention_weight < 0:
            raise ValueError("weights must be non-negative")
        total = ffn_weight + attention_weight
        if total <= 0:
            raise ValueError("ffn_weight + attention_weight must be positive")
        self._ffn_weight = ffn_weight / total
        self._attention_weight = attention_weight / total
        self._risk_threshold = _unit_interval("risk_threshold", risk_threshold)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._top_k = top_k

    def attribute(
        self,
        layer_signals: Sequence[LayerSignal],
        head_signals: Sequence[HeadSignal] = (),
    ) -> MechanisticAttributionReport:
        """Return the mechanistic attribution for one analysed response."""
        if not layer_signals:
            raise ValueError("at least one layer signal is required")
        contributions = [
            LayerContribution(
                layer_index=signal.layer_index,
                ffn_knowledge=signal.ffn_knowledge,
                external_attention=signal.external_attention,
                decoupled_risk=self._layer_risk(signal),
            )
            for signal in layer_signals
        ]
        risk = sum(c.decoupled_risk for c in contributions) / len(contributions)
        is_hallucination = risk >= self._risk_threshold
        top_layers = tuple(
            sorted(contributions, key=lambda c: c.decoupled_risk, reverse=True)[
                : self._top_k
            ]
        )
        top_heads = tuple(
            HeadContribution(
                layer_index=head.layer_index,
                head_index=head.head_index,
                copying_score=head.copying_score,
            )
            for head in sorted(
                head_signals, key=lambda h: h.copying_score, reverse=True
            )[: self._top_k]
        )
        return MechanisticAttributionReport(
            hallucination_risk=risk,
            is_hallucination=is_hallucination,
            knowledge_ffn_layers=top_layers,
            copying_heads=top_heads,
            reason=self._reason(risk, is_hallucination, top_layers),
        )

    def attribute_from(
        self, provider: ActivationProvider
    ) -> MechanisticAttributionReport:
        """Attribute using signals pulled from an :class:`ActivationProvider`."""
        return self.attribute(
            list(provider.layer_signals()), list(provider.head_signals())
        )

    def _layer_risk(self, signal: LayerSignal) -> float:
        """Per-layer risk: high parametric reliance, low external use."""
        return self._ffn_weight * signal.ffn_knowledge + self._attention_weight * (
            1.0 - signal.external_attention
        )

    def _reason(
        self,
        risk: float,
        is_hallucination: bool,
        top_layers: tuple[LayerContribution, ...],
    ) -> str:
        verdict = "hallucination" if is_hallucination else "grounded"
        # top_layers is always non-empty: attribute() rejects empty input.
        worst = top_layers[0]
        return (
            f"{verdict}: risk={risk:.3f} (threshold {self._risk_threshold:.3f}); "
            f"top Knowledge-FFN layer {worst.layer_index} "
            f"(ffn={worst.ffn_knowledge:.3f}, external={worst.external_attention:.3f})"
        )

    @staticmethod
    def layer_signals_from_arrays(
        ffn_knowledge: Sequence[float],
        external_attention: Sequence[float],
    ) -> list[LayerSignal]:
        """Build per-layer signals from two parallel per-layer arrays.

        Convenience for a real integration that has already reduced a
        transformer's MLP activations and context-attention into per-layer
        scalars.
        """
        if len(ffn_knowledge) != len(external_attention):
            raise ValueError("ffn_knowledge and external_attention length mismatch")
        return [
            LayerSignal(
                layer_index=index,
                ffn_knowledge=float(ffn),
                external_attention=float(att),
            )
            for index, (ffn, att) in enumerate(
                zip(ffn_knowledge, external_attention, strict=True)
            )
        ]

    @staticmethod
    def head_signals_from_matrix(
        copying_scores: Sequence[Sequence[float]],
    ) -> list[HeadSignal]:
        """Build per-head signals from a ``[layer][head]`` copying-score matrix."""
        signals: list[HeadSignal] = []
        for layer_index, row in enumerate(copying_scores):
            for head_index, score in enumerate(row):
                signals.append(
                    HeadSignal(
                        layer_index=layer_index,
                        head_index=head_index,
                        copying_score=float(score),
                    )
                )
        return signals
