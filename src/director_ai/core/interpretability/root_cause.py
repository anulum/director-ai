# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Mechanistic Hallucination Root-Cause Analysis
"""Turn a mechanistic attribution report into a prescriptive root-cause diagnosis.

:class:`~director_ai.core.interpretability.redeep.MechanisticAttributor` says
*where* a hallucination signal came from (which Knowledge-FFN layers and
Copying-Heads). This module says *why* and *what to do about it*: it classifies
the dominant failure mode and emits targeted fine-tuning recommendations that can
feed the Customer Model Factory — moving the guard from detective ("this is a
hallucination") to prescriptive ("the model over-trusted parametric memory at
layers 18–20; retrain to up-weight retrieved context").

The diagnosis is computed from the attribution report alone (no model access), so
it is deterministic and tenant-safe (component indices and scores only, never the
prompt or response text).
"""

from __future__ import annotations

from dataclasses import dataclass

from .redeep import MechanisticAttributionReport

PARAMETRIC_OVERRIDE = "parametric_knowledge_override"
ATTENTION_IGNORES_EVIDENCE = "attention_ignores_evidence"
UNDERACTIVE_COPYING = "underactive_copying_heads"
NO_HALLUCINATION = "no_hallucination"


@dataclass(frozen=True)
class Recommendation:
    """A targeted, prescriptive fix for one identified failure mode."""

    cause: str
    action: str
    targets: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the recommendation as a JSON-serialisable dict."""
        return {
            "cause": self.cause,
            "action": self.action,
            "targets": list(self.targets),
        }


@dataclass(frozen=True)
class RootCauseDiagnosis:
    """A prescriptive, tenant-safe diagnosis of a mechanistic hallucination."""

    is_hallucination: bool
    hallucination_risk: float
    dominant_cause: str
    causes: tuple[str, ...]
    recommendations: tuple[Recommendation, ...]
    implicated_layers: tuple[int, ...]
    implicated_heads: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, object]:
        """Tenant-safe view (component indices + scores + recommendations only)."""
        return {
            "is_hallucination": self.is_hallucination,
            "hallucination_risk": self.hallucination_risk,
            "dominant_cause": self.dominant_cause,
            "causes": list(self.causes),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "implicated_layers": list(self.implicated_layers),
            "implicated_heads": [list(h) for h in self.implicated_heads],
        }


class HallucinationRootCauseAnalyzer:
    """Classify the dominant mechanistic failure mode and prescribe a fix.

    Parameters
    ----------
    parametric_knowledge_threshold:
        A Knowledge-FFN layer with ``ffn_knowledge`` at or above this is treated
        as strongly recalling parametric memory.
    low_attention_threshold:
        A layer with ``external_attention`` at or below this is treated as
        ignoring the retrieved context.
    low_copying_threshold:
        A Copying-Head with ``copying_score`` at or below this is treated as
        failing to copy from context.
    """

    def __init__(
        self,
        *,
        parametric_knowledge_threshold: float = 0.5,
        low_attention_threshold: float = 0.3,
        low_copying_threshold: float = 0.3,
    ) -> None:
        self._ffn_t = parametric_knowledge_threshold
        self._attn_t = low_attention_threshold
        self._copy_t = low_copying_threshold

    def diagnose(self, report: MechanisticAttributionReport) -> RootCauseDiagnosis:
        """Return the prescriptive root-cause diagnosis for an attribution report."""
        if not report.is_hallucination:
            return RootCauseDiagnosis(
                is_hallucination=False,
                hallucination_risk=report.hallucination_risk,
                dominant_cause=NO_HALLUCINATION,
                causes=(),
                recommendations=(),
                implicated_layers=(),
                implicated_heads=(),
            )

        parametric_layers = tuple(
            layer.layer_index
            for layer in report.knowledge_ffn_layers
            if layer.ffn_knowledge >= self._ffn_t
            and layer.external_attention <= self._attn_t
        )
        ignoring_layers = tuple(
            layer.layer_index
            for layer in report.knowledge_ffn_layers
            if layer.external_attention <= self._attn_t
            and layer.layer_index not in parametric_layers
        )
        weak_heads = tuple(
            (head.layer_index, head.head_index)
            for head in report.copying_heads
            if head.copying_score <= self._copy_t
        )

        # Severity per cause drives the dominant classification.
        severity: dict[str, float] = {}
        recommendations: list[Recommendation] = []
        if parametric_layers:
            severity[PARAMETRIC_OVERRIDE] = max(
                layer.decoupled_risk
                for layer in report.knowledge_ffn_layers
                if layer.layer_index in parametric_layers
            )
            recommendations.append(
                Recommendation(
                    cause=PARAMETRIC_OVERRIDE,
                    action=(
                        "Fine-tune to down-weight parametric recall and up-weight "
                        "retrieved context at the implicated Knowledge-FFN layers."
                    ),
                    targets=tuple(f"ffn_layer:{i}" for i in parametric_layers),
                )
            )
        if ignoring_layers:
            severity[ATTENTION_IGNORES_EVIDENCE] = max(
                layer.decoupled_risk
                for layer in report.knowledge_ffn_layers
                if layer.layer_index in ignoring_layers
            )
            recommendations.append(
                Recommendation(
                    cause=ATTENTION_IGNORES_EVIDENCE,
                    action=(
                        "Add retrieval-grounding training examples so attention "
                        "attends to the provided evidence at these layers."
                    ),
                    targets=tuple(f"ffn_layer:{i}" for i in ignoring_layers),
                )
            )
        if weak_heads:
            severity[UNDERACTIVE_COPYING] = float(len(weak_heads))
            recommendations.append(
                Recommendation(
                    cause=UNDERACTIVE_COPYING,
                    action=(
                        "Train on copy-from-context exemplars to strengthen the "
                        "implicated Copying-Heads."
                    ),
                    targets=tuple(f"head:{layer}.{head}" for layer, head in weak_heads),
                )
            )

        causes = tuple(severity)
        dominant = (
            max(severity, key=lambda c: severity[c]) if severity else report.reason
        )
        return RootCauseDiagnosis(
            is_hallucination=True,
            hallucination_risk=report.hallucination_risk,
            dominant_cause=dominant,
            causes=causes,
            recommendations=tuple(recommendations),
            implicated_layers=tuple(sorted(set(parametric_layers + ignoring_layers))),
            implicated_heads=weak_heads,
        )
