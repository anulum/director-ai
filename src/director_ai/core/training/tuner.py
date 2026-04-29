# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Threshold Tuner

from __future__ import annotations

from dataclasses import dataclass

from ..scoring.scorer import CoherenceScorer

__all__ = ["TuneResult", "format_profile_overlay", "tune"]

DEFAULT_THRESHOLDS = [round(0.30 + i * 0.05, 2) for i in range(13)]  # 0.30..0.90
DEFAULT_WEIGHT_PAIRS = [(0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]


@dataclass
class TuneResult:
    threshold: float
    w_logic: float
    w_fact: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    samples: int

    def to_profile_overlay(
        self,
        *,
        profile: str = "tuned",
        base_profile: str = "",
    ) -> dict[str, object]:
        """Return a ready-to-use config overlay for this tuning result."""
        hard_limit = max(0.0, round(self.threshold - 0.10, 4))
        soft_limit = min(1.0, round(self.threshold + 0.10, 4))
        extra = {
            "tune_samples": str(self.samples),
            "tune_balanced_accuracy": f"{self.balanced_accuracy:.4f}",
            "tune_precision": f"{self.precision:.4f}",
            "tune_recall": f"{self.recall:.4f}",
            "tune_f1": f"{self.f1:.4f}",
        }
        if base_profile:
            extra["tuned_from_profile"] = base_profile
        return {
            "profile": profile,
            "coherence_threshold": self.threshold,
            "hard_limit": hard_limit,
            "soft_limit": soft_limit,
            "w_logic": self.w_logic,
            "w_fact": self.w_fact,
            "extra": extra,
        }


def format_profile_overlay(
    result: TuneResult,
    *,
    profile: str = "tuned",
    base_profile: str = "",
) -> str:
    """Render a deterministic YAML profile overlay for CLI output."""
    overlay = result.to_profile_overlay(profile=profile, base_profile=base_profile)
    extra = overlay.pop("extra")
    lines = [
        "# Director-AI tuned profile overlay",
        "# Load with DirectorConfig.from_yaml(...) or merge over a base profile.",
    ]
    if base_profile:
        lines.append(f"# Base profile: {base_profile}")
    lines.extend(
        [
            f"# Samples: {result.samples}",
            f"# Balanced accuracy: {result.balanced_accuracy:.4f}",
            f"# Precision: {result.precision:.4f}",
            f"# Recall: {result.recall:.4f}",
            f"# F1: {result.f1:.4f}",
        ],
    )
    for key, value in overlay.items():
        lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("extra:")
    assert isinstance(extra, dict)
    for key, value in extra.items():
        lines.append(f"  {key}: {_yaml_scalar(value)}")
    lines.append("")
    return "\n".join(lines)


def tune(
    samples: list[dict],
    thresholds: list[float] | None = None,
    weight_pairs: list[tuple[float, float]] | None = None,
    use_nli: bool = False,
    ground_truth_store=None,
) -> TuneResult:
    """Grid-search over thresholds and weight pairs, maximize balanced accuracy.

    Each sample: ``{"prompt": str, "response": str, "label": bool}``
    where ``label=True`` means the response is correct (should be approved).
    """
    if not samples:
        raise ValueError("samples must be non-empty")

    thresholds = thresholds or DEFAULT_THRESHOLDS
    weight_pairs = weight_pairs or DEFAULT_WEIGHT_PAIRS

    best: TuneResult | None = None

    for wl, wf in weight_pairs:
        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=use_nli,
            ground_truth_store=ground_truth_store,
            w_logic=wl,
            w_fact=wf,
        )
        scores = []
        for s in samples:
            _, cs = scorer.review(s["prompt"], s["response"])
            scores.append((cs.score, s["label"]))

        for thr in thresholds:
            tp = fp = tn = fn = 0
            for score_val, label in scores:
                predicted = score_val >= thr
                if label and predicted:
                    tp += 1
                elif label and not predicted:
                    fn += 1
                elif not label and predicted:
                    fp += 1
                else:
                    tn += 1

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ba = (tpr + tnr) / 2.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tpr
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            result = TuneResult(
                threshold=thr,
                w_logic=wl,
                w_fact=wf,
                balanced_accuracy=round(ba, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                samples=len(samples),
            )
            if best is None or ba > best.balanced_accuracy:
                best = result

    if best is None:
        raise ValueError("No valid tuning result — check that samples is non-empty")
    return best


def _yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
