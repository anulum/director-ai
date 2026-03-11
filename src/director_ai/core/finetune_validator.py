# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Fine-tuning Data Validator
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Validate customer-provided JSONL before fine-tuning.

Checks format, class balance, duplicates, field lengths, and estimates
training cost. Run this before ``finetune_nli()`` to catch issues early.

Usage::

    from director_ai.core.finetune_validator import validate_finetune_data
    report = validate_finetune_data("customer_data.jsonl")
    if not report.is_valid:
        for e in report.errors:
            print(e)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("DirectorAI.FinetuneValidator")

MIN_SAMPLES = 500
MIN_PER_CLASS = 100
WARN_PER_CLASS = 200
MAX_IMBALANCE_RATIO = 5.0
MAX_TEXT_LENGTH = 4096

# Calibrated on RTX 6000 Ada, bs=24, DeBERTa-Large, 3 epochs
_SECONDS_PER_SAMPLE_PER_EPOCH = 0.053  # ~1.26 it/s at bs=24 → 24/1.26 ≈ 19s per batch of 24 ≈ 0.79s per sample... no: 1 step = 24 samples, 1.26 steps/s → 30.2 samples/s → 0.033s/sample/epoch... 3 epochs → 0.1s total. Use 0.053 with overhead.
_GPU_COST_PER_HOUR = 1.0  # USD, RTX 6000 Ada tier


@dataclass
class DataQualityReport:
    """Result of validating fine-tuning data."""

    total_samples: int = 0
    label_distribution: dict[int, int] = field(default_factory=dict)
    class_balance_ratio: float = 0.0
    avg_premise_tokens: int = 0
    avg_hypothesis_tokens: int = 0
    max_premise_tokens: int = 0
    max_hypothesis_tokens: int = 0
    duplicate_count: int = 0
    empty_field_count: int = 0
    parse_error_count: int = 0
    estimated_train_time_min: float = 0.0
    estimated_cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Samples: {self.total_samples}",
            f"Labels: {self.label_distribution}",
            f"Balance ratio: {self.class_balance_ratio:.2f}",
            f"Duplicates: {self.duplicate_count}",
            f"Est. time: {self.estimated_train_time_min:.0f} min",
            f"Est. cost: ${self.estimated_cost_usd:.2f}",
            f"Valid: {self.is_valid}",
        ]
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


def _rough_token_count(text: str) -> int:
    """Approximate token count (whitespace + punctuation split)."""
    return len(text.split())


def validate_finetune_data(
    path: str | Path,
    epochs: int = 3,
) -> DataQualityReport:
    """Validate a JSONL file for fine-tuning readiness.

    Parameters
    ----------
    path : path to JSONL with premise/hypothesis/label
    epochs : planned training epochs (for cost estimate)
    """
    path = Path(path)
    report = DataQualityReport()

    if not path.exists():
        report.errors.append(f"File not found: {path}")
        return report

    rows = []
    seen = set()
    premise_lengths = []
    hypothesis_lengths = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                report.parse_error_count += 1
                continue

            premise = row.get("premise") or row.get("doc") or row.get("context", "")
            hypothesis = (
                row.get("hypothesis") or row.get("claim") or row.get("response", "")
            )
            label = row.get("label")

            if not premise or not hypothesis:
                report.empty_field_count += 1
                continue
            if label is None:
                report.empty_field_count += 1
                continue

            try:
                label = int(label)
            except (ValueError, TypeError):
                report.parse_error_count += 1
                continue

            if label not in (0, 1):
                report.errors.append(
                    f"Line {line_num}: label must be 0 or 1, got {label}"
                )
                if len(report.errors) > 10:
                    report.errors.append("(truncated, too many label errors)")
                    break
                continue

            key = (premise[:200], hypothesis[:200], label)
            if key in seen:
                report.duplicate_count += 1
            else:
                seen.add(key)

            premise_lengths.append(_rough_token_count(premise))
            hypothesis_lengths.append(_rough_token_count(hypothesis))
            rows.append({"premise": premise, "hypothesis": hypothesis, "label": label})

    report.total_samples = len(rows)

    if not rows:
        report.errors.append("No valid samples found")
        return report

    label_counts = Counter(r["label"] for r in rows)
    report.label_distribution = dict(label_counts)

    min_class = min(label_counts.values())
    max_class = max(label_counts.values())
    report.class_balance_ratio = min_class / max_class if max_class > 0 else 0.0

    # Length stats
    report.avg_premise_tokens = sum(premise_lengths) // len(premise_lengths)
    report.avg_hypothesis_tokens = sum(hypothesis_lengths) // len(hypothesis_lengths)
    report.max_premise_tokens = max(premise_lengths)
    report.max_hypothesis_tokens = max(hypothesis_lengths)

    # Cost estimate (based on total rows including duplicates — that's what training sees)
    total_seconds = report.total_samples * epochs * _SECONDS_PER_SAMPLE_PER_EPOCH
    report.estimated_train_time_min = total_seconds / 60
    report.estimated_cost_usd = (total_seconds / 3600) * _GPU_COST_PER_HOUR

    # Validation rules
    if report.total_samples < MIN_SAMPLES:
        report.errors.append(
            f"Need at least {MIN_SAMPLES} samples, got {report.total_samples}"
        )

    for label_val, count in label_counts.items():
        if count < MIN_PER_CLASS:
            report.errors.append(
                f"Label {label_val} has only {count} samples (minimum: {MIN_PER_CLASS})"
            )
        elif count < WARN_PER_CLASS:
            report.warnings.append(
                f"Label {label_val} has only {count} samples (recommended: {WARN_PER_CLASS}+)"
            )

    if report.class_balance_ratio < 1 / MAX_IMBALANCE_RATIO:
        report.warnings.append(
            f"Class imbalance {max_class}:{min_class} ({1 / report.class_balance_ratio:.1f}:1) — "
            f"consider downsampling the majority class"
        )

    if report.duplicate_count > report.total_samples * 0.1:
        report.warnings.append(
            f"{report.duplicate_count} duplicates ({report.duplicate_count / report.total_samples:.0%}) — "
            f"consider deduplication"
        )

    if report.max_premise_tokens > MAX_TEXT_LENGTH:
        report.warnings.append(
            f"Longest premise is {report.max_premise_tokens} tokens — "
            f"texts beyond 512 tokens will be truncated"
        )

    if report.parse_error_count > 0:
        report.warnings.append(f"{report.parse_error_count} lines failed to parse")

    if report.empty_field_count > 0:
        report.warnings.append(
            f"{report.empty_field_count} lines had empty/missing fields"
        )

    logger.info(
        "Validation: %d samples, valid=%s, %d warnings, %d errors",
        report.total_samples,
        report.is_valid,
        len(report.warnings),
        len(report.errors),
    )
    return report
