# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Anti-Regression Benchmark Gate

"""Benchmark a fine-tuned NLI model against a held-out general dataset.

Detects catastrophic forgetting by comparing the fine-tuned model's
general accuracy against the baseline. Three outcomes:

- ``deploy``          â€” regression < 3pp, safe as default scorer
- ``deploy_domain_only`` â€” regression 3-8pp, use only via domain backend
- ``reject``          â€” regression > 8pp, catastrophic forgetting

Usage::

    from director_ai.core.finetune_benchmark import benchmark_finetuned_model
    report = benchmark_finetuned_model("./my-model", eval_path="domain_eval.jsonl")
    print(report.recommendation)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("DirectorAI.FinetuneBenchmark")

_BASELINE_ACCURACY = 0.758  # FactCG-DeBERTa-v3-Large on AggreFact-XSUM

# Regression thresholds (percentage points)
_DEPLOY_THRESHOLD_PP = 3.0
_REJECT_THRESHOLD_PP = 8.0


@dataclass
class RegressionReport:
    """Result of anti-regression benchmark."""

    domain_accuracy: float = 0.0
    domain_f1: float = 0.0
    general_accuracy: float = 0.0
    general_f1: float = 0.0
    baseline_accuracy: float = _BASELINE_ACCURACY
    regression_pp: float = 0.0
    regression_acceptable: bool = True
    recommendation: str = "deploy"
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Domain bal_acc: {self.domain_accuracy:.1%}",
            f"General bal_acc: {self.general_accuracy:.1%}",
            f"Baseline bal_acc: {self.baseline_accuracy:.1%}",
            f"Regression: {self.regression_pp:+.1f}pp",
            f"Recommendation: {self.recommendation}",
        ]
        return "\n".join(lines)


def _load_benchmark_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            premise = row.get("premise") or row.get("doc") or row.get("context", "")
            hypothesis = (
                row.get("hypothesis") or row.get("claim") or row.get("response", "")
            )
            label = row.get("label")
            if premise and hypothesis and label is not None:
                rows.append(
                    {"premise": premise, "hypothesis": hypothesis, "label": int(label)},
                )
    return rows


def _evaluate_model(
    model_path: str | Path,
    samples: list[dict],
    batch_size: int = 48,
) -> dict:
    """Run a model on samples and return balanced_accuracy + f1."""
    from .finetune import _balanced_accuracy, _binary_f1_score

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("pip install director-ai[finetune]") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    is_factcg = "factcg" in str(model_path).lower()
    if is_factcg:
        from director_ai.core.training.finetune import _FACTCG_TEMPLATE

        texts = [
            _FACTCG_TEMPLATE.format(premise=s["premise"], hypothesis=s["hypothesis"])
            for s in samples
        ]
    else:
        texts = [
            f"{s['premise']} {tokenizer.sep_token} {s['hypothesis']}" for s in samples
        ]
    labels = [s["label"] for s in samples]

    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            logits = model(**encodings).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())

    bal_acc = _balanced_accuracy(labels, all_preds)
    f1 = _binary_f1_score(labels, all_preds)
    return {"balanced_accuracy": bal_acc, "f1": f1}


def benchmark_finetuned_model(
    model_path: str | Path,
    general_path: str | Path | None = None,
    eval_path: str | Path | None = None,
    baseline_accuracy: float = _BASELINE_ACCURACY,
    batch_size: int = 48,
) -> RegressionReport:
    """Benchmark a fine-tuned model for regression.

    Parameters
    ----------
    model_path : path to fine-tuned model directory
    general_path : JSONL with general benchmark samples (shipped AggreFact subset).
                   If None, looks for ``data/aggrefact_benchmark_1k.jsonl`` in package.
    eval_path : optional JSONL with domain-specific eval samples
    baseline_accuracy : baseline balanced accuracy to compare against
    batch_size : inference batch size

    """
    report = RegressionReport(baseline_accuracy=baseline_accuracy)

    # Domain evaluation
    if eval_path is not None:
        domain_samples = _load_benchmark_jsonl(eval_path)
        if domain_samples:
            logger.info("Evaluating domain data: %d samples", len(domain_samples))
            domain_metrics = _evaluate_model(model_path, domain_samples, batch_size)
            report.domain_accuracy = domain_metrics["balanced_accuracy"]
            report.domain_f1 = domain_metrics["f1"]
            report.details["domain_samples"] = len(domain_samples)

    # General evaluation
    if general_path is None:
        pkg_dir = Path(__file__).parent.parent
        candidate = pkg_dir / "data" / "aggrefact_benchmark_1k.jsonl"
        if candidate.exists():
            general_path = candidate

    if general_path is not None:
        general_samples = _load_benchmark_jsonl(general_path)
        if general_samples:
            logger.info(
                "Evaluating general benchmark: %d samples",
                len(general_samples),
            )
            general_metrics = _evaluate_model(model_path, general_samples, batch_size)
            report.general_accuracy = general_metrics["balanced_accuracy"]
            report.general_f1 = general_metrics["f1"]
            report.details["general_samples"] = len(general_samples)
    else:
        logger.warning("No general benchmark data â€” skipping regression check")
        report.details["general_skipped"] = True

    # Regression decision
    if report.general_accuracy > 0:
        report.regression_pp = (report.general_accuracy - baseline_accuracy) * 100
        regression_magnitude = -report.regression_pp  # positive = worse

        if regression_magnitude <= _DEPLOY_THRESHOLD_PP:
            report.recommendation = "deploy"
            report.regression_acceptable = True
        elif regression_magnitude <= _REJECT_THRESHOLD_PP:
            report.recommendation = "deploy_domain_only"
            report.regression_acceptable = False
        else:
            report.recommendation = "reject"
            report.regression_acceptable = False
    else:
        report.recommendation = "deploy_domain_only"
        report.details["reason"] = "no general benchmark available"

    logger.info(
        "Benchmark: domain=%.1f%%, general=%.1f%%, regression=%+.1fpp, rec=%s",
        report.domain_accuracy * 100,
        report.general_accuracy * 100,
        report.regression_pp,
        report.recommendation,
    )
    return report
