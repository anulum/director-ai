# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Anti-Regression Benchmark Gate

"""Benchmark a fine-tuned NLI model against a held-out general dataset.

Detects catastrophic forgetting by comparing the fine-tuned model's
general accuracy against the baseline. Three outcomes:

- ``deploy``          — regression < 3pp, safe as default scorer
- ``deploy_domain_only`` — regression 3-8pp, use only via domain backend
- ``reject``          — regression > 8pp, catastrophic forgetting

Usage::

    from director_ai.core.finetune_benchmark import benchmark_finetuned_model
    report = benchmark_finetuned_model("./my-model", eval_path="domain_eval.jsonl")
    print(report.recommendation)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_registry import (
    TrainingModelProfile,
    resolve_finetune_model,
)

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


@dataclass
class ModelBenchmarkResult:
    """Benchmark result for one model candidate in a model-selection sweep."""

    requested_model: str
    alias: str
    model_id: str
    model_path: str
    status: str
    template: str
    label_count: int
    baseline_accuracy: float
    recommended_batch_size: int
    domain_accuracy: float = 0.0
    domain_f1: float = 0.0
    general_accuracy: float = 0.0
    general_f1: float = 0.0
    regression_pp: float = 0.0
    recommendation: str = "reject"
    elapsed_seconds: float = 0.0
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_report(
        cls,
        *,
        requested_model: str,
        profile: TrainingModelProfile,
        model_path: str | Path,
        report: RegressionReport,
        elapsed_seconds: float,
    ) -> ModelBenchmarkResult:
        alias = (
            requested_model if profile.alias == "custom-experimental" else profile.alias
        )
        return cls(
            requested_model=requested_model,
            alias=alias,
            model_id=profile.model_id,
            model_path=str(model_path),
            status=profile.status,
            template=profile.template,
            label_count=profile.label_count,
            baseline_accuracy=report.baseline_accuracy,
            recommended_batch_size=profile.recommended_batch_size,
            domain_accuracy=report.domain_accuracy,
            domain_f1=report.domain_f1,
            general_accuracy=report.general_accuracy,
            general_f1=report.general_f1,
            regression_pp=report.regression_pp,
            recommendation=report.recommendation,
            elapsed_seconds=elapsed_seconds,
            details=dict(report.details),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "alias": self.alias,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "status": self.status,
            "template": self.template,
            "label_count": self.label_count,
            "baseline_accuracy": self.baseline_accuracy,
            "recommended_batch_size": self.recommended_batch_size,
            "domain_accuracy": self.domain_accuracy,
            "domain_f1": self.domain_f1,
            "general_accuracy": self.general_accuracy,
            "general_f1": self.general_f1,
            "regression_pp": self.regression_pp,
            "recommendation": self.recommendation,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "details": self.details,
        }


@dataclass
class ModelBenchmarkReport:
    """Same-input benchmark sweep across model candidates."""

    results: list[ModelBenchmarkResult]
    general_path: str = ""
    eval_path: str = ""
    generated_at: float = field(default_factory=time.time)
    seed: int = 42
    best_model_alias: str = ""
    best_model_id: str = ""
    selection_policy: str = "deployable_highest_general_accuracy"

    def __post_init__(self) -> None:
        if not self.best_model_alias:
            winner = self._select_winner()
            if winner is not None:
                self.best_model_alias = winner.alias
                self.best_model_id = winner.model_id

    def _select_winner(self) -> ModelBenchmarkResult | None:
        candidates = [
            result
            for result in self.results
            if not result.error and result.recommendation != "reject"
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda result: (
                result.general_accuracy,
                result.domain_accuracy,
                -result.elapsed_seconds,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "general_path": self.general_path,
            "eval_path": self.eval_path,
            "seed": self.seed,
            "selection_policy": self.selection_policy,
            "best_model_alias": self.best_model_alias,
            "best_model_id": self.best_model_id,
            "results": [result.to_dict() for result in self.results],
        }

    def summary(self) -> str:
        lines = [
            "Model benchmark sweep",
            f"General data: {self.general_path or 'not provided'}",
            f"Domain data: {self.eval_path or 'not provided'}",
            f"Best model: {self.best_model_alias or 'none'}",
        ]
        for result in self.results:
            suffix = f" error={result.error}" if result.error else ""
            lines.append(
                f"- {result.alias}: general={result.general_accuracy:.1%}, "
                f"domain={result.domain_accuracy:.1%}, rec={result.recommendation}"
                f"{suffix}",
            )
        return "\n".join(lines)


def _load_benchmark_jsonl(path: str | Path) -> list[dict]:
    path_obj = Path(path).expanduser().resolve(strict=True)
    if not path_obj.is_file():
        raise FileNotFoundError(f"benchmark data path is not a file: {path_obj}")
    rows = []
    with path_obj.open(encoding="utf-8") as f:
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

    from .._device import release_torch_cuda, select_torch_device

    device = torch.device(select_torch_device())
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

    try:
        all_preds: list[int] = []
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
            all_preds.extend(int(p) for p in preds.flatten())

        bal_acc = _balanced_accuracy(labels, all_preds)
        f1 = _binary_f1_score(labels, all_preds)
        return {"balanced_accuracy": bal_acc, "f1": f1}
    finally:
        with suppress(Exception):
            model.to("cpu")
        release_torch_cuda()


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
        logger.warning("No general benchmark data — skipping regression check")
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


def benchmark_model_candidates(
    model_artifacts: Mapping[str, str | Path],
    *,
    general_path: str | Path | None = None,
    eval_path: str | Path | None = None,
    batch_size: int | None = None,
    allow_experimental: bool = False,
    seed: int = 42,
) -> ModelBenchmarkReport:
    """Benchmark already-trained model artifacts with identical inputs.

    ``model_artifacts`` maps a registry alias or model id to the artifact path
    produced by a fine-tune job. Unit tests can patch ``_evaluate_model``; live
    runs use the same anti-regression benchmark gate as single-model activation.
    """

    if not model_artifacts:
        raise ValueError("model_artifacts must contain at least one model")

    results: list[ModelBenchmarkResult] = []
    for requested_model, model_path in model_artifacts.items():
        started = time.perf_counter()
        try:
            profile = resolve_finetune_model(
                requested_model,
                allow_experimental=allow_experimental,
            )
            baseline = profile.baseline_accuracy or _BASELINE_ACCURACY
            report = benchmark_finetuned_model(
                model_path,
                general_path=general_path,
                eval_path=eval_path,
                baseline_accuracy=baseline,
                batch_size=batch_size or profile.recommended_batch_size,
            )
            results.append(
                ModelBenchmarkResult.from_report(
                    requested_model=requested_model,
                    profile=profile,
                    model_path=model_path,
                    report=report,
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
        except Exception as exc:
            results.append(
                ModelBenchmarkResult(
                    requested_model=requested_model,
                    alias=requested_model,
                    model_id=requested_model,
                    model_path=str(model_path),
                    status="error",
                    template="unknown",
                    label_count=0,
                    baseline_accuracy=0.0,
                    recommended_batch_size=batch_size or 0,
                    recommendation="reject",
                    elapsed_seconds=time.perf_counter() - started,
                    error=str(exc),
                ),
            )

    return ModelBenchmarkReport(
        results=results,
        general_path=str(general_path or ""),
        eval_path=str(eval_path or ""),
        seed=seed,
    )
