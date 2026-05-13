# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HalluBench benchmark harness

"""Evaluate supplied VLM predictions on HalluBench without storing gated data.

HalluBench is a gated, non-commercial, no-derivatives geospatial VQA benchmark.
This runner intentionally separates dataset access from model execution:
operators provide a JSONL file of predictions keyed by ``question_id``. The
runner loads the gated rows locally, computes aggregate metrics, and writes only
tenant-safe/result-safe metadata. Raw images, questions, ground-truth answers,
and model predictions are not copied into result files.

Usage::

    python -m benchmarks.hallubench_eval \
      --predictions-jsonl validation/hallubench_predictions.jsonl \
      --output-json benchmarks/results/hallubench_internal_validation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_ID = "AuwAuwAuw/HalluBench"
DATASET_SPLIT = "train"
RESULTS_DIR = Path(__file__).parent / "results"
_VALID_APPLICATIONS = frozenset(("emergency", "urban"))
_VALID_TASKS = frozenset(("recognition", "report", "counting", "loc_reason"))
_VALID_OUTPUT_FORMS = frozenset(("short", "long"))
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")

logger = logging.getLogger("DirectorAI.Benchmark.HalluBench")


class HalluBenchAccessError(RuntimeError):
    """Raised when the gated HalluBench dataset cannot be loaded."""


class HalluBenchInputError(ValueError):
    """Raised when runner inputs are malformed."""


@dataclass(frozen=True)
class ImageRef:
    """Reference to a HalluBench image path without image payloads."""

    path: str
    modality: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "modality": self.modality, "role": self.role}


@dataclass(frozen=True)
class HalluBenchSample:
    """Normalised HalluBench row with raw media excluded."""

    question_id: str
    two_images: bool
    is_temporal: bool
    image_refs: tuple[ImageRef, ...]
    application: str
    sub_application: str
    task_type: str
    output_form: str
    question: str
    ground_truth: str
    source_dataset: str
    original_id: str
    original_qtype: str

    def to_result_dict(self) -> dict[str, Any]:
        """Serialise non-sensitive benchmark metadata only."""
        return {
            "question_id": self.question_id,
            "two_images": self.two_images,
            "is_temporal": self.is_temporal,
            "image_refs": [ref.to_dict() for ref in self.image_refs],
            "application": self.application,
            "sub_application": self.sub_application,
            "task_type": self.task_type,
            "output_form": self.output_form,
            "source_dataset": self.source_dataset,
            "original_id": self.original_id,
            "original_qtype": self.original_qtype,
        }


@dataclass(frozen=True)
class PredictionMetric:
    """Per-sample answer agreement metrics."""

    question_id: str
    exact_match: bool
    numeric_match: bool
    token_f1: float
    passed: bool
    reason: str

    def to_result_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "exact_match": self.exact_match,
            "numeric_match": self.numeric_match,
            "token_f1": round(self.token_f1, 6),
            "passed": self.passed,
            "reason": self.reason,
        }


def _hf_load_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def load_hallubench_rows(
    *,
    split: str = DATASET_SPLIT,
    hf_token: str | None = None,
) -> list[dict[str, Any]]:
    """Load HalluBench rows from Hugging Face with a clear gated-access error."""
    token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
    try:
        dataset = _hf_load_dataset(DATASET_ID, split=split, token=token)
    except Exception as exc:  # noqa: BLE001 - wrap HF-specific failures for CLI users
        raise HalluBenchAccessError(
            "Could not load gated HalluBench dataset. Accept access conditions on "
            "Hugging Face and provide HF_TOKEN or a local HF login."
        ) from exc
    return [dict(row) for row in dataset]


def normalise_sample(row: Mapping[str, Any]) -> HalluBenchSample:
    """Validate and normalise one HalluBench row."""
    question_id = _required_str(row, "question_id")
    application = _enum(row, "application", _VALID_APPLICATIONS)
    task_type = _enum(row, "task_type", _VALID_TASKS)
    output_form = _enum(row, "output_form", _VALID_OUTPUT_FORMS)
    question = _required_str(row, "question")
    ground_truth = _required_str(row, "ground_truth")
    two_images = _bool_field(row.get("two_images"))
    is_temporal = _bool_field(row.get("is_temporal"))

    image_refs: list[ImageRef] = []
    img1 = str(row.get("img1_path", "")).strip()
    if img1:
        image_refs.append(
            ImageRef(
                path=img1,
                modality=str(row.get("img1_type", "")).strip() or "unknown",
                role="img1",
            )
        )
    img2 = str(row.get("img2_path", "")).strip()
    if img2:
        image_refs.append(
            ImageRef(
                path=img2,
                modality=str(row.get("img2_type", "")).strip() or "unknown",
                role="img2",
            )
        )

    if two_images and len(image_refs) != 2:
        raise HalluBenchInputError(f"{question_id}: two_images row requires img1/img2")
    if not two_images and not image_refs:
        raise HalluBenchInputError(
            f"{question_id}: at least one image path is required"
        )

    return HalluBenchSample(
        question_id=question_id,
        two_images=two_images,
        is_temporal=is_temporal,
        image_refs=tuple(image_refs),
        application=application,
        sub_application=str(row.get("sub_application", "")).strip(),
        task_type=task_type,
        output_form=output_form,
        question=question,
        ground_truth=ground_truth,
        source_dataset=str(row.get("source_dataset", "")).strip(),
        original_id=str(row.get("original_id", "")).strip(),
        original_qtype=str(row.get("original_qtype", "")).strip(),
    )


def evaluate_prediction(
    sample: HalluBenchSample,
    prediction: str,
    *,
    numeric_tolerance: float = 0.0,
    long_answer_f1_threshold: float = 0.45,
) -> PredictionMetric:
    """Evaluate one prediction against a HalluBench reference answer."""
    pred_text = str(prediction).strip()
    if not pred_text:
        return PredictionMetric(
            question_id=sample.question_id,
            exact_match=False,
            numeric_match=False,
            token_f1=0.0,
            passed=False,
            reason="empty_prediction",
        )

    exact = _normalise_text(pred_text) == _normalise_text(sample.ground_truth)
    numeric = _numeric_match(
        pred_text,
        sample.ground_truth,
        tolerance=numeric_tolerance,
    )
    token_f1 = _token_f1(pred_text, sample.ground_truth)
    if sample.output_form == "short":
        passed = exact or numeric or token_f1 >= 0.9
        reason = "short_answer_match" if passed else "short_answer_mismatch"
    else:
        passed = token_f1 >= long_answer_f1_threshold
        reason = "long_answer_reference_overlap" if passed else "long_answer_mismatch"

    return PredictionMetric(
        question_id=sample.question_id,
        exact_match=exact,
        numeric_match=numeric,
        token_f1=token_f1,
        passed=passed,
        reason=reason,
    )


def run_hallubench_benchmark(
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    predictions_jsonl: Path | str,
    split: str = DATASET_SPLIT,
    applications: Sequence[str] = (),
    task_types: Sequence[str] = (),
    output_forms: Sequence[str] = (),
    max_samples: int | None = None,
    model_id: str = "",
    numeric_tolerance: float = 0.0,
    long_answer_f1_threshold: float = 0.45,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Run HalluBench evaluation from local predictions."""
    start = time.monotonic()
    if rows is None:
        loaded_rows = load_hallubench_rows(split=split, hf_token=hf_token)
    else:
        loaded_rows = [dict(row) for row in rows]

    samples = _filter_samples(
        [normalise_sample(row) for row in loaded_rows],
        applications=applications,
        task_types=task_types,
        output_forms=output_forms,
        max_samples=max_samples,
    )
    predictions = load_predictions(Path(predictions_jsonl))

    per_sample: list[dict[str, Any]] = []
    by_application: dict[str, Counter[str]] = defaultdict(Counter)
    by_task_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_output_form: dict[str, Counter[str]] = defaultdict(Counter)
    by_temporal: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    modalities: Counter[str] = Counter()

    for sample in samples:
        for ref in sample.image_refs:
            modalities[ref.modality] += 1
        pred = predictions.get(sample.question_id)
        if pred is None:
            metric = PredictionMetric(
                question_id=sample.question_id,
                exact_match=False,
                numeric_match=False,
                token_f1=0.0,
                passed=False,
                reason="missing_prediction",
            )
        else:
            metric = evaluate_prediction(
                sample,
                pred,
                numeric_tolerance=numeric_tolerance,
                long_answer_f1_threshold=long_answer_f1_threshold,
            )

        _update_counts(overall, metric)
        _update_counts(by_application[sample.application], metric)
        _update_counts(by_task_type[sample.task_type], metric)
        _update_counts(by_output_form[sample.output_form], metric)
        _update_counts(
            by_temporal["temporal" if sample.is_temporal else "single"], metric
        )
        entry = sample.to_result_dict()
        entry["metrics"] = metric.to_result_dict()
        entry["ground_truth_sha256"] = _sha256(sample.ground_truth)
        if pred is not None:
            entry["prediction_sha256"] = _sha256(pred)
        per_sample.append(entry)

    elapsed = time.monotonic() - start
    return {
        "benchmark": "HalluBench",
        "schema_version": "1.0.0",
        "benchmark_evidence": False,
        "claim_boundary": (
            "Internal benchmark harness output only. Do not publish as a public "
            "claim until dataset access, model provenance, and metric review are "
            "recorded in the external validation packet."
        ),
        "dataset": {
            "id": "hallubench",
            "source": DATASET_ID,
            "split": split,
            "access": "gated; requires accepted Hugging Face access conditions",
            "licence": "CC BY-NC-ND 4.0; non-commercial use only; no derivatives",
            "raw_data_committed": False,
        },
        "model_id": model_id,
        "parameters": {
            "numeric_tolerance": numeric_tolerance,
            "long_answer_f1_threshold": long_answer_f1_threshold,
            "applications": list(applications),
            "task_types": list(task_types),
            "output_forms": list(output_forms),
            "max_samples": max_samples,
        },
        "overall": _summary(overall),
        "by_application": _summaries(by_application),
        "by_task_type": _summaries(by_task_type),
        "by_output_form": _summaries(by_output_form),
        "by_temporal": _summaries(by_temporal),
        "image_modalities": dict(sorted(modalities.items())),
        "elapsed_seconds": round(elapsed, 6),
        "per_sample": per_sample,
    }


def load_predictions(path: Path) -> dict[str, str]:
    """Load prediction JSONL keyed by question_id."""
    if not path.exists():
        raise HalluBenchInputError(f"prediction file not found: {path}")
    predictions: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise HalluBenchInputError(
                f"{path}:{line_number}: invalid JSONL prediction row"
            ) from exc
        question_id = str(payload.get("question_id", "")).strip()
        prediction = str(payload.get("prediction", "")).strip()
        if not question_id:
            raise HalluBenchInputError(f"{path}:{line_number}: question_id is required")
        if question_id in predictions:
            raise HalluBenchInputError(f"{path}:{line_number}: duplicate {question_id}")
        predictions[question_id] = prediction
    return predictions


def _filter_samples(
    samples: Sequence[HalluBenchSample],
    *,
    applications: Sequence[str],
    task_types: Sequence[str],
    output_forms: Sequence[str],
    max_samples: int | None,
) -> list[HalluBenchSample]:
    app_filter = _validated_filter("applications", applications, _VALID_APPLICATIONS)
    task_filter = _validated_filter("task_types", task_types, _VALID_TASKS)
    form_filter = _validated_filter("output_forms", output_forms, _VALID_OUTPUT_FORMS)
    selected = [
        sample
        for sample in samples
        if (not app_filter or sample.application in app_filter)
        and (not task_filter or sample.task_type in task_filter)
        and (not form_filter or sample.output_form in form_filter)
    ]
    if max_samples is not None:
        if max_samples < 1:
            raise HalluBenchInputError("max_samples must be >= 1")
        selected = selected[:max_samples]
    return selected


def _validated_filter(
    name: str,
    values: Sequence[str],
    allowed: frozenset[str],
) -> frozenset[str]:
    normalised = frozenset(value.strip() for value in values if value.strip())
    invalid = normalised - allowed
    if invalid:
        raise HalluBenchInputError(f"invalid {name}: {sorted(invalid)}")
    return normalised


def _update_counts(counter: Counter[str], metric: PredictionMetric) -> None:
    counter["total"] += 1
    counter["passed"] += int(metric.passed)
    counter["failed"] += int(not metric.passed)
    counter["exact_match"] += int(metric.exact_match)
    counter["numeric_match"] += int(metric.numeric_match)
    counter["missing_predictions"] += int(metric.reason == "missing_prediction")
    counter["token_f1_sum"] += metric.token_f1


def _summary(counter: Counter[str]) -> dict[str, Any]:
    total = int(counter["total"])
    return {
        "total": total,
        "passed": int(counter["passed"]),
        "failed": int(counter["failed"]),
        "missing_predictions": int(counter["missing_predictions"]),
        "accuracy": round(counter["passed"] / total, 6) if total else 0.0,
        "exact_match_rate": round(counter["exact_match"] / total, 6) if total else 0.0,
        "numeric_match_rate": (
            round(counter["numeric_match"] / total, 6) if total else 0.0
        ),
        "mean_token_f1": round(counter["token_f1_sum"] / total, 6) if total else 0.0,
    }


def _summaries(counters: Mapping[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    return {key: _summary(counter) for key, counter in sorted(counters.items())}


def _bool_field(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return True
    if text in {"no", "false", "0", ""}:
        return False
    raise HalluBenchInputError(f"invalid boolean field value {value!r}")


def _required_str(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise HalluBenchInputError(f"{key} is required")
    return value


def _enum(row: Mapping[str, Any], key: str, allowed: frozenset[str]) -> str:
    value = _required_str(row, key)
    if value not in allowed:
        raise HalluBenchInputError(f"{key} must be one of {sorted(allowed)}")
    return value


def _normalise_text(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.lower()))


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _TOKEN_RE.findall(prediction.lower())
    gold_tokens = _TOKEN_RE.findall(ground_truth.lower())
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _numeric_match(prediction: str, ground_truth: str, *, tolerance: float) -> bool:
    pred_numbers = _numbers(prediction)
    gold_numbers = _numbers(ground_truth)
    if not pred_numbers or not gold_numbers:
        return False
    for pred in pred_numbers:
        for gold in gold_numbers:
            if math.isclose(pred, gold, rel_tol=0.0, abs_tol=max(0.0, tolerance)):
                return True
    return False


def _numbers(value: str) -> list[float]:
    return [float(match.group(0)) for match in _NUMBER_RE.finditer(value)]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_csv_option(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HalluBench prediction evaluator")
    parser.add_argument("--predictions-jsonl", required=True, type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULTS_DIR / "hallubench_internal_validation.json",
    )
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--applications", default="")
    parser.add_argument("--task-types", default="")
    parser.add_argument("--output-forms", default="")
    parser.add_argument("--numeric-tolerance", type=float, default=0.0)
    parser.add_argument("--long-answer-f1-threshold", type=float, default=0.45)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args(argv)
    result = run_hallubench_benchmark(
        predictions_jsonl=args.predictions_jsonl,
        split=args.split,
        applications=_parse_csv_option(args.applications),
        task_types=_parse_csv_option(args.task_types),
        output_forms=_parse_csv_option(args.output_forms),
        max_samples=args.max_samples,
        model_id=args.model_id,
        numeric_tolerance=args.numeric_tolerance,
        long_answer_f1_threshold=args.long_answer_f1_threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {args.output_json}")
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
