# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact logprob evaluator core
"""Core evaluation logic for Gemma AggreFact logprob judging."""

from __future__ import annotations

import importlib
import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from _gemma_aggrefact_eval_schema import AggreFactDataset, DatasetLoader

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are a fact-checking assistant. Decide if the CLAIM is fully supported by the CONTEXT.

CONTEXT:
{premise}

CLAIM:
{hypothesis}

Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."""


class LogprobLlamaModel(Protocol):
    """Protocol for the llama-cpp logprob chat-completion surface."""

    def tokenize(self, value: bytes, *, add_bos: bool) -> Sequence[int]:
        """Return token IDs for a byte prompt fragment."""

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        logprobs: bool,
        top_logprobs: int,
    ) -> Mapping[str, object]:
        """Create a chat completion with next-token logprob metadata."""


class LogprobLlamaFactory(Protocol):
    """Factory protocol for constructing a llama-cpp logprob model."""

    def __call__(self, **kwargs: object) -> LogprobLlamaModel:
        """Construct a llama-cpp model."""


class LogprobJudgeBackend(Protocol):
    """Backend protocol for continuous supported-claim scores."""

    def judge(self, premise: str, hypothesis: str) -> tuple[float, str]:
        """Return ``(P(SUPPORTED), raw_text)`` for one premise/claim pair."""


class LlamaCppLogprobBackend:
    """llama-cpp backend that extracts continuous support probabilities."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 2) -> None:
        """Load a llama-cpp model through the public constructor surface."""
        llama_module = importlib.import_module("llama_cpp")
        llama_factory = cast(LogprobLlamaFactory, vars(llama_module)["Llama"])
        logger.info("Loading: %s", model_path)
        self.llm = llama_factory(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            verbose=False,
            logits_all=True,
        )
        supported_tokens = self.llm.tokenize(b" SUPPORTED", add_bos=False)
        not_tokens = self.llm.tokenize(b" NOT", add_bos=False)
        logger.info("token ids: SUPPORTED=%s, NOT=%s", supported_tokens, not_tokens)
        self.tok_supported = supported_tokens[0] if supported_tokens else None
        self.tok_not = not_tokens[0] if not_tokens else None
        logger.info("Loaded")

    def judge(self, premise: str, hypothesis: str) -> tuple[float, str]:
        """Return a continuous support score and raw model response."""
        prompt = JUDGE_PROMPT.format(premise=premise, hypothesis=hypothesis)
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
        )
        text = _chat_content(response).strip()
        return _score_from_logprobs(response, text), text


def compute_balanced_accuracy(
    scores: Sequence[float | None],
    labels: Sequence[int],
    threshold: float = 0.5,
) -> float:
    """Compute balanced accuracy while ignoring missing score values."""
    positive = 0
    negative = 0
    true_positive = 0
    true_negative = 0
    for score, label in zip(scores, labels, strict=True):
        if score is None:
            continue
        prediction = 1 if score >= threshold else 0
        if label == 1:
            positive += 1
            if prediction == 1:
                true_positive += 1
        else:
            negative += 1
            if prediction == 0:
                true_negative += 1
    if positive == 0 or negative == 0:
        return 0.0
    return (true_positive / positive + true_negative / negative) / 2


def sweep_threshold(
    scores: Sequence[float | None], labels: Sequence[int]
) -> tuple[float, float]:
    """Find the best balanced-accuracy threshold over the 0.05..0.95 grid."""
    best_threshold = 0.5
    best_balanced_accuracy = 0.0
    for threshold in (0.05 * index for index in range(1, 20)):
        balanced_accuracy = compute_balanced_accuracy(scores, labels, threshold)
        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_threshold = threshold
    return best_threshold, best_balanced_accuracy


def per_dataset_sweep(
    scores: Sequence[float | None],
    labels: Sequence[int],
    datasets: Sequence[str],
) -> tuple[dict[str, dict[str, float | int]], float]:
    """Compute optimal-threshold balanced accuracy for every dataset subset."""
    grouped: dict[str, tuple[list[float | None], list[int]]] = {}
    for score, label, dataset_name in zip(scores, labels, datasets, strict=True):
        dataset_scores, dataset_labels = grouped.setdefault(dataset_name, ([], []))
        dataset_scores.append(score)
        dataset_labels.append(label)

    metrics: dict[str, dict[str, float | int]] = {}
    total_balanced_accuracy = 0.0
    for dataset_name, (dataset_scores, dataset_labels) in grouped.items():
        threshold, balanced_accuracy = sweep_threshold(dataset_scores, dataset_labels)
        metrics[dataset_name] = {
            "samples": len(dataset_labels),
            "balanced_accuracy": balanced_accuracy,
            "threshold": threshold,
        }
        total_balanced_accuracy += balanced_accuracy
    dataset_count = len(metrics)
    return metrics, total_balanced_accuracy / dataset_count if dataset_count else 0.0


def load_aggrefact(max_samples: int | None = None) -> AggreFactDataset:
    """Load and optionally truncate the gated AggreFact test split."""
    datasets_module = importlib.import_module("datasets")
    load_dataset = cast(DatasetLoader, vars(datasets_module)["load_dataset"])
    dataset = load_dataset("lytang/LLM-AggreFact", split="test")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    return dataset


def evaluate_dataset(
    dataset: AggreFactDataset,
    backend: LogprobJudgeBackend,
    *,
    log_every: int,
) -> tuple[list[float | None], list[int], list[str], list[str], list[float], float]:
    """Run logprob judging across an AggreFact dataset."""
    scores: list[float | None] = []
    labels: list[int] = []
    datasets_per_sample: list[str] = []
    raw_responses: list[str] = []
    latencies: list[float] = []
    total_samples = len(dataset)
    started_at = time.time()

    for index, sample in enumerate(dataset):
        premise = _row_text(sample, "doc", "document")
        hypothesis = _row_text(sample, "claim", "hypothesis")
        label = _row_label(sample)
        dataset_name = _row_dataset(sample)

        sample_started_at = time.time()
        try:
            score, text = backend.judge(premise, hypothesis)
        except (KeyError, RuntimeError, TypeError, ValueError, OSError) as exc:
            logger.warning("Sample %d failed: %s", index, exc)
            score, text = None, "ERROR"
        latencies.append(time.time() - sample_started_at)

        scores.append(score)
        labels.append(label)
        datasets_per_sample.append(dataset_name)
        raw_responses.append(text[:32])

        if (index + 1) % log_every == 0:
            elapsed = time.time() - started_at
            eta = (total_samples - index - 1) * elapsed / (index + 1) / 60
            logger.info(
                "[%d/%d] BA@0.5=%.4f %.0fms/sample ETA=%.1fmin",
                index + 1,
                total_samples,
                compute_balanced_accuracy(scores, labels, 0.5),
                1000 * elapsed / (index + 1),
                eta,
            )

    return scores, labels, datasets_per_sample, raw_responses, latencies, started_at


def _row_text(row: Mapping[str, object], primary: str, fallback: str) -> str:
    """Read a required text field with AggreFact schema compatibility."""
    value = row.get(primary, row.get(fallback))
    if value is None:
        raise ValueError(f"AggreFact row missing {primary!r}/{fallback!r}")
    return value if isinstance(value, str) else str(value)


def _row_label(row: Mapping[str, object]) -> int:
    """Read a binary AggreFact label with legacy annotation compatibility."""
    value = row.get("label", row.get("annotations"))
    if value is None:
        raise ValueError("AggreFact row missing 'label'/'annotations'")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"invalid label value: {value!r}")


def _row_dataset(row: Mapping[str, object]) -> str:
    """Read the source dataset name from an AggreFact row."""
    value = row.get("dataset", "unknown")
    return value if isinstance(value, str) else str(value)


def _chat_content(response: Mapping[str, object]) -> str:
    """Extract the first chat-completion message content."""
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return ""
    if not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return ""
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content", "")
    return content if isinstance(content, str) else str(content)


def _score_from_logprobs(response: Mapping[str, object], text: str) -> float:
    """Convert first-token logprob metadata into ``P(SUPPORTED)``."""
    probability_supported = 0.0
    probability_not = 0.0
    for entry in _first_top_logprobs(response):
        if not isinstance(entry, Mapping):
            continue
        token = _normalise_token(entry.get("token", ""))
        logprob = _float_or_default(entry.get("logprob"), -1e9)
        if "SUPPORTED" in token and "NOT" not in token:
            probability_supported += math.exp(logprob)
        elif "NOT" in token or token in {"UN", "NO"}:
            probability_not += math.exp(logprob)
    denominator = probability_supported + probability_not
    if denominator > 0:
        return probability_supported / denominator

    normalised_text = text.upper()
    if "NOT" in normalised_text:
        return 0.0
    if "SUPPORTED" in normalised_text:
        return 1.0
    return 0.5


def _first_top_logprobs(response: Mapping[str, object]) -> Sequence[object]:
    """Return the first generated token's top-logprob entries."""
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return []
    if not choices:
        return []
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return []
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, Mapping):
        return []
    content = logprobs.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return []
    if not content:
        return []
    first = content[0]
    if not isinstance(first, Mapping):
        return []
    top_logprobs = first.get("top_logprobs", [])
    if not isinstance(top_logprobs, Sequence) or isinstance(top_logprobs, (str, bytes)):
        return []
    return top_logprobs


def _normalise_token(value: object) -> str:
    """Return an uppercase token string without surrounding whitespace."""
    return (
        value.strip().upper() if isinstance(value, str) else str(value).strip().upper()
    )


def _float_or_default(value: object, default: float) -> float:
    """Coerce numeric logprob metadata while preserving a fallback value."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default
