# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact routed HiSS evaluator core
"""Core evaluation logic for routed Gemma AggreFact HiSS judging."""

from __future__ import annotations

import importlib
import logging
import time
from collections import defaultdict
from collections.abc import Mapping
from typing import TypedDict, cast

from _gemma_aggrefact_eval_schema import (
    AggreFactDataset,
    DatasetLoader,
    LlamaFactory,
    LlamaModel,
)
from _judge_common import (
    DATASET_TO_FAMILY,
    DECOMPOSE_PROMPT,
    PROMPTS,
    compute_balanced_accuracy,
    parse_response,
    parse_subclaims,
)

logger = logging.getLogger(__name__)


class HiSSRoutedEvaluation(TypedDict):
    """In-memory routed HiSS evaluation result before report rendering."""

    preds: list[int]
    support_fractions: list[float | None]
    labels: list[int]
    datasets_per_sample: list[str]
    families_per_sample: list[str]
    subclaim_counts: list[int]
    decomposed_flags: list[bool]
    latencies: list[float]
    unknown_predictions: int
    skipped_decompose: int
    started_at: float


def load_aggrefact(max_samples: int | None) -> AggreFactDataset:
    """Load and optionally truncate the gated AggreFact test split."""
    datasets_module = importlib.import_module("datasets")
    load_dataset = cast(DatasetLoader, vars(datasets_module)["load_dataset"])
    dataset = load_dataset("lytang/LLM-AggreFact", split="test")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    return dataset


def build_llama(model_path: str, *, n_ctx: int, n_threads: int) -> LlamaModel:
    """Construct the llama-cpp model through the public factory surface."""
    llama_module = importlib.import_module("llama_cpp")
    llama_factory = cast(LlamaFactory, vars(llama_module)["Llama"])
    logger.info("Loading: %s", model_path)
    model = llama_factory(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=512,
        verbose=False,
        logits_all=False,
    )
    logger.info("Loaded")
    return model


def family_distribution(dataset: AggreFactDataset) -> dict[str, int]:
    """Return routed task-family counts for operator logging."""
    family_counts: defaultdict[str, int] = defaultdict(int)
    for sample in dataset:
        dataset_name = _row_dataset(sample)
        family_counts[DATASET_TO_FAMILY.get(dataset_name, "claim")] += 1
    return dict(family_counts)


def evaluate_dataset(
    dataset: AggreFactDataset,
    llm: LlamaModel,
    *,
    min_decompose_words: int,
    support_frac: float,
    max_subclaims: int,
    max_decompose_tokens: int,
    max_verify_tokens: int,
    log_every: int,
) -> HiSSRoutedEvaluation:
    """Run routed HiSS decomposition and verification across a dataset."""
    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError("dataset is empty")

    preds: list[int] = []
    support_fractions: list[float | None] = []
    labels: list[int] = []
    datasets_per_sample: list[str] = []
    families_per_sample: list[str] = []
    subclaim_counts: list[int] = []
    decomposed_flags: list[bool] = []
    latencies: list[float] = []
    unknown_predictions = 0
    skipped_decompose = 0
    started_at = time.time()

    for index, sample in enumerate(dataset):
        premise = _row_text(sample, "doc", "document")
        hypothesis = _row_text(sample, "claim", "hypothesis")
        label = _row_label(sample)
        dataset_name = _row_dataset(sample)
        family = DATASET_TO_FAMILY.get(dataset_name, "claim")
        verify_template = PROMPTS[family]

        sample_started_at = time.time()
        word_count = len(hypothesis.split())
        if word_count < min_decompose_words:
            skipped_decompose += 1
            decomposed = False
            subclaims = [hypothesis]
        else:
            decomposed = True
            subclaims = _decompose_claim(
                llm,
                hypothesis=hypothesis,
                max_tokens=max_decompose_tokens,
                max_subclaims=max_subclaims,
                sample_index=index,
            )

        sub_verdicts = [
            _verify_claim(
                llm,
                verify_template=verify_template,
                premise=premise,
                hypothesis=subclaim,
                max_tokens=max_verify_tokens,
                sample_index=index,
            )
            for subclaim in subclaims
        ]
        pred, support_fraction = _aggregate_subverdicts(
            sub_verdicts,
            support_frac=support_frac,
        )
        if pred < 0:
            unknown_predictions += 1

        latencies.append(time.time() - sample_started_at)
        preds.append(pred)
        support_fractions.append(support_fraction)
        labels.append(label)
        datasets_per_sample.append(dataset_name)
        families_per_sample.append(family)
        subclaim_counts.append(len(subclaims))
        decomposed_flags.append(decomposed)

        if (index + 1) % log_every == 0:
            elapsed = time.time() - started_at
            eta = (total_samples - index - 1) * elapsed / (index + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f unk=%d skip=%d %.0fms/sample ETA=%.1fmin",
                index + 1,
                total_samples,
                compute_balanced_accuracy(preds, labels),
                unknown_predictions,
                skipped_decompose,
                1000 * elapsed / (index + 1),
                eta,
            )

    return {
        "preds": preds,
        "support_fractions": support_fractions,
        "labels": labels,
        "datasets_per_sample": datasets_per_sample,
        "families_per_sample": families_per_sample,
        "subclaim_counts": subclaim_counts,
        "decomposed_flags": decomposed_flags,
        "latencies": latencies,
        "unknown_predictions": unknown_predictions,
        "skipped_decompose": skipped_decompose,
        "started_at": started_at,
    }


def _decompose_claim(
    llm: LlamaModel,
    *,
    hypothesis: str,
    max_tokens: int,
    max_subclaims: int,
    sample_index: int,
) -> list[str]:
    """Decompose one claim through the llama-cpp chat-completion surface."""
    try:
        output = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": DECOMPOSE_PROMPT.format(claim=hypothesis),
                },
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        raw = _chat_content(output)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Sample %d decompose failed: %s", sample_index, exc)
        raw = ""
    return parse_subclaims(raw, hypothesis, max_n=max_subclaims)


def _verify_claim(
    llm: LlamaModel,
    *,
    verify_template: str,
    premise: str,
    hypothesis: str,
    max_tokens: int,
    sample_index: int,
) -> int:
    """Verify one routed claim or subclaim against the source context."""
    try:
        output = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": verify_template.format(
                        premise=premise,
                        hypothesis=hypothesis,
                    ),
                },
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return parse_response(_chat_content(output))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Sample %d verify failed: %s", sample_index, exc)
        return -1


def _aggregate_subverdicts(
    sub_verdicts: list[int],
    *,
    support_frac: float,
) -> tuple[int, float | None]:
    """Aggregate routed subclaim verdicts into one prediction and score."""
    valid_verdicts = [verdict for verdict in sub_verdicts if verdict >= 0]
    if not valid_verdicts:
        return -1, None
    supported = sum(1 for verdict in valid_verdicts if verdict == 1)
    support_fraction = supported / len(valid_verdicts)
    return (1 if support_fraction >= support_frac else 0), support_fraction


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
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return ""
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content", "")
    return content if isinstance(content, str) else str(content)
