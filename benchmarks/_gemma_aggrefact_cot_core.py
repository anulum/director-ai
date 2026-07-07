# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact CoT evaluator core
"""Core evaluation logic for Gemma AggreFact chain-of-thought judging."""

from __future__ import annotations

import importlib
import logging
import re
import time
from collections.abc import Mapping, Sequence
from typing import cast

from _gemma_aggrefact_eval_schema import (
    AggreFactDataset,
    DatasetLoader,
    LlamaFactory,
    LlamaModel,
)

logger = logging.getLogger(__name__)

COT_PROMPT = """You are a careful fact-checking assistant. Decide if the CLAIM is fully supported by the CONTEXT.

CONTEXT:
{premise}

CLAIM:
{hypothesis}

Think step by step in 1-2 short sentences, then on the last line write exactly one of:
ANSWER: SUPPORTED
ANSWER: NOT_SUPPORTED"""


def parse_cot(text: str) -> int:
    """Extract ``SUPPORTED`` or ``NOT_SUPPORTED`` from a CoT response."""
    normalised = text.upper()
    answer = re.search(
        r"ANSWER\s*:\s*(NOT[_\s-]?SUPPORTED|SUPPORTED)",
        normalised,
    )
    if answer:
        return 0 if "NOT" in answer.group(1) else 1
    if "NOT_SUPPORTED" in normalised or "NOT SUPPORTED" in normalised:
        return 0
    if "SUPPORTED" in normalised:
        return 1
    return -1


def compute_ba(preds: Sequence[int], labels: Sequence[int]) -> float:
    """Compute balanced accuracy while ignoring unknown predictions."""
    positive = 0
    negative = 0
    true_positive = 0
    true_negative = 0
    for prediction, label in zip(preds, labels, strict=True):
        if prediction < 0:
            continue
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
    )
    logger.info("Loaded")
    return model


def evaluate_dataset(
    dataset: AggreFactDataset,
    llm: LlamaModel,
    *,
    max_tokens: int,
    log_every: int,
) -> tuple[list[int], list[int], list[str], list[float], list[str], int, float]:
    """Run CoT prompts across an AggreFact dataset."""
    preds: list[int] = []
    labels: list[int] = []
    datasets_per_sample: list[str] = []
    latencies: list[float] = []
    raw_responses: list[str] = []
    unknown = 0
    total_samples = len(dataset)
    started_at = time.time()

    for index, sample in enumerate(dataset):
        prompt = COT_PROMPT.format(
            premise=_row_text(sample, "doc", "document"),
            hypothesis=_row_text(sample, "claim", "hypothesis"),
        )
        label = _row_label(sample)
        dataset_name = _row_dataset(sample)

        sample_started_at = time.time()
        try:
            output = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            response_text = _chat_content(output)
        except Exception as exc:  # noqa: BLE001 - benchmark records per-row failures.
            logger.warning("Sample %d failed: %s", index, exc)
            response_text = "ERROR"
        latencies.append(time.time() - sample_started_at)

        prediction = parse_cot(response_text)
        if prediction < 0:
            unknown += 1
        preds.append(prediction)
        labels.append(label)
        datasets_per_sample.append(dataset_name)
        raw_responses.append(response_text[:80])

        if (index + 1) % log_every == 0:
            elapsed = time.time() - started_at
            eta = (total_samples - index - 1) * elapsed / (index + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f unk=%d %.0fms/sample ETA=%.1fmin",
                index + 1,
                total_samples,
                compute_ba(preds, labels),
                unknown,
                1000 * elapsed / (index + 1),
                eta,
            )

    return (
        preds,
        labels,
        datasets_per_sample,
        latencies,
        raw_responses,
        unknown,
        started_at,
    )


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
