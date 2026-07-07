# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact HiSS evaluator core
"""Core evaluation logic for Gemma AggreFact HiSS judging."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Mapping, Sequence
from typing import TypedDict, cast

from _gemma_aggrefact_eval_schema import (
    AggreFactDataset,
    DatasetLoader,
    LlamaFactory,
    LlamaModel,
)
from _judge_common import (
    DECOMPOSE_PROMPT,
    compute_balanced_accuracy,
    parse_response,
    parse_subclaims,
)

logger = logging.getLogger(__name__)

VERIFY_PROMPT = """You are a fact-checking assistant. Decide if the CLAIM is fully supported by the CONTEXT.

CONTEXT:
{premise}

CLAIM:
{hypothesis}

Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."""


class HiSSSampleTrace(TypedDict):
    """Trace payload for one early HiSS sample."""

    claim: str
    n_sub: int
    sub_verdicts: list[int]
    pred: int
    label: int


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


def evaluate_dataset(
    dataset: AggreFactDataset,
    llm: LlamaModel,
    *,
    max_decompose_tokens: int,
    max_verify_tokens: int,
    log_every: int,
) -> tuple[
    list[int],
    list[int],
    list[str],
    list[int],
    list[float],
    list[HiSSSampleTrace],
    int,
    float,
]:
    """Run decomposition and subclaim verification across a dataset."""
    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError("dataset is empty")

    preds: list[int] = []
    labels: list[int] = []
    datasets_per_sample: list[str] = []
    subclaim_counts: list[int] = []
    latencies: list[float] = []
    traces: list[HiSSSampleTrace] = []
    unknown = 0
    started_at = time.time()

    for index, sample in enumerate(dataset):
        premise = _row_text(sample, "doc", "document")
        hypothesis = _row_text(sample, "claim", "hypothesis")
        label = _row_label(sample)
        dataset_name = _row_dataset(sample)

        sample_started_at = time.time()
        try:
            subclaims = _decompose_claim(
                llm,
                hypothesis=hypothesis,
                max_tokens=max_decompose_tokens,
            )
            sub_verdicts = [
                _verify_subclaim(
                    llm,
                    premise=premise,
                    subclaim=subclaim,
                    max_tokens=max_verify_tokens,
                )
                for subclaim in subclaims
            ]
            pred = _aggregate_subverdicts(sub_verdicts)
        except Exception as exc:  # noqa: BLE001 - benchmark records per-row failures.
            logger.warning("Sample %d failed: %s", index, exc)
            subclaims = []
            sub_verdicts = []
            pred = -1

        latencies.append(time.time() - sample_started_at)
        if pred < 0:
            unknown += 1
        preds.append(pred)
        labels.append(label)
        datasets_per_sample.append(dataset_name)
        subclaim_counts.append(len(subclaims))

        if index < 10:
            traces.append(
                {
                    "claim": hypothesis[:120],
                    "n_sub": len(subclaims),
                    "sub_verdicts": sub_verdicts,
                    "pred": pred,
                    "label": label,
                }
            )

        if (index + 1) % log_every == 0:
            elapsed = time.time() - started_at
            mean_subclaims = sum(subclaim_counts) / len(subclaim_counts)
            eta = (total_samples - index - 1) * elapsed / (index + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f unk=%d mean_sub=%.2f %.0fms/sample ETA=%.1fmin",
                index + 1,
                total_samples,
                compute_balanced_accuracy(preds, labels),
                unknown,
                mean_subclaims,
                1000 * elapsed / (index + 1),
                eta,
            )

    return (
        preds,
        labels,
        datasets_per_sample,
        subclaim_counts,
        latencies,
        traces,
        unknown,
        started_at,
    )


def _decompose_claim(llm: LlamaModel, *, hypothesis: str, max_tokens: int) -> list[str]:
    """Decompose a claim through the llama-cpp chat-completion surface."""
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
    return parse_subclaims(_chat_content(output), hypothesis)


def _verify_subclaim(
    llm: LlamaModel,
    *,
    premise: str,
    subclaim: str,
    max_tokens: int,
) -> int:
    """Verify one subclaim against the source context."""
    output = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": VERIFY_PROMPT.format(premise=premise, hypothesis=subclaim),
            },
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return parse_response(_chat_content(output))


def _aggregate_subverdicts(sub_verdicts: Sequence[int]) -> int:
    """Aggregate subclaim verdicts into one HiSS sample prediction."""
    if any(verdict < 0 for verdict in sub_verdicts):
        return -1
    return 1 if all(verdict == 1 for verdict in sub_verdicts) else 0


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
