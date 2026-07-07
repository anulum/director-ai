# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact self-consistency evaluator core
"""Core evaluation logic for routed Gemma AggreFact self-consistency judging."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Mapping
from typing import Protocol, cast

from _gemma_aggrefact_eval_schema import AggreFactDataset, DatasetLoader
from _judge_common import DATASET_TO_FAMILY, PROMPTS, compute_balanced_accuracy
from _judge_common import parse_response as parse_verdict

logger = logging.getLogger(__name__)


class SelfConsistencyLlamaModel(Protocol):
    """Protocol for the llama-cpp chat-completion surface used by voting."""

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Mapping[str, object]:
        """Create one sampled chat completion for a routed prompt."""


class SelfConsistencyLlamaFactory(Protocol):
    """Factory protocol for constructing a llama-cpp self-consistency model."""

    def __call__(self, **kwargs: object) -> SelfConsistencyLlamaModel:
        """Construct a llama-cpp model."""


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


def build_llama(
    model_path: str,
    *,
    n_ctx: int,
    n_threads: int,
) -> SelfConsistencyLlamaModel:
    """Construct the llama-cpp model through the public factory surface."""
    llama_module = importlib.import_module("llama_cpp")
    llama_factory = cast(SelfConsistencyLlamaFactory, vars(llama_module)["Llama"])
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
    """Return routed-family sample counts for an AggreFact dataset."""
    counts: dict[str, int] = {}
    for sample in dataset:
        family = dataset_family(_row_dataset(sample))
        counts[family] = counts.get(family, 0) + 1
    return counts


def evaluate_dataset(
    dataset: AggreFactDataset,
    llm: SelfConsistencyLlamaModel,
    *,
    k: int,
    temperature: float,
    top_p: float,
    log_every: int,
) -> tuple[
    list[int],
    list[float | None],
    list[int],
    list[str],
    list[str],
    list[float],
    int,
    float,
]:
    """Run routed self-consistency voting across an AggreFact dataset."""
    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError("dataset is empty")

    preds: list[int] = []
    support_fractions: list[float | None] = []
    labels: list[int] = []
    datasets_per_sample: list[str] = []
    families_per_sample: list[str] = []
    latencies: list[float] = []
    unknown = 0
    started_at = time.time()

    for index, sample in enumerate(dataset):
        premise = _row_text(sample, "doc", "document")
        hypothesis = _row_text(sample, "claim", "hypothesis")
        label = _row_label(sample)
        dataset_name = _row_dataset(sample)
        family = dataset_family(dataset_name)
        prompt = PROMPTS.get(family, PROMPTS["claim"]).format(
            premise=premise,
            hypothesis=hypothesis,
        )

        sample_started_at = time.time()
        support_fraction = vote_support_fraction(
            llm,
            prompt=prompt,
            k=k,
            temperature=temperature,
            top_p=top_p,
            sample_index=index,
        )
        latencies.append(time.time() - sample_started_at)

        if support_fraction is None:
            pred = -1
            unknown += 1
        else:
            pred = 1 if support_fraction >= 0.5 else 0

        preds.append(pred)
        support_fractions.append(support_fraction)
        labels.append(label)
        datasets_per_sample.append(dataset_name)
        families_per_sample.append(family)

        if (index + 1) % log_every == 0:
            elapsed = time.time() - started_at
            eta = (total_samples - index - 1) * elapsed / (index + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f unk=%d %.0fms/sample ETA=%.1fmin",
                index + 1,
                total_samples,
                compute_balanced_accuracy(preds, labels),
                unknown,
                1000 * elapsed / (index + 1),
                eta,
            )

    return (
        preds,
        support_fractions,
        labels,
        datasets_per_sample,
        families_per_sample,
        latencies,
        unknown,
        started_at,
    )


def vote_support_fraction(
    llm: SelfConsistencyLlamaModel,
    *,
    prompt: str,
    k: int,
    temperature: float,
    top_p: float,
    sample_index: int,
) -> float | None:
    """Return the support vote fraction for one routed prompt."""
    votes_supported = 0
    votes_not_supported = 0
    for vote_index in range(k):
        try:
            output = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=temperature,
                top_p=top_p,
            )
            text = _chat_content(output)
        except (KeyError, RuntimeError, TypeError, ValueError, OSError) as exc:
            logger.warning("Sample %d k=%d failed: %s", sample_index, vote_index, exc)
            text = "ERROR"

        verdict = parse_verdict(text)
        if verdict == 1:
            votes_supported += 1
        elif verdict == 0:
            votes_not_supported += 1

    known_votes = votes_supported + votes_not_supported
    if known_votes == 0:
        return None
    return votes_supported / known_votes


def dataset_family(dataset_name: str) -> str:
    """Return the routed prompt family for a dataset name."""
    return DATASET_TO_FAMILY.get(dataset_name, "claim")


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
        label = int(value)
    elif isinstance(value, int):
        label = value
    elif (isinstance(value, float) and value.is_integer()) or isinstance(value, str):
        label = int(value)
    else:
        raise ValueError(f"invalid label value: {value!r}")
    if label not in {0, 1}:
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    return label


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
