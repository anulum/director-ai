# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Gemma AggreFact evaluator schema
"""Typed schemas and protocol boundaries for the Gemma AggreFact evaluator."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol, TypedDict

JUDGE_PROMPT = """You are a fact-checking assistant. Decide if the CLAIM is fully supported by the CONTEXT.

CONTEXT:
{premise}

CLAIM:
{hypothesis}

Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."""


class DatasetMetric(TypedDict):
    """Per-dataset metric payload."""

    samples: int
    balanced_accuracy: float


class EvalReport(TypedDict):
    """Gemma AggreFact evaluator report schema."""

    model: str
    backend: str
    samples: int
    global_balanced_accuracy: float
    per_dataset: dict[str, DatasetMetric]
    unknown_predictions: int
    total_time_seconds: float
    mean_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    predictions: list[int]
    labels: list[int]
    datasets_per_sample: list[str]


class AggreFactDataset(Protocol):
    """Subset of the Hugging Face Dataset API used by this evaluator."""

    def select(self, indices: range) -> AggreFactDataset:
        """Return a selected subset of rows."""

    def __len__(self) -> int:
        """Return the number of rows."""

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        """Iterate over AggreFact rows."""


class JudgeBackend(Protocol):
    """Backend protocol shared by llama-cpp and transformers judges."""

    def judge(self, premise: str, hypothesis: str) -> str:
        """Return a textual supported/not-supported verdict."""


class LlamaModel(Protocol):
    """Protocol for the llama-cpp chat-completion surface."""

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Mapping[str, object]:
        """Create a chat completion for one prompt."""


class LlamaFactory(Protocol):
    """Factory protocol for constructing a llama-cpp model."""

    def __call__(self, **kwargs: object) -> LlamaModel:
        """Construct a llama-cpp model."""


class Tokenizer(Protocol):
    """Protocol for the transformer tokenizer surface used here."""

    eos_token_id: int | None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        return_tensors: str,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> TensorLike:
        """Tokenize a chat prompt for generation."""

    def decode(self, tokens: object, *, skip_special_tokens: bool) -> str | list[str]:
        """Decode generated token IDs."""


class TokenizerFactory(Protocol):
    """Factory protocol for loading a tokenizer."""

    def from_pretrained(self, model_id: str) -> Tokenizer:
        """Load a tokenizer by model identifier."""


class Model(Protocol):
    """Protocol for the transformer causal language model surface."""

    device: object

    def eval(self) -> object:
        """Switch the model to evaluation mode."""

    def generate(
        self,
        inputs: object,
        *,
        max_new_tokens: int,
        do_sample: bool,
        pad_token_id: int | None,
    ) -> Sequence[TokenSequence]:
        """Generate token IDs for one prompt."""


class ModelFactory(Protocol):
    """Factory protocol for loading a causal language model."""

    def from_pretrained(self, model_id: str, **kwargs: object) -> Model:
        """Load a causal language model by model identifier."""


class BitsAndBytesFactory(Protocol):
    """Factory protocol for optional 4-bit quantization config."""

    def __call__(self, **kwargs: object) -> object:
        """Build a quantization config object."""


class TensorLike(Protocol):
    """Small tensor protocol for tokenizer outputs."""

    shape: Sequence[int]

    def to(self, device: object) -> TensorLike:
        """Move the tensor batch to a device."""


class TokenSequence(Protocol):
    """Generated-token sequence supporting slicing."""

    def __getitem__(self, index: object) -> object:
        """Return one generated token item or slice."""


class NoGrad(AbstractContextManager[None]):
    """Typed context manager for ``torch.no_grad``."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the context without suppressing exceptions."""


class TorchModule(Protocol):
    """Subset of torch used by the transformers path."""

    bfloat16: object

    def no_grad(self) -> NoGrad:
        """Return a no-gradient context manager."""


class DatasetLoader(Protocol):
    """Factory protocol for loading an AggreFact dataset."""

    def __call__(self, name: str, *, split: str) -> AggreFactDataset:
        """Load a dataset split."""
