# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Gemma AggreFact evaluator core
"""Backend loading and evaluation logic for the Gemma AggreFact evaluator."""

from __future__ import annotations

import importlib
import logging
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

from _gemma_aggrefact_eval_schema import (
    JUDGE_PROMPT,
    AggreFactDataset,
    BitsAndBytesFactory,
    DatasetLoader,
    DatasetMetric,
    EvalReport,
    JudgeBackend,
    LlamaFactory,
    ModelFactory,
    TokenizerFactory,
    TorchModule,
)
from _judge_common import (
    compute_balanced_accuracy,
    parse_response,
)

logger = logging.getLogger(__name__)


class LlamaCppBackend:
    """Local llama-cpp-python backend."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 2):
        """Load a llama-cpp model from ``model_path``."""
        llama_module = importlib.import_module("llama_cpp")
        llama_factory = cast(LlamaFactory, vars(llama_module)["Llama"])

        logger.info("Loading llama-cpp model: %s", model_path)
        self.llm = llama_factory(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            verbose=False,
            logits_all=False,
        )
        logger.info("Loaded")

    def judge(self, premise: str, hypothesis: str) -> str:
        """Judge one premise/hypothesis pair through llama-cpp."""
        prompt = JUDGE_PROMPT.format(premise=premise, hypothesis=hypothesis)
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
        )
        return _extract_chat_content(out)


class TransformersBackend:
    """Hugging Face transformers backend for cloud GPU runs."""

    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        quantize: str | None = None,
    ) -> None:
        """Load a transformer model by identifier."""
        torch_module = cast(TorchModule, importlib.import_module("torch"))
        transformers_module = importlib.import_module("transformers")
        tokenizer_factory = cast(
            TokenizerFactory,
            vars(transformers_module)["AutoTokenizer"],
        )
        model_factory = cast(
            ModelFactory,
            vars(transformers_module)["AutoModelForCausalLM"],
        )

        logger.info(
            "Loading transformers model: %s (dtype=%s, quant=%s)",
            model_id,
            dtype,
            quantize,
        )
        self.tokenizer = tokenizer_factory.from_pretrained(model_id)

        kwargs: dict[str, object] = {"device_map": device}
        if quantize == "4bit":
            bits_and_bytes = cast(
                BitsAndBytesFactory,
                vars(transformers_module)["BitsAndBytesConfig"],
            )
            kwargs["quantization_config"] = bits_and_bytes(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_module.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = getattr(torch_module, dtype)

        self.model = model_factory.from_pretrained(model_id, **kwargs)
        self.model.eval()
        logger.info("Loaded on %s", self.model.device)
        self._torch = torch_module

    def judge(self, premise: str, hypothesis: str) -> str:
        """Judge one premise/hypothesis pair through transformers."""
        prompt = JUDGE_PROMPT.format(premise=premise, hypothesis=hypothesis)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
        ).to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0][inputs.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if isinstance(text, list):
            return " ".join(text)
        return text


def _extract_chat_content(payload: Mapping[str, object]) -> str:
    """Extract a chat-completion message from a llama-cpp response."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("llama-cpp response missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("llama-cpp choice must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("llama-cpp choice missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("llama-cpp message content must be a string")
    return content


def _row_text(row: Mapping[str, object], primary: str, fallback: str) -> str:
    """Read a text field from an AggreFact row."""
    value = row.get(primary, row.get(fallback, ""))
    return value if isinstance(value, str) else str(value)


def _row_label(row: Mapping[str, object]) -> int:
    """Read a binary label from an AggreFact row."""
    value = row.get("label", row.get("annotations", 0))
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
    """Read a dataset name from an AggreFact row."""
    value = row.get("dataset", "unknown")
    return value if isinstance(value, str) else str(value)


def load_aggrefact(max_samples: int | None = None) -> AggreFactDataset:
    """Load the gated LLM-AggreFact test split."""
    datasets_module = importlib.import_module("datasets")
    load_dataset = cast(DatasetLoader, vars(datasets_module)["load_dataset"])

    logger.info("Loading LLM-AggreFact...")
    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    logger.info("Loaded %d samples", len(ds))
    return ds


def build_backend(
    backend_name: str,
    *,
    model: str,
    n_ctx: int,
    n_threads: int,
    dtype: str,
    quantize: str | None,
) -> JudgeBackend:
    """Build the selected evaluator backend."""
    if backend_name == "llama-cpp":
        return LlamaCppBackend(model, n_ctx=n_ctx, n_threads=n_threads)
    return TransformersBackend(model, dtype=dtype, device="cuda", quantize=quantize)


def evaluate_dataset(
    dataset: AggreFactDataset,
    backend: JudgeBackend,
    *,
    model: str,
    backend_name: str,
    log_every: int,
) -> EvalReport:
    """Evaluate an AggreFact dataset with a judge backend."""
    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    preds: list[int] = []
    labels: list[int] = []
    datasets: list[str] = []
    latencies: list[float] = []
    unknown_count = 0
    t_start = time.time()

    for index, sample in enumerate(dataset):
        premise = _row_text(sample, "doc", "document")
        hypothesis = _row_text(sample, "claim", "hypothesis")
        label = _row_label(sample)
        dataset_name = _row_dataset(sample)

        t0 = time.time()
        try:
            response = backend.judge(premise, hypothesis)
        except Exception as exc:  # noqa: BLE001 - benchmark must record per-sample failures.
            logger.warning("Sample %d failed: %s", index, exc)
            response = "ERROR"
        elapsed = time.time() - t0

        pred = parse_response(response)
        if pred < 0:
            unknown_count += 1

        preds.append(pred)
        labels.append(label)
        datasets.append(dataset_name)
        latencies.append(elapsed)

        if (index + 1) % log_every == 0:
            elapsed_total = time.time() - t_start
            current_ba = compute_balanced_accuracy(preds, labels)
            eta_s = (len(dataset) - index - 1) * (elapsed_total / (index + 1))
            logger.info(
                "[%d/%d] BA=%.4f unk=%d %.0fms/sample ETA=%.1fmin",
                index + 1,
                len(dataset),
                current_ba,
                unknown_count,
                1000 * elapsed_total / (index + 1),
                eta_s / 60,
            )

    per_dataset = _per_dataset_metrics(preds, labels, datasets)
    total_time = time.time() - t_start
    sorted_latencies = sorted(latencies)
    return {
        "model": model,
        "backend": backend_name,
        "samples": len(dataset),
        "global_balanced_accuracy": compute_balanced_accuracy(preds, labels),
        "per_dataset": per_dataset,
        "unknown_predictions": unknown_count,
        "total_time_seconds": total_time,
        "mean_latency_ms": 1000 * sum(latencies) / len(latencies),
        "p50_latency_ms": 1000 * sorted_latencies[len(sorted_latencies) // 2],
        "p99_latency_ms": 1000
        * sorted_latencies[
            min(int(len(sorted_latencies) * 0.99), len(sorted_latencies) - 1)
        ],
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets,
    }


def _per_dataset_metrics(
    preds: Sequence[int],
    labels: Sequence[int],
    datasets: Sequence[str],
) -> dict[str, DatasetMetric]:
    """Compute balanced accuracy for each dataset."""
    grouped: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for pred, label, dataset in zip(preds, labels, datasets, strict=True):
        grouped[dataset][0].append(pred)
        grouped[dataset][1].append(label)
    return {
        dataset: {
            "samples": len(dataset_labels),
            "balanced_accuracy": compute_balanced_accuracy(
                dataset_preds,
                dataset_labels,
            ),
        }
        for dataset, (dataset_preds, dataset_labels) in grouped.items()
    }
