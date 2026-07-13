# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — claim-decomposition detection benchmark (WCA-2 part 2)
"""Downstream detection F1: LLM atomic decomposition vs regex sentence split.

WCA-2 part 2 gate (``docs/internal/BACKLOG.md``): *"claim-coverage F1 vs
regex baseline needs a live provider or local checkpoint; without a
measurement the default does not flip."* This harness supplies the
measurement.

The experiment scores three decomposition strategies on the WICE subset of
``lytang/LLM-AggreFact`` (short, single-passage claims — the regime where
compound-sentence decomposition can plausibly help):

1. ``no-decomp`` — NLI(doc, claim) directly.
2. ``regex-decomp`` — split the claim with the production sentence splitter
   (:func:`director_ai.core.text_segmentation.split_sentences`), NLI each
   sub-claim, aggregate with ``min`` (weakest-link support).
3. ``llm-decomp`` — :class:`AtomicClaimDecomposer` with a **local** instruct
   model as its injected transport, NLI each atomic claim, aggregate with
   ``min``.

For each strategy a support score in ``[0, 1]`` is turned into a binary
label (``supported`` iff ``score >= threshold``) and compared to the gold
``label``. We report metrics at the fixed 0.5 threshold **and** at the
balanced-accuracy-maximising threshold (oracle sweep), so the read is not an
artefact of one cut point. The gate answer is the ``llm-decomp − regex-decomp``
delta in hallucination F1 and balanced accuracy.

The scoring and metric logic is pure and takes an injected ``predictor`` and
``decomposer`` so it can be exercised offline with fakes; the heavy models are
constructed only in :func:`run_claim_decomp_benchmark` / :func:`main`.

Usage::

    export HF_TOKEN=hf_...          # gated LLM-AggreFact
    python benchmarks/run_claim_decomp_benchmark.py \
        --samples 300 \
        --decomposer-model Qwen/Qwen2.5-7B-Instruct \
        --out benchmarks/results/claim_decomp_wice.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("DirectorAI")

#: Decomposition strategies, in report order.
CONFIGS: tuple[str, ...] = ("no-decomp", "regex-decomp", "llm-decomp")

#: Candidate thresholds for the oracle balanced-accuracy sweep.
_SWEEP_THRESHOLDS: tuple[float, ...] = tuple(i / 100 for i in range(5, 100, 5))

#: Default local instruct model for the decomposer transport.
DEFAULT_DECOMPOSER_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class Predictor(Protocol):
    """Minimal NLI interface: entailment probability of a (premise, hyp)."""

    def score(self, premise: str, hypothesis: str) -> float: ...


class Decomposer(Protocol):
    """Minimal decomposer interface matching AtomicClaimDecomposer."""

    def decompose(
        self, text: str, *, sentence_splitter: Callable[[str], list[str]]
    ) -> Any: ...


# ── Scoring ──────────────────────────────────────────────────────────────


def _subclaims(
    strategy: str,
    claim: str,
    *,
    decomposer: Decomposer | None,
    splitter: Callable[[str], list[str]],
) -> tuple[list[str], str | None]:
    """Return ``(sub-claims, backend)`` a strategy checks for one claim.

    ``no-decomp`` keeps the whole claim; ``regex-decomp`` uses the
    production sentence splitter; ``llm-decomp`` uses the decomposer
    (which itself falls back to *splitter* on provider failure, labelling
    the result). Every branch guarantees a non-empty list — an empty split
    degrades to the original claim so the claim is always scored. The
    backend label is set only for ``llm-decomp``.
    """
    if strategy == "no-decomp":
        return [claim], None
    if strategy == "regex-decomp":
        subs = [s for s in splitter(claim) if s.strip()]
        return subs or [claim], None
    if strategy == "llm-decomp":
        if decomposer is None:
            raise ValueError("llm-decomp requires a decomposer")
        result = decomposer.decompose(claim, sentence_splitter=splitter)
        subs = [s for s in result.claims if s.strip()]
        return subs or [claim], getattr(result, "backend", "unknown")
    raise ValueError(f"unknown strategy {strategy!r}")


def _support_score(
    doc: str,
    subclaims: Sequence[str],
    predictor: Predictor,
) -> float:
    """Weakest-link support: min entailment prob across sub-claims.

    A claim is supported only if *every* atomic part is supported, so the
    claim score is the minimum sub-claim entailment probability.
    """
    return min(predictor.score(doc, sub) for sub in subclaims)


@dataclass
class ConfigScores:
    """Per-sample (gold label, support score) for one strategy."""

    strategy: str
    labels: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    #: Number of sub-claims produced per sample (decomposition granularity).
    n_subclaims: list[int] = field(default_factory=list)
    #: How many llm-decomp samples used the LLM vs the sentence fallback.
    backends: dict[str, int] = field(default_factory=dict)


def score_configs(
    rows: Sequence[dict[str, Any]],
    predictor: Predictor,
    *,
    decomposer: Decomposer | None,
    splitter: Callable[[str], list[str]],
    strategies: Sequence[str] = CONFIGS,
) -> dict[str, ConfigScores]:
    """Score every row under each strategy; return per-strategy scores.

    Rows missing ``doc``/``claim``/``label`` are skipped uniformly across
    strategies so the comparison stays paired.
    """
    out = {s: ConfigScores(strategy=s) for s in strategies}
    for row in rows:
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        if label is None or not doc or not claim:
            continue
        for strat in strategies:
            subs, backend = _subclaims(
                strat, claim, decomposer=decomposer, splitter=splitter
            )
            score = _support_score(doc, subs, predictor)
            cs = out[strat]
            cs.labels.append(int(label))
            cs.scores.append(score)
            cs.n_subclaims.append(len(subs))
            if backend is not None:
                cs.backends[backend] = cs.backends.get(backend, 0) + 1
    return out


# ── Metrics ──────────────────────────────────────────────────────────────


def _balanced_accuracy(labels: Sequence[int], preds: Sequence[int]) -> float:
    pos = neg = tp = tn = 0
    for lab, pred in zip(labels, preds, strict=True):
        if lab == 1:
            pos += 1
            tp += pred == 1
        else:
            neg += 1
            tn += pred == 0
    if pos == 0 or neg == 0:
        return 0.0
    return (tp / pos + tn / neg) / 2


def _prf1(
    labels: Sequence[int], preds: Sequence[int], target: int
) -> tuple[float, float, float]:
    """Precision/recall/F1 for *target* class."""
    tp = sum(
        1 for lab, p in zip(labels, preds, strict=True) if p == target and lab == target
    )
    fp = sum(
        1 for lab, p in zip(labels, preds, strict=True) if p == target and lab != target
    )
    fn = sum(
        1 for lab, p in zip(labels, preds, strict=True) if p != target and lab == target
    )
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def metrics_at_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, float]:
    """Full metric block at one threshold (supported iff score >= t)."""
    preds = [1 if s >= threshold else 0 for s in scores]
    hp, hr, hf1 = _prf1(labels, preds, 0)  # hallucination = not supported
    sp, sr, sf1 = _prf1(labels, preds, 1)  # supported
    return {
        "threshold": float(threshold),
        "balanced_accuracy": _balanced_accuracy(labels, preds),
        "hallucination_precision": hp,
        "hallucination_recall": hr,
        "hallucination_f1": hf1,
        "supported_precision": sp,
        "supported_recall": sr,
        "supported_f1": sf1,
    }


def best_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    candidates: Sequence[float] = _SWEEP_THRESHOLDS,
) -> dict[str, float]:
    """Metrics at the balanced-accuracy-maximising candidate threshold."""
    best = max(
        (metrics_at_threshold(labels, scores, t) for t in candidates),
        key=lambda m: m["balanced_accuracy"],
    )
    return best


def summarise(scores: dict[str, ConfigScores]) -> dict[str, Any]:
    """Build the report dict: per-config metrics + llm−regex deltas."""
    per_config: dict[str, Any] = {}
    for strat, cs in scores.items():
        n = len(cs.labels)
        avg_subs = sum(cs.n_subclaims) / n if n else 0.0
        per_config[strat] = {
            "samples": n,
            "positives": sum(cs.labels),
            "avg_subclaims": avg_subs,
            "backends": cs.backends,
            "fixed_0.5": metrics_at_threshold(cs.labels, cs.scores, 0.5),
            "oracle": best_threshold(cs.labels, cs.scores),
        }
    report: dict[str, Any] = {"per_config": per_config}
    if "llm-decomp" in per_config and "regex-decomp" in per_config:
        llm, regex = per_config["llm-decomp"], per_config["regex-decomp"]
        report["delta_llm_minus_regex"] = {
            "hallucination_f1_fixed": llm["fixed_0.5"]["hallucination_f1"]
            - regex["fixed_0.5"]["hallucination_f1"],
            "balanced_accuracy_fixed": llm["fixed_0.5"]["balanced_accuracy"]
            - regex["fixed_0.5"]["balanced_accuracy"],
            "hallucination_f1_oracle": llm["oracle"]["hallucination_f1"]
            - regex["oracle"]["hallucination_f1"],
            "balanced_accuracy_oracle": llm["oracle"]["balanced_accuracy"]
            - regex["oracle"]["balanced_accuracy"],
        }
    return report


# ── Local instruct-model transport ───────────────────────────────────────


def build_local_transport(
    model_name: str,
    *,
    max_new_tokens: int = 512,
) -> Callable[[str, list[dict[str, str]], int], str | None]:
    """Load an instruct model and return an AtomicClaimDecomposer transport.

    The returned callable applies the model's chat template to the
    ``messages`` and greedily decodes a JSON reply. On any generation error
    it returns ``None`` so the decomposer degrades to its labelled sentence
    fallback (never a fabricated claim list).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading decomposer instruct model: %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    def transport(
        model_arg: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> str | None:
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, max_new_tokens),
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            gen = out[0][inputs["input_ids"].shape[1] :]
            return tok.decode(gen, skip_special_tokens=True)
        except Exception as exc:  # pragma: no cover - GPU-only path
            logger.warning("local decomposer transport failed: %s", exc)
            return None

    return transport


# ── Orchestration ────────────────────────────────────────────────────────


def _load_wice(max_samples: int | None) -> list[dict[str, Any]]:
    """Load the WICE subset of the gated LLM-AggreFact test split."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    logger.info("Loading LLM-AggreFact (gated); filtering to WICE...")
    ds = load_dataset("lytang/LLM-AggreFact", split="test", token=token)
    rows = [dict(r) for r in ds if r.get("dataset") == "Wice"]
    if max_samples:
        rows = rows[:max_samples]
    logger.info("Loaded %d WICE rows", len(rows))
    return rows


def run_claim_decomp_benchmark(
    *,
    max_samples: int | None = None,
    decomposer_model: str = DEFAULT_DECOMPOSER_MODEL,
    nli_model: str | None = None,
    rows: Sequence[dict[str, Any]] | None = None,
    predictor: Predictor | None = None,
    decomposer: Decomposer | None = None,
    splitter: Callable[[str], list[str]] | None = None,
) -> dict[str, Any]:
    """Run the three-strategy WICE detection benchmark and return the report.

    Injectable ``rows`` / ``predictor`` / ``decomposer`` / ``splitter`` keep
    the function testable offline; unset arguments are built from the real
    dataset and models.
    """
    from director_ai.core.text_segmentation import split_sentences

    if splitter is None:
        splitter = split_sentences
    if rows is None:
        rows = _load_wice(max_samples)
    if predictor is None:
        from benchmarks.aggrefact_eval import _BinaryNLIPredictor

        predictor = _BinaryNLIPredictor(model_name=nli_model)
    if decomposer is None:
        from director_ai.core.scoring.claim_decomposition import (
            AtomicClaimDecomposer,
        )

        transport = build_local_transport(decomposer_model)
        decomposer = AtomicClaimDecomposer(
            provider="openai",  # label only; transport is the local model
            model=decomposer_model,
            transport=transport,
        )

    t0 = time.perf_counter()
    scores = score_configs(rows, predictor, decomposer=decomposer, splitter=splitter)
    report = summarise(scores)
    report["meta"] = {
        "samples": len(rows),
        "decomposer_model": decomposer_model,
        "nli_model": nli_model or os.environ.get("DIRECTOR_NLI_MODEL", "default"),
        "elapsed_s": time.perf_counter() - t0,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--decomposer-model", default=DEFAULT_DECOMPOSER_MODEL)
    parser.add_argument("--nli-model", default=None)
    parser.add_argument(
        "--out",
        default="benchmarks/results/claim_decomp_wice.json",
    )
    args = parser.parse_args(argv)

    report = run_claim_decomp_benchmark(
        max_samples=args.samples,
        decomposer_model=args.decomposer_model,
        nli_model=args.nli_model,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Saved report to %s", out)

    delta = report.get("delta_llm_minus_regex", {})
    print("\n=== claim-decomp WICE detection (llm − regex) ===")
    for k, v in delta.items():
        print(f"  {k}: {v:+.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
