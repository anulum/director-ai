# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Repaired HiSS with per-task-family verify
"""HiSS (Hierarchical Step-by-Step) hallucination detection — repaired.

The original ``gemma_aggrefact_hiss.py`` lost to baseline (0.7872 vs
0.7934) for three reasons:

1. The verify step used a single generic prompt; the routed prompts
   that win for K=1 were not used.
2. Aggregation was strict AND ("every sub-claim must be SUPPORTED");
   for claims with 4-5 sub-claims this systematically biases toward
   NOT_SUPPORTED.
3. Decomposition fired on every claim, including short ones where
   sub-claim splitting is meaningless and adds noise.

This repaired variant fixes all three:

1. **Routed verify** — the verify pass uses ``PROMPT_SUMM`` /
   ``PROMPT_RAG`` / ``PROMPT_CLAIM`` based on the AggreFact subset's
   task family, exactly as the 82.11 % routed champion does.
2. **Soft aggregation** — the verdict is SUPPORTED if at least
   ``--support-frac 0.75`` of the sub-claims verify positive (default
   0.75; sweep-able). The strict-AND behaviour of the original is
   recovered with ``--support-frac 1.0``.
3. **Length gate** — claims under ``--min-decompose-words 12`` words
   skip decomposition entirely and fall back to the routed K=1 path,
   which is the same as the champion. This preserves the champion's
   performance on short claims while still letting decomposition help
   on long, multi-fact ones.

Soft score: ``support_fraction = supported_subclaims / total_subclaims``,
which is threshold-able and ensemble-friendly.

Usage::

    GGML_VK_VISIBLE_DEVICES=5 python benchmarks/gemma_aggrefact_hiss_routed.py \\
        --model /tmp/gemma-models/google_gemma-4-E4B-it-Q6_K.gguf \\
        --max-samples 29320 \\
        --output benchmarks/results/gemma_e4b_q6_hiss_routed.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

from _judge_common import (
    DATASET_TO_FAMILY,
    PROMPTS,
    compute_balanced_accuracy,
)
from _judge_common import (
    parse_response as parse_verdict,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ── Decomposition prompt (single, generic — only the verify step is routed) ──

DECOMPOSE_PROMPT = """Break the CLAIM into 1-4 atomic sub-claims that can be checked independently.
Return them as a numbered list, one sub-claim per line. Do not add explanation. Do not repeat the original claim.

CLAIM:
{claim}

Sub-claims:
1."""

LIST_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$")


def parse_subclaims(raw: str, original_claim: str, max_n: int = 4) -> list[str]:
    """Extract atomic sub-claims from the decomposition response."""
    text = (
        "1. " + raw
        if not raw.lstrip().startswith(("1", "-", "*"))
        else raw
    )
    out: list[str] = []
    for line in text.splitlines():
        m = LIST_LINE_RE.match(line)
        if m:
            sub = m.group(1).strip()
            if sub and sub.lower() not in {"sub-claims", "sub-claim", "claim"}:
                out.append(sub)
        if len(out) >= max_n:
            break
    if not out:
        out = [original_claim]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/gemma_hiss_routed.json",
    )
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument(
        "--min-decompose-words",
        type=int,
        default=12,
        help=(
            "Claims shorter than this skip decomposition and use the "
            "routed K=1 path"
        ),
    )
    p.add_argument(
        "--support-frac",
        type=float,
        default=0.75,
        help=(
            "Verdict is SUPPORTED if support_fraction >= this value "
            "(default 0.75 = at least 75%% of sub-claims must verify)"
        ),
    )
    p.add_argument(
        "--max-subclaims",
        type=int,
        default=4,
        help="Cap on sub-claims per decomposition (default 4)",
    )
    args = p.parse_args()

    from datasets import load_dataset
    from llama_cpp import Llama

    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    logger.info(
        "Samples: %d  min_decompose_words=%d  support_frac=%.2f  max_sub=%d",
        len(ds),
        args.min_decompose_words,
        args.support_frac,
        args.max_subclaims,
    )

    family_counts: dict[str, int] = defaultdict(int)
    for sample in ds:
        family_counts[
            DATASET_TO_FAMILY.get(sample["dataset"], "claim")
        ] += 1
    logger.info("Family distribution: %s", dict(family_counts))

    logger.info("Loading: %s", args.model)
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=512,
        verbose=False,
        logits_all=False,
    )
    logger.info("Loaded")

    preds: list[int] = []
    support_fractions: list[float | None] = []
    labels: list[int] = []
    datasets_list: list[str] = []
    families: list[str] = []
    subclaim_counts: list[int] = []
    decomposed_flags: list[bool] = []
    latencies: list[float] = []
    unknown = 0
    skipped_decompose = 0
    t_start = time.time()

    for i, sample in enumerate(ds):
        premise = sample["doc"]
        hypothesis = sample["claim"]
        label = int(sample["label"])
        dataset_name = sample["dataset"]
        family = DATASET_TO_FAMILY.get(dataset_name, "claim")
        verify_template = PROMPTS[family]

        t0 = time.time()
        word_count = len(hypothesis.split())
        if word_count < args.min_decompose_words:
            # Length gate — fall back to the K=1 routed path
            skipped_decompose += 1
            decomposed = False
            try:
                out = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": verify_template.format(
                                premise=premise, hypothesis=hypothesis
                            ),
                        }
                    ],
                    max_tokens=8,
                    temperature=0.0,
                )
                text = out["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sample %d (short) failed: %s", i, exc)
                text = "ERROR"
            v = parse_verdict(text)
            if v < 0:
                pred = -1
                support_fraction: float | None = None
                unknown += 1
            else:
                pred = v
                support_fraction = float(v)
            n_subclaims = 1
        else:
            # Decompose then routed-verify
            decomposed = True
            try:
                out = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": DECOMPOSE_PROMPT.format(
                                claim=hypothesis
                            ),
                        }
                    ],
                    max_tokens=160,
                    temperature=0.0,
                )
                decomp_raw = out["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sample %d decompose failed: %s", i, exc)
                decomp_raw = ""
            subclaims = parse_subclaims(
                decomp_raw, hypothesis, max_n=args.max_subclaims
            )
            n_subclaims = len(subclaims)

            # Verify each sub-claim with the routed prompt for the family
            sub_supported = 0
            sub_unknown = 0
            for sub in subclaims:
                try:
                    out = llm.create_chat_completion(
                        messages=[
                            {
                                "role": "user",
                                "content": verify_template.format(
                                    premise=premise, hypothesis=sub
                                ),
                            }
                        ],
                        max_tokens=8,
                        temperature=0.0,
                    )
                    text = out["choices"][0]["message"]["content"]
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Sample %d sub-verify failed: %s", i, exc
                    )
                    text = "ERROR"
                v = parse_verdict(text)
                if v == 1:
                    sub_supported += 1
                elif v < 0:
                    sub_unknown += 1
            valid = n_subclaims - sub_unknown
            if valid <= 0:
                pred = -1
                support_fraction = None
                unknown += 1
            else:
                support_fraction = sub_supported / valid
                pred = 1 if support_fraction >= args.support_frac else 0
        latencies.append(time.time() - t0)

        preds.append(pred)
        support_fractions.append(support_fraction)
        labels.append(label)
        datasets_list.append(dataset_name)
        families.append(family)
        subclaim_counts.append(n_subclaims)
        decomposed_flags.append(decomposed)

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            ba = compute_balanced_accuracy(preds, labels)
            eta_min = (len(ds) - i - 1) * elapsed / (i + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f unk=%d skip=%d %.0fms/sample ETA=%.1fmin",
                i + 1,
                len(ds),
                ba,
                unknown,
                skipped_decompose,
                1000 * elapsed / (i + 1),
                eta_min,
            )

    by_ds: dict[str, tuple[list[int], list[int]]] = defaultdict(
        lambda: ([], [])
    )
    for p_, l_, d_ in zip(preds, labels, datasets_list, strict=True):
        by_ds[d_][0].append(p_)
        by_ds[d_][1].append(l_)
    per_ds_metrics = {
        ds: {
            "samples": len(l_),
            "balanced_accuracy": compute_balanced_accuracy(p_, l_),
        }
        for ds, (p_, l_) in by_ds.items()
    }

    by_fam: dict[str, tuple[list[int], list[int]]] = defaultdict(
        lambda: ([], [])
    )
    for p_, l_, f_ in zip(preds, labels, families, strict=True):
        by_fam[f_][0].append(p_)
        by_fam[f_][1].append(l_)
    per_family_metrics = {
        f: {
            "samples": len(l_),
            "balanced_accuracy": compute_balanced_accuracy(p_, l_),
        }
        for f, (p_, l_) in by_fam.items()
    }

    total = time.time() - t_start
    results = {
        "schema_version": 2,
        "model": args.model,
        "method": (
            f"HiSS routed: decompose then per-family verify, "
            f"min_words={args.min_decompose_words} "
            f"support_frac={args.support_frac} "
            f"max_sub={args.max_subclaims}"
        ),
        "samples": len(ds),
        "min_decompose_words": args.min_decompose_words,
        "support_frac": args.support_frac,
        "max_subclaims": args.max_subclaims,
        "skipped_decompose": skipped_decompose,
        "global_balanced_accuracy": compute_balanced_accuracy(
            preds, labels
        ),
        "per_dataset": per_ds_metrics,
        "per_family": per_family_metrics,
        "dataset_to_family": DATASET_TO_FAMILY,
        "unknown_predictions": unknown,
        "total_time_seconds": total,
        "p50_latency_ms": 1000
        * sorted(latencies)[len(latencies) // 2]
        if latencies
        else 0,
        "p99_latency_ms": 1000
        * sorted(latencies)[int(len(latencies) * 0.99)]
        if latencies
        else 0,
        "predictions": preds,
        "support_fractions": support_fractions,
        "subclaim_counts": subclaim_counts,
        "decomposed_flags": decomposed_flags,
        "labels": labels,
        "datasets_per_sample": datasets_list,
        "families_per_sample": families,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    logger.info(
        "Global BA:    %.4f", results["global_balanced_accuracy"]
    )
    logger.info(
        "Decomposed:   %d (%.1f%%)",
        len(ds) - skipped_decompose,
        100 * (len(ds) - skipped_decompose) / len(ds),
    )
    logger.info(
        "Skipped:      %d (%.1f%%, used K=1 routed)",
        skipped_decompose,
        100 * skipped_decompose / len(ds),
    )
    logger.info(
        "Unknown:      %d (%.1f%%)", unknown, 100 * unknown / len(ds)
    )
    logger.info("Time:         %.1fmin", total / 60)
    logger.info("=" * 60)
    for fam, m in sorted(per_family_metrics.items()):
        logger.info(
            "  %-8s %5d  %.4f",
            fam,
            m["samples"],
            m["balanced_accuracy"],
        )
    for n, m in sorted(per_ds_metrics.items()):
        logger.info(
            "  %-20s %5d  %.4f",
            n,
            m["samples"],
            m["balanced_accuracy"],
        )
    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
