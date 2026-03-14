"""Frontier LLM evaluation on AggreFact (29K samples).

Three modes:
  binary:     "supported" / "not_supported" → 0 or 1 (original, fastest)
  confidence: asks for 0-100 confidence → continuous score (fair comparison)
  fewshot:    3 labeled examples + confidence → best-case LLM

We compute macro-averaged balanced accuracy across 11 sub-datasets, matching
the Director-AI AggreFact metric.

Usage:
    python -m benchmarks.frontier_llm_eval --model gpt-4o --max-samples 1000
    python -m benchmarks.frontier_llm_eval --model gpt-4o-mini --mode confidence
    python -m benchmarks.frontier_llm_eval --model claude-sonnet-4-6 --mode fewshot

Cost estimation (per 1K samples, approximate):
    See COST_TABLE below or run with --cost-only.

Set OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY in environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks._load_aggrefact_patch import _load_aggrefact_local

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("frontier_llm_eval")

# ── Pricing (USD per million tokens, as of 2026-03) ──────────────────

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}

SUPPORTED_MODELS = {
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "claude-opus-4-6": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    "claude-haiku-4-5-20251001": "anthropic",
    "gemini-1.5-pro": "google",
    "gemini-1.5-flash": "google",
}

# ── Prompt templates ─────────────────────────────────────────────────

PROMPT_BINARY = """\
You are a factual consistency checker. Given a document and a claim, \
determine whether the claim is directly supported by the document.

Document:
{document}

Claim: {claim}

Answer with EXACTLY one word: "supported" or "not_supported"."""

PROMPT_CONFIDENCE = """\
You are a factual consistency checker. Given a document and a claim, \
rate how confident you are that the claim is supported by the document.

Document:
{document}

Claim: {claim}

Respond with ONLY a number from 0 to 100, where:
  0 = definitely not supported
  100 = definitely supported
Your answer (number only):"""

FEWSHOT_EXAMPLES = """\
Here are three examples:

Example 1:
Document: The company reported revenue of $4.2 billion in Q3 2024, up 12% year-over-year.
Claim: The company's Q3 2024 revenue was $4.2 billion.
Confidence: 95

Example 2:
Document: The study enrolled 500 patients across 12 hospitals in Europe.
Claim: The study enrolled 1000 patients across 6 hospitals.
Confidence: 5

Example 3:
Document: Solar panel installations increased by 30% in 2023 compared to the previous year.
Claim: Solar panel installations grew significantly in 2023.
Confidence: 80

Now evaluate this case:

"""

PROMPT_FEWSHOT = (
    FEWSHOT_EXAMPLES
    + """\
Document:
{document}

Claim: {claim}

Confidence (0-100, number only):"""
)

# ── Token estimation (chars / 4 ≈ tokens) ───────────────────────────

AVG_DOC_CHARS = 1500  # after 3000-char truncation, avg is ~1500
AVG_CLAIM_CHARS = 100
PROMPT_OVERHEAD_BINARY = 200
PROMPT_OVERHEAD_CONFIDENCE = 250
PROMPT_OVERHEAD_FEWSHOT = 750  # 3 examples add ~500 chars
AVG_OUTPUT_BINARY = 5  # "supported" or "not_supported"
AVG_OUTPUT_CONFIDENCE = 8  # "85" or similar


def estimate_cost(model: str, mode: str, n_samples: int) -> float:
    prices = PRICING.get(model)
    if not prices:
        return 0.0
    overhead = {
        "binary": PROMPT_OVERHEAD_BINARY,
        "confidence": PROMPT_OVERHEAD_CONFIDENCE,
        "fewshot": PROMPT_OVERHEAD_FEWSHOT,
    }[mode]
    output_chars = AVG_OUTPUT_BINARY if mode == "binary" else AVG_OUTPUT_CONFIDENCE
    input_tokens = (AVG_DOC_CHARS + AVG_CLAIM_CHARS + overhead) / 4
    output_tokens = output_chars / 4
    input_cost = input_tokens * prices["input"] / 1_000_000
    output_cost = output_tokens * prices["output"] / 1_000_000
    return (input_cost + output_cost) * n_samples


def print_cost_table():
    modes = ["binary", "confidence", "fewshot"]
    sample_counts = [200, 1000, 29320]
    models = list(PRICING.keys())

    print(f"\n{'Model':<32s}", end="")
    for mode in modes:
        for n in sample_counts:
            label = f"{mode[:3]}/{n}"
            print(f"  {label:>10s}", end="")
    print()
    print("-" * 130)

    for model in models:
        print(f"  {model:<30s}", end="")
        for mode in modes:
            for n in sample_counts:
                cost = estimate_cost(model, mode, n)
                print(f"  ${cost:>8.2f}", end="")
        print()

    print(f"\n{'Recommended test matrix':>32s}")
    print("-" * 70)
    recommended = [
        ("gpt-4o", "confidence", 1000),
        ("gpt-4o-mini", "confidence", 1000),
        ("gpt-4o-mini", "confidence", 29320),
        ("claude-sonnet-4-6", "confidence", 1000),
        ("claude-haiku-4-5-20251001", "confidence", 1000),
        ("gpt-4o", "fewshot", 1000),
        ("claude-sonnet-4-6", "fewshot", 1000),
    ]
    total = 0.0
    for model, mode, n in recommended:
        cost = estimate_cost(model, mode, n)
        total += cost
        print(f"  {model:<30s} {mode:<12s} {n:>6d} samples  ${cost:.2f}")
    print(f"\n  {'TOTAL':<30s} {'':12s} {'':>6s}          ${total:.2f}")


# ── API callers ──────────────────────────────────────────────────────


def _parse_confidence(text: str) -> float | None:
    """Extract a 0-100 confidence number from LLM output."""
    match = re.search(r"\b(\d{1,3})\b", text.strip())
    if match:
        val = int(match.group(1))
        return min(max(val / 100.0, 0.0), 1.0)
    return None


def _parse_binary(text: str) -> float:
    text = text.strip().lower()
    return 1.0 if "supported" in text and "not" not in text else 0.0


def _call_openai(model: str, prompt: str, mode: str) -> float | None:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    max_tokens = 5 if mode == "binary" else 10
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        if mode == "binary":
            return _parse_binary(text)
        return _parse_confidence(text)
    except Exception as exc:
        log.warning("OpenAI error: %s", exc)
        return None


def _call_anthropic(model: str, prompt: str, mode: str) -> float | None:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    max_tokens = 5 if mode == "binary" else 10
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        if mode == "binary":
            return _parse_binary(text)
        return _parse_confidence(text)
    except Exception as exc:
        log.warning("Anthropic error: %s", exc)
        return None


def _call_google(model: str, prompt: str, mode: str) -> float | None:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    max_tokens = 5 if mode == "binary" else 10
    try:
        m = genai.GenerativeModel(model)
        resp = m.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.0},
        )
        text = resp.text
        if mode == "binary":
            return _parse_binary(text)
        return _parse_confidence(text)
    except Exception as exc:
        log.warning("Google error: %s", exc)
        return None


def _score_pair(
    provider: str,
    model: str,
    doc: str,
    claim: str,
    mode: str,
) -> float | None:
    template = {
        "binary": PROMPT_BINARY,
        "confidence": PROMPT_CONFIDENCE,
        "fewshot": PROMPT_FEWSHOT,
    }[mode]
    prompt = template.format(document=doc[:3000], claim=claim)
    call_mode = mode if mode != "fewshot" else "confidence"
    if provider == "openai":
        return _call_openai(model, prompt, call_mode)
    if provider == "anthropic":
        return _call_anthropic(model, prompt, call_mode)
    if provider == "google":
        return _call_google(model, prompt, call_mode)
    raise ValueError(f"Unknown provider: {provider}")


# ── Metrics ──────────────────────────────────────────────────────────


def _macro_ba(by_ds: dict[str, list[tuple[int, float]]], threshold: float) -> float:
    bas = []
    for rows in by_ds.values():
        labels = np.array([r[0] for r in rows])
        scores = np.array([r[1] for r in rows])
        if len(np.unique(labels)) < 2:
            continue
        preds = (scores >= threshold).astype(int)
        recalls = [(preds[labels == c] == c).mean() for c in np.unique(labels)]
        bas.append(float(np.mean(recalls)))
    return float(np.mean(bas)) if bas else 0.0


def _best_threshold(by_ds: dict) -> tuple[float, float]:
    best_ba, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        ba = _macro_ba(by_ds, t)
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t, best_ba


# ── Main evaluation ─────────────────────────────────────────────────


def run_frontier_eval(
    model: str,
    mode: str = "binary",
    max_samples: int | None = None,
    seed: int = 42,
    rate_limit_rps: float = 2.0,
) -> dict:
    provider = SUPPORTED_MODELS.get(model)
    if provider is None:
        raise ValueError(
            f"Unknown model '{model}'. Supported: {list(SUPPORTED_MODELS)}",
        )

    est_cost = estimate_cost(model, mode, max_samples or 29320)
    log.info(
        "Estimated cost: $%.2f (%s, %s, %d samples)",
        est_cost,
        model,
        mode,
        max_samples or 29320,
    )

    rows = _load_aggrefact_local()
    log.info("Loaded %d AggreFact rows", len(rows))

    if max_samples and max_samples < len(rows):
        by_ds: dict[str, list] = {}
        for r in rows:
            by_ds.setdefault(r.get("dataset", "unknown"), []).append(r)
        rng = random.Random(seed)
        sampled = []
        per_ds = max(1, max_samples // len(by_ds))
        for ds_rows in by_ds.values():
            rng.shuffle(ds_rows)
            sampled.extend(ds_rows[:per_ds])
        rng.shuffle(sampled)
        rows = sampled[:max_samples]
        log.info(
            "Sampled %d rows (%d per dataset, %d datasets)",
            len(rows),
            per_ds,
            len(by_ds),
        )

    results: dict[str, list[tuple[int, float]]] = {}
    errors = 0
    t0 = time.perf_counter()
    delay = 1.0 / rate_limit_rps

    for i, row in enumerate(rows):
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds = row.get("dataset", "unknown")
        if label is None or not doc or not claim:
            continue

        score = _score_pair(provider, model, doc, claim, mode)
        if score is None:
            errors += 1
            continue
        results.setdefault(ds, []).append((int(label), score))

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            eta_min = (len(rows) - i - 1) * (elapsed / (i + 1)) / 60
            log.info(
                "%s: %d/%d (%.0f min remaining, %d errors)",
                model,
                i + 1,
                len(rows),
                eta_min,
                errors,
            )
        time.sleep(delay)

    elapsed = time.perf_counter() - t0
    best_t, best_ba = _best_threshold(results)

    per_dataset = {}
    for ds, ds_rows in results.items():
        labels = np.array([r[0] for r in ds_rows])
        scores = np.array([r[1] for r in ds_rows])
        if len(np.unique(labels)) >= 2:
            preds = (scores >= best_t).astype(int)
            recalls = [(preds[labels == c] == c).mean() for c in np.unique(labels)]
            per_dataset[ds] = {
                "n": len(ds_rows),
                "balanced_accuracy": float(np.mean(recalls)),
            }

    actual_cost = estimate_cost(model, mode, sum(len(v) for v in results.values()))

    summary = {
        "model": model,
        "provider": provider,
        "mode": mode,
        "n_evaluated": sum(len(v) for v in results.values()),
        "n_errors": errors,
        "best_threshold": best_t,
        "macro_balanced_accuracy": best_ba,
        "elapsed_s": round(elapsed, 1),
        "estimated_cost_usd": round(actual_cost, 4),
        "per_dataset": per_dataset,
    }

    log.info(
        "%s [%s]: macro BA=%.4f @ t=%.2f (%d samples, %d errors, %.1f min, ~$%.2f)",
        model,
        mode,
        best_ba,
        best_t,
        summary["n_evaluated"],
        errors,
        elapsed / 60,
        actual_cost,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Frontier LLM vs AggreFact")
    parser.add_argument("--model", choices=list(SUPPORTED_MODELS))
    parser.add_argument(
        "--mode",
        choices=["binary", "confidence", "fewshot"],
        default="binary",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate-limit", type=float, default=2.0, help="API calls/sec")
    parser.add_argument("--out", type=str, help="Output JSON path")
    parser.add_argument(
        "--cost-only",
        action="store_true",
        help="Print cost table and exit",
    )
    args = parser.parse_args()

    if args.cost_only:
        print_cost_table()
        return

    if not args.model:
        parser.error("--model is required (unless --cost-only)")

    result = run_frontier_eval(
        model=args.model,
        mode=args.mode,
        max_samples=args.max_samples,
        seed=args.seed,
        rate_limit_rps=args.rate_limit,
    )

    out_path = (
        args.out
        or f"benchmarks/results/frontier_{args.model}_{args.mode}_{args.max_samples}samples.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Model:          {result['model']}")
    print(f"Mode:           {result['mode']}")
    print(f"Samples:        {result['n_evaluated']}")
    print(
        f"Macro BA:       {result['macro_balanced_accuracy']:.4f}  @ t={result['best_threshold']:.2f}",
    )
    print(f"Errors:         {result['n_errors']}")
    print(f"Time:           {result['elapsed_s'] / 60:.1f} min")
    print(f"Est. cost:      ${result['estimated_cost_usd']:.2f}")
    print("\nPer-dataset breakdown:")
    for ds, m in sorted(
        result["per_dataset"].items(),
        key=lambda x: x[1]["balanced_accuracy"],
    ):
        print(f"  {ds:<30s} {m['balanced_accuracy']:.3f}  (n={m['n']})")
    print(f"\nResult saved to: {out_path}")

    print(f"\n{'Director-AI baseline':30s} 0.7586  (75.86% macro BA on full 29K)")
    diff = result["macro_balanced_accuracy"] - 0.7586
    sign = "+" if diff >= 0 else ""
    print(
        f"{result['model']:30s} {result['macro_balanced_accuracy']:.4f}  ({sign}{diff * 100:.2f}pp vs Director-AI)",
    )


if __name__ == "__main__":
    main()
