# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — pre-sampling logits-processor overhead benchmark

"""Per-token overhead the inference-server logits adapter adds to generation.

Measures the cost Director-AI contributes at each decode step when wired as a
vLLM/TGI/llama.cpp logits processor, with a local heuristic scorer (no model, no
network) so the numbers are the adapter's own overhead, not LLM or NLI latency:

* **pass-through** — the steady-state per-token cost between claim boundaries
  (just the boundary check; logits untouched);
* **allow at boundary** — boundary check + one local coherence score;
* **EOS mask** — the one-off O(vocab) masking applied when a halt fires.

Vocab sizes cover common tokenizers (32k, 128k). Run::

    python -m benchmarks.inference_hook_overhead --repeats 2000
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.integrations.inference_logits_adapters import (
    LogitsHaltProcessor,
    _force_halt,
)
from director_ai.integrations.inference_server_hooks import build_inference_server_hook

RESULTS_DIR = Path(__file__).parent / "results"
VOCAB_SIZES = (32_000, 128_256)
_PROMPT = "What is the capital of France?"
_SAFE_CLAIM = "The capital of France is Paris."  # claim boundary, grounded
_PARTIAL = "The capital of France is Par"  # no terminator -> pass-through

# Mutable decode target so one processor can be pointed at different texts
# without rebuilding the scorer between measurement phases.
_DECODE_STATE = {"text": _SAFE_CLAIM}


def _percentiles(samples_us: list[float]) -> dict:
    ordered = sorted(samples_us)
    return {
        "n": len(ordered),
        "p50_us": round(statistics.median(ordered), 3),
        "p95_us": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "mean_us": round(statistics.fmean(ordered), 3),
    }


def _build_processor() -> LogitsHaltProcessor:
    store = GroundTruthStore()
    store.add(_PROMPT, _SAFE_CLAIM)
    scorer = CoherenceScorer(threshold=0.3, ground_truth_store=store, use_nli=False)
    hook = build_inference_server_hook(
        "vllm",
        score_fn=lambda text: scorer.review(_PROMPT, text)[1].score,
        hard_limit=0.4,
    )
    return LogitsHaltProcessor(
        hook, decode_fn=lambda _ids: _DECODE_STATE["text"], eos_token_id=2
    )


def _time(fn, repeats: int) -> dict:
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)  # microseconds
    return _percentiles(samples)


def run_benchmark(repeats: int) -> dict:
    proc = _build_processor()
    token_ids = list(range(16))
    # Pre-allocate the logits buffer once; allocating it inside the timed loop
    # would measure list construction, not the adapter's own per-token overhead.
    # pass-through and allow paths never mutate the buffer, so it is reused.
    logits = [1.0] * 32_000

    # Steady-state: between claim boundaries the logits are passed through.
    _DECODE_STATE["text"] = _PARTIAL
    pass_through = _time(lambda: proc(token_ids, logits), repeats)

    # At a claim boundary: boundary check + one local coherence score (allow).
    _DECODE_STATE["text"] = _SAFE_CLAIM
    allow_boundary = _time(lambda: proc(token_ids, logits), repeats)

    # The one-off EOS mask cost scales with vocab size. Pre-allocate per vocab and
    # re-mask in place each iteration (re-masking costs the same O(vocab) pass).
    mask_by_vocab = {}
    for vocab in VOCAB_SIZES:
        buf = [1.0] * vocab
        mask_by_vocab[str(vocab)] = _time(
            lambda b=buf: _force_halt(b, eos_token_id=2),
            max(repeats // 10, 50),
        )

    return {
        "benchmark": "inference_hook_overhead",
        "scorer": "CoherenceScorer (heuristic, use_nli=False), local",
        "repeats": repeats,
        "pass_through_per_token": pass_through,
        "allow_at_boundary": allow_boundary,
        "eos_mask_by_vocab": mask_by_vocab,
        "note": (
            "Per-token overhead added by the Director-AI logits adapter only "
            "(detokenisation is the server's cost). The EOS mask is a one-off at "
            "halt, not a per-token cost."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()

    result = run_benchmark(args.repeats)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "inference_hook_overhead.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    pt = result["pass_through_per_token"]
    ab = result["allow_at_boundary"]
    print(f"\nLogits-adapter overhead ({args.repeats} repeats):")
    print(
        f"  pass-through / token : p50 {pt['p50_us']:.2f}us  mean {pt['mean_us']:.2f}us"
    )
    print(
        f"  allow at boundary    : p50 {ab['p50_us']:.2f}us  mean {ab['mean_us']:.2f}us"
    )
    for vocab, m in result["eos_mask_by_vocab"].items():
        print(f"  EOS mask (vocab {vocab}): p50 {m['p50_us']:.2f}us (one-off at halt)")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
