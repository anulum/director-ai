# Token-level hallucinated-span detection

> **Status: opt-in, model-backed (`director-ai[nli]`).** Off by default; set
> `span_detection_enabled` to load the token classifier. A flagged span means
> "this text is not supported by the context", reported for review.

The response-level scorer and the claim-decompose path judge a whole answer or a
whole claim. They miss the failure mode that dominates RAGTruth: a short
**baseless addition** — a plausible detail absent from the source — embedded in
otherwise-grounded text. Entailment scoring treats the answer as mostly supported
and lets the span through. This detector works at the token level instead: it
reads `[context] [SEP] [response]` through a ModernBERT token classifier, labels
each response token *supported* or *hallucinated*, and reports the character spans
that are unsupported.

## Measured results

`benchmarks/ragtruth_token_detector_bench.py` on the balanced RAGTruth test split
(2700 examples: 943 hallucinated / 1757 grounded), example level (a response is
flagged when it contains any hallucinated span):

| approach | F1 | balanced acc | FPR | precision |
|---|---|---|---|---|
| NLI / claim-decompose | 0.366 | — | 0.347 | — |
| **token detector** | **0.763** | **0.814** | **0.071** | 0.841 |
| LettuceDetect (reference) | 0.792 | — | — | — |

Operating point: token probability ≥ 0.95, at least one flagged token. The
detector reaches parity with the open token-level baseline on balanced accuracy
and sits a few F1 points behind LettuceDetect; it is not a differentiator over
that open-source model, but it removes a real weakness in the grounding path.
Context is truncated to 1024 tokens, which caps recall on long sources.

## Usage

```python
from director_ai.guard import ProductionGuard
from director_ai.core.config import DirectorConfig

guard = ProductionGuard(config=DirectorConfig(span_detection_enabled=True))

result = guard.detect_spans(
    context="The restaurant serves Chinese and Szechuan dishes.",
    response="It serves Szechuan dishes and offers outdoor seating.",
)
result.hallucinated          # True
[(s.text, s.score) for s in result.spans]
# -> [("and offers outdoor seating", 0.98)]
```

Direct use without the guard:

```python
from director_ai.core.scoring.span_detector import HallucinationSpanDetector

detector = HallucinationSpanDetector.from_pretrained()
detection = detector.detect(context, response)
detection.coverage           # fraction of response tokens flagged
```

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `span_detection_enabled` | `False` | load the detector (opt-in) |
| `span_model` | `anulum/director-ragtruth-token-modernbert` | token classifier |
| `span_token_threshold` | `0.95` | per-token probability cut |
| `span_min_tokens` | `1` | flagged tokens needed to mark a response |
| `span_max_length` | `1024` | context+response token budget |
| `span_device` | `-1` | CUDA device index, `-1` for CPU |

## Performance

The transformer forward pass dominates the cost, but the token-probability →
character-span reduction that follows it has a Rust path
(`backfire_kernel.rust_merge_flagged_spans`) with a bit-identical pure-Python
floor. The detector uses the accelerator transparently when it is installed and
falls back to Python otherwise — span output is identical either way (a
20,000-case differential check finds no drift), so the Rust build is an
optimisation, never a correctness dependency.

Measured by `benchmarks/rust_compute_bench.py` (functional evidence — shared
workstation, not an isolated claim-grade run):

| reduction input | Python floor | Rust path | speedup |
|---|---|---|---|
| 30-token response | 4.32 µs | 1.68 µs | 2.6× |
| 400-token response | 59.52 µs | 22.04 µs | 2.7× |

## Where it fits

This is a span-level *grounding* signal, complementary to the real-time
[streaming halt](streaming.md) (which acts on contradiction during generation)
and the response-level [scoring](scoring.md). Use it for post-hoc review of RAG
answers where you need to know *which* phrases are unsupported, not just whether
the answer as a whole is grounded.
