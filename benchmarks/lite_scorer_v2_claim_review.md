# Lite Scorer v2 Benchmark Claim Review

Date: 2026-06-19

## Decision

No public benchmark or edge/mobile score claim is approved by this file.
The evidence packet is complete enough for operator review, but the release
claim remains pending explicit operator approval.

## Recorded Evidence

Evidence packet: `benchmarks/lite_scorer_v2_evidence_packet.toml`

Run manifest: `benchmarks/lite_scorer_v2_run_manifest.toml`

Student candidate: `minilm_l6`

Teacher artefact:
`training/output/deberta-v3-large-hallucination/model.safetensors`

Teacher SHA-256:
`75c6cf7d9945143581b8c517d1d44f5717d6fed156b354fcd995b6eb89570163`

Student artefact:
`MODELS/lite-scorer-v2/student/model.safetensors`

Student SHA-256:
`652590428a92ebcfd6f51ae90df9c93ff2c1b8b1bd2fe4d11db5d9eff7bd887a`

Quantized ONNX artefact:
`MODELS/lite-scorer-v2/onnx/model_quantized.onnx`

ONNX SHA-256:
`6ea859c2c3d17c5799e6e2dcd9ee183304f0df7d6cd44da489a94424ab5a747d`

## Held-Out Evaluation

Dataset: `benchmarks/heldout/lite_scorer_v2.jsonl`

Rows: `1000`

Threshold: `0.685047`

Balanced accuracy: `0.756`

True-positive rate: `0.676`

True-negative rate: `0.836`

## Latency

Backend: ONNX Runtime

Device: CPU

Samples: `100`

p50: `71.907606` ms

p95: `239.390694` ms

## Claim Boundary

The recorded result supports only an internal statement that a Lite Scorer v2
student artefact exists and has been evaluated on a 1000-row held-out set with
the metrics above. It does not support a public accuracy claim, a leaderboard
claim, a production readiness claim, or an edge/mobile superiority claim.

The CPU latency evidence is local recorded evidence only. It is not isolated
benchmark evidence under the production benchmark policy and must not be used as
a production latency claim without a separate isolated run.

## Review Notes

The recorded balanced accuracy is below the current FactCG production baseline
used elsewhere in the project. Treat this artefact as an edge/mobile candidate
or pre-filter candidate, not as a replacement for the production NLI scorer.

Operator approval, if granted later, should name the exact permitted wording and
the exact surfaces where it may appear.
