# Multimodal Checks

Multimodal checks adapt image, audio, and video evidence into the shared
`GuardDecision` and `SafetyEvent` contracts. The adapter is explicitly opt-in:
a modality must be enabled before it can be checked, and a modality must be
marked benchmarked before a supported result may become `allow`.

## Decision Boundary

`MultimodalVerifierAdapter` enforces production-safe defaults:

- disabled or unsupported modalities raise errors instead of silently passing
- uncertain evidence maps to `warn`, never `allow`
- hallucinated or temporally inconsistent evidence maps to `halt`
- unbenchmarked modalities map to `warn` even when the low-level checker says
  the claim is consistent
- optional caption and metadata grounding can reduce a modality score before a
  decision is emitted
- invalid video temporal policy fails during adapter construction:
  `temporal_alpha` must be finite and in `(0, 1]`, and `temporal_floor` must be
  finite and in `[0, 1]`
- audit payloads and safety events include media references, not raw media,
  transcripts, frame data, captions, metadata values, or claim text

```python
from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.multimodal_guard import (
    MultimodalCheckRequest,
    MultimodalVerifierAdapter,
)

adapter = MultimodalVerifierAdapter(
    image_guard=image_guard,
    caption_score_fn=caption_grounder,
    metadata_score_fn=metadata_grounder,
    enabled_modalities=("image",),
    benchmarked_modalities=("image",),
)

result = adapter.check(
    MultimodalCheckRequest(
        modality="image",
        claim_text="The image shows a labelled package.",
        media_ref="media://image-42",
        image_bytes=image_bytes,
        caption_text="Package label is absent.",
        metadata={"captured_at": "2026-05-13", "source": "inspection-rig"},
    ),
    risk_envelope=RiskEnvelope(
        action_category="multimodal",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    ),
    policy_id="policy.multimodal.regulated",
)
```

Grounding callbacks receive either `(caption_text, claim_text)` or
`(metadata, claim_text)` and must return a finite score in `[0, 1]`. Scores
below the grounding floor halt the claim; scores below the grounding allow
threshold produce a warning unless the base verifier already found a stricter
verdict. Evidence references use suffixes such as `#caption` and
`#metadata:captured_at`, so downstream audit logs can identify which grounding
channel was used without storing private captions or metadata values.

## Local Evidence Packet

Generate the local multimodal temporal evidence packet before promoting a
deployment that enables image, audio, or video checks:

```bash
PYTHONPATH=src python -m benchmarks.multimodal_temporal_evidence
```

The packet exercises image allow/halt paths, caption-grounding conflicts, video
frame temporal halts, and the dependency-free hash-bag encoder/verifier path. It
does not include an external Vision-NLI benchmark or a real video model run.

## Live endpoint (opt-in, isolated)

The guard is reachable at `POST /v1/multimodal/check`, but stays out of the
default path: the endpoint returns **404** unless experimental hooks are enabled
(`DIRECTOR_AI_ENABLE_EXPERIMENTAL_HOOKS=1`) **and** at least one modality is set
via `DirectorConfig.multimodal_enabled_modalities`. When enabled, the server
builds a dependency-free hash-bag adapter (`build_hashbag_adapter`) at startup —
image via the hash-bag guard, audio/caption/metadata via `text_bag_similarity`,
video via per-request frame similarities — so no ML stack is required.

The request carries `modality`, `claim_text`, `media_ref`, the modality payload
(`image_base64` / `transcript_text` / `frame_similarities`), and optional
`caption_text` / `metadata`. The response is the tenant-safe `GuardDecision`
(`allow` / `warn` / `halt`); it never echoes raw media, transcript, or claim text.

## In-process live path

Besides the `/v1/multimodal/check` REST endpoint, the guard exposes the check
in-process so a library caller does not need the server:

```python
from director_ai.core.config import DirectorConfig
from director_ai.guard import ProductionGuard
from director_ai.core.multimodal_guard import MultimodalCheckRequest

guard = ProductionGuard(config=DirectorConfig(
    multimodal_enabled_modalities=("image",),
    multimodal_benchmarked_modalities=("image",),
))
result = guard.check_multimodal(MultimodalCheckRequest(
    modality="image",
    claim_text="a tabby cat on a sofa",
    media_ref="img://catalogue/1",
    image_bytes=image_payload,
))
print(result.guard_decision.decision)   # allow / warn / halt
```

`guard.multimodal_adapter` is built lazily from the `multimodal_*` config (the
dependency-free hash-bag adapter — no torch). The guard is opt-in: it raises
unless `multimodal_enabled_modalities` is set, and an enabled-but-unbenchmarked
modality resolves to `warn`, never a silent `allow`. Invalid temporal video
settings fail before the first request is accepted, so a misconfigured
`multimodal_temporal_alpha` or `multimodal_temporal_floor` cannot silently weaken
frame-drift halting. Pass a torch/CLIP-backed adapter as
`guard.check_multimodal(request, adapter=...)` for semantic verification. The
same tenant-safe `MultimodalCheckResult` is returned as the endpoint.

## Full API

::: director_ai.core.multimodal_guard.adapter.MultimodalCheckRequest

::: director_ai.core.multimodal_guard.adapter.MultimodalCheckResult

::: director_ai.core.multimodal_guard.adapter.MultimodalVerifierAdapter
