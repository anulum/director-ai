# Inference-server hooks (vLLM / TGI / llama.cpp)

Director-AI can guard a self-hosted LLM **at the pre-sampling boundary** — it
scores a candidate continuation and masks or biases the token's logit *before*
the server samples it. This is stronger than guarding the output stream after the
fact: an unsafe token can be prevented from ever being emitted, not detected once
it already has.

The hook is **server-neutral**: it returns a decision plus a per-server payload,
and a thin adapter applies that to the server's logits processor / sampling
callback. The same hook drives vLLM, TGI, and llama.cpp.

## Why this matters

| | Output-stream halt (`StreamingKernel`) | Pre-sampling hook |
|---|---|---|
| Acts | after a token is emitted | before a token is sampled |
| Can suppress a token | no (only halts the stream) | yes (masks/biases the logit) |
| Integration point | response stream | server logits processor |

For anyone running vLLM, TGI, or llama.cpp in production, the pre-sampling hook
is the tightest place to enforce a guard.

## Quickstart

```python
from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.integrations.inference_server_hooks import (
    InferenceHookRequest,
    build_inference_server_hook,
)

store = GroundTruthStore()
store.add("capital of France", "The capital of France is Paris.")
scorer = CoherenceScorer(threshold=0.3, ground_truth_store=store, use_nli=False)

# score_fn maps the candidate continuation to a coherence score in [0, 1].
hook = build_inference_server_hook(
    "vllm",
    score_fn=lambda text: scorer.review("capital of France", text)[1].score,
    hard_limit=0.4,          # below this, the candidate token is rejected
)

request = InferenceHookRequest(
    server="vllm",
    accumulated_text="The capital of France is ",
    candidate_token="Berlin",
    token_id=12345,
    request_id="req-1",
    tenant_id="tenant-a",
)

decision = hook.check(request, logits=current_logits)  # current_logits optional
if not decision.allow:
    # decision.adjusted_logits has the candidate token masked to block_logit;
    # decision.blocked_token_ids lists what was suppressed;
    # decision.safety_event is a tenant-safe audit record;
    # decision.server_payload carries the vLLM/TGI/llama.cpp-specific action.
    apply_to_server(decision.server_payload)
```

`hook.check()` scores `request.candidate_text` (the accumulated text plus the
candidate token). If the score is at or above `hard_limit` the token is allowed;
otherwise its logit is set to `block_logit` (default `-1e9`, an effective mask),
a `SafetyEvent` is emitted, and the blocked token id is reported.

## API

### `build_inference_server_hook(server, score_fn, *, hard_limit=0.4, block_token_id=None, block_logit=-1e9)`

Builds an `InferenceServerHook` with the default policy. `server` is one of
`"vllm"`, `"tgi"`, `"llama_cpp"`. `score_fn` is `Callable[[str], float]`
returning a coherence/grounding score in `[0, 1]`.

### `InferenceHookRequest`

The candidate token plus stream context: `server`, `accumulated_text`,
`candidate_token`, optional `token_id`, `request_id`, `tenant_id`, and
`metadata`. The `candidate_text` property returns `accumulated_text +
candidate_token` — what `score_fn` receives.

### `InferenceServerHook.check(request, logits=None) -> InferenceHookDecision`

Scores one candidate token and returns a server-neutral action.
`InferenceHookDecision` carries `allow`, `score`, `reason`, `adjusted_logits`
(the masked logit vector when `logits` is supplied), `blocked_token_ids`,
`safety_event`, and `server_payload`.

### `InferenceServerHookPolicy`

Tunes the decision: `hard_limit`, `block_token_id`, `block_logit`,
`steering_bias_logit`, `halt_reason`, `tenant_safe_explanation`.

## Predictive pre-halt steering (advanced tier)

`InferenceServerHook.steer(request, steering_decision, logits)` applies a
predictive `PreHaltSteeringDecision` (from the advanced `core.trajectory`
module): `proceed` allows the token, `escalate` biases its logit by
`steering_bias_logit` (discouraged but not blocked), and `halt` masks it. This
lets the guard steer generation away from a predicted unsafe trajectory before it
commits, rather than only reacting at the hard limit.

## Multi-tenant and audit

Every rejection produces a tenant-safe `SafetyEvent` (`hook_scope =
"inference_server"`) carrying the request id, tenant id, threshold, and observed
score — never the raw prompt or response — so the pre-sampling guard feeds the
same hash-chained audit trail as the rest of Director-AI.
