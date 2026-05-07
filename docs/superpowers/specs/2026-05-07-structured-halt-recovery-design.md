# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Structured Halt Recovery Design

# Structured Halt Recovery Design

> Status: approved for planning on 2026-05-07.

## Goal

Add an opt-in structured halt recovery layer for JSON, tool-call, and reasoning-chain streams so downstream parsers can recover a deterministic last-valid payload after a safety halt without changing existing plain text streaming behaviour.

## Scope

This design covers the four open roadmap items under "Halt Recovery and Structured Stream Resilience":

- publish a halt recovery patterns cookbook
- add configurable `partial_output_on_halt` handling for JSON, tool-call, and reasoning-chain streams
- emit the last valid structured chunk plus a `halted_at` marker
- add structured-stream recovery tests for JSON objects, arrays, tool calls, nested tool arguments, and reasoning-chain envelopes

The feature is enabled only when callers pass a structured recovery configuration. Existing calls to `StreamingKernel.stream_tokens()` and `stream_output()` without that configuration keep their current hard-interrupt behaviour.

## Public API

Add a typed configuration object:

```python
StructuredRecoveryConfig(
    kind="json",
    policy="last_valid",
    json_schema={...},
)
```

Supported `kind` values:

- `json`
- `tool_call`
- `reasoning_chain`

Supported `policy` values:

- `last_valid`: expose only the last fully valid structured checkpoint observed before halt
- `redacted`: expose no partial payload and only provide recovery metadata
- `raw_partial`: expose raw partial text with `valid=False` for debugging or operator review

The config object should allow kind-specific validation inputs:

- `json_schema` for JSON output
- `tool_manifest` and optional `execution_log` for tool calls
- `reasoning_support_threshold` and optional `score_fn` for reasoning chains

Add a recovery result on `StreamSession`:

```python
session.structured_recovery
```

The result should include:

- `kind`
- `policy`
- `halted_at`
- `last_valid_output`
- `raw_partial`
- `valid`
- `errors`
- `metadata`

## Behaviour

The recovery layer observes the same accumulated text that the streaming scorer sees. It maintains a small state object with the raw partial text, validation verdicts, and the latest valid checkpoint. On each token, it attempts to update the checkpoint using the configured verifier.

For JSON streams, the implementation should accept fully parseable JSON roots only. It must support object and array roots. If a schema is configured, a checkpoint is valid only when `verify_json()` reports valid JSON and schema validity is not false.

For tool-call streams, the implementation should expect a JSON envelope containing at least `function_name` and `arguments`, plus optional `claimed_result`. A checkpoint is valid only when the JSON parses and `verify_tool_call()` accepts the function and arguments. Nested argument objects must be preserved exactly.

For reasoning-chain streams, the implementation should accept an envelope that can be verified deterministically, for example JSON with `steps` or text that `verify_reasoning_chain()` can parse. A checkpoint is valid when the reasoning verifier reports a valid chain. The implementation must not expose hidden chain-of-thought beyond the text supplied by the caller.

When a halt occurs:

- `halted_at` is set to `session.halt_index`.
- `last_valid` returns the last valid checkpoint and suppresses unsafe trailing partial text.
- `redacted` returns no payload.
- `raw_partial` returns the raw accumulated partial with `valid=False`.

No policy may synthesise missing braces, infer tool arguments, complete JSON, or generate replacement reasoning. Recovery is non-inventive by design.

## Integration Points

Primary code path:

- `src/director_ai/core/runtime/streaming.py`

Supporting verification modules:

- `src/director_ai/core/verification/json_verifier.py`
- `src/director_ai/core/verification/tool_call_verifier.py`
- `src/director_ai/core/verification/reasoning_verifier.py`

The implementation should keep new structured recovery logic in a focused module under `src/director_ai/core/runtime/` or `src/director_ai/core/verification/` rather than expanding `streaming.py` with parsing details.

`AsyncStreamingKernel` can reuse the same state object in a later task if the synchronous implementation lands first. The first complete feature should include synchronous `StreamingKernel` recovery and tests; async parity may be included only if it stays small and uses the same state object.

## Error Handling

Malformed intermediate payloads are expected during streaming and must not log noisy errors. They should update recovery metadata, not interrupt the stream.

Invalid configuration should raise `ValueError` at construction time with explicit messages for unsupported kind, unsupported policy, or missing kind-specific requirements.

Verifier exceptions should be captured into `errors` and should not replace the last valid checkpoint. If validation fails after a previous valid checkpoint, `last_valid` still returns the previous valid checkpoint.

## Testing

Use test-driven development. Required behavioural tests:

- JSON object stream halts after a valid object and returns that object under `last_valid`.
- JSON array stream halts mid-next-element and returns the previous valid array.
- JSON schema rejection does not replace an earlier valid checkpoint.
- Tool-call stream preserves nested arguments in the last valid checkpoint.
- Tool-call stream rejects unknown functions and keeps previous valid checkpoint.
- Reasoning-chain stream returns the last verifier-valid envelope before halt.
- `redacted` policy returns no payload and still sets `halted_at`.
- `raw_partial` policy returns partial text with `valid=False`.
- Unconfigured `structured_recovery` leaves current plain text `output` and `stream_output()` interrupt behaviour unchanged.

Focused verification should include ruff, the new recovery tests, and existing streaming tests. Broader CI remains the exhaustive gate.

## Documentation

Add a cookbook page describing operational halt recovery patterns:

- KB refresh
- threshold rollback
- human review routing
- temporary policy fallback
- parser-safe structured recovery

Update `ROADMAP.md` only after the implementation and tests are complete.

## Non-Goals

- No automatic JSON repair.
- No bracket completion.
- No inferred tool arguments.
- No replacement reasoning generation.
- No behaviour change for plain text streaming unless `StructuredRecoveryConfig` is supplied.
