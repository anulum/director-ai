# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Structured Halt Recovery Implementation Plan

# Structured Halt Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in structured halt recovery for JSON, tool-call, and reasoning-chain streams while preserving current plain text streaming behaviour.

**Architecture:** Create a focused `structured_recovery.py` runtime helper that owns configuration validation, incremental checkpointing, and final recovery result construction. `StreamingKernel` passes accumulated stream text into that helper and attaches the final result to `StreamSession` only when callers opt in through `StructuredRecoveryConfig`.

**Tech Stack:** Python dataclasses, stdlib JSON parser, existing `verify_json()`, `verify_tool_call()`, `verify_reasoning_chain()`, pytest, ruff, MkDocs docs.

---

## File Structure

- Create `src/director_ai/core/runtime/structured_recovery.py`
  - Owns `StructuredRecoveryConfig`, `StructuredRecoveryResult`, and `StructuredRecoveryState`.
  - Keeps parsing and verifier-specific logic out of `streaming.py`.
- Modify `src/director_ai/core/runtime/streaming.py`
  - Adds `structured_recovery` field to `StreamSession`.
  - Adds optional `structured_recovery` parameter to `StreamingKernel.stream_tokens()`.
  - Updates recovery state once per token and finalises it on halt.
- Modify `src/director_ai/core/__init__.py`
  - Re-export `StructuredRecoveryConfig` and `StructuredRecoveryResult`.
- Modify `src/director_ai/__init__.py`
  - Add lazy imports for the two public recovery types.
- Create `tests/test_structured_halt_recovery.py`
  - Behavioural tests for JSON, tool-call, reasoning-chain, policies, and backwards compatibility.
- Create `docs-site/cookbook/halt-recovery-patterns.md`
  - Operator cookbook for KB refresh, threshold rollback, human review routing, temporary policy fallback, and parser-safe structured recovery.
- Modify `mkdocs.yml`
  - Add the cookbook page to the docs navigation if cookbook pages are listed explicitly.
- Modify `ROADMAP.md`
  - Mark the four structured halt recovery items complete only after implementation and tests pass.

---

### Task 1: Recovery Types and Config Validation

**Files:**
- Create: `src/director_ai/core/runtime/structured_recovery.py`
- Test: `tests/test_structured_halt_recovery.py`

- [ ] **Step 1: Write the failing config validation tests**

Add this new test file:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Structured Halt Recovery Tests

from __future__ import annotations

import pytest

from director_ai.core.runtime.structured_recovery import (
    StructuredRecoveryConfig,
    StructuredRecoveryResult,
)


def test_structured_recovery_config_accepts_supported_kinds_and_policies() -> None:
    config = StructuredRecoveryConfig(kind="json", policy="last_valid")

    assert config.kind == "json"
    assert config.policy == "last_valid"


@pytest.mark.parametrize("kind", ["xml", "", "JSON"])
def test_structured_recovery_config_rejects_unknown_kind(kind: str) -> None:
    with pytest.raises(ValueError, match="structured_recovery kind"):
        StructuredRecoveryConfig(kind=kind, policy="last_valid")


@pytest.mark.parametrize("policy", ["repair", "", "debug"])
def test_structured_recovery_config_rejects_unknown_policy(policy: str) -> None:
    with pytest.raises(ValueError, match="structured_recovery policy"):
        StructuredRecoveryConfig(kind="json", policy=policy)


def test_structured_recovery_result_defaults_are_parser_safe() -> None:
    result = StructuredRecoveryResult(
        kind="json",
        policy="redacted",
        halted_at=3,
    )

    assert result.last_valid_output == ""
    assert result.raw_partial == ""
    assert result.valid is False
    assert result.errors == []
    assert result.metadata == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py -q
```

Expected: import failure for `director_ai.core.runtime.structured_recovery`.

- [ ] **Step 3: Add the minimal recovery type implementation**

Create `src/director_ai/core/runtime/structured_recovery.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Structured Halt Recovery

"""Structured recovery checkpoints for halted token streams."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "StructuredRecoveryConfig",
    "StructuredRecoveryResult",
    "StructuredRecoveryState",
]

_KINDS = frozenset({"json", "tool_call", "reasoning_chain"})
_POLICIES = frozenset({"last_valid", "redacted", "raw_partial"})


@dataclass(frozen=True)
class StructuredRecoveryConfig:
    """Configuration for opt-in parser-safe recovery after stream halt."""

    kind: str
    policy: str = "last_valid"
    json_schema: dict[str, Any] | None = None
    tool_manifest: dict[str, Any] | None = None
    execution_log: Sequence[dict[str, Any]] | None = None
    score_fn: Callable[..., float] | None = None
    reasoning_support_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported structured_recovery kind {self.kind!r}")
        if self.policy not in _POLICIES:
            raise ValueError(f"unsupported structured_recovery policy {self.policy!r}")
        if self.reasoning_support_threshold < 0.0:
            raise ValueError("reasoning_support_threshold must be non-negative")


@dataclass(frozen=True)
class StructuredRecoveryResult:
    """Parser-safe structured recovery result attached to StreamSession."""

    kind: str
    policy: str
    halted_at: int
    last_valid_output: str = ""
    raw_partial: str = ""
    valid: bool = False
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuredRecoveryState:
    """Mutable structured checkpoint state for a single stream."""

    def __init__(self, config: StructuredRecoveryConfig) -> None:
        self.config = config
        self.raw_partial = ""
        self.last_valid_output = ""
        self.last_valid_metadata: dict[str, Any] = {}
        self.errors: list[str] = []

    def update(self, text: str) -> None:
        self.raw_partial = text

    def finalise(self, halted_at: int) -> StructuredRecoveryResult:
        return StructuredRecoveryResult(
            kind=self.config.kind,
            policy=self.config.policy,
            halted_at=halted_at,
            last_valid_output=(
                self.last_valid_output if self.config.policy == "last_valid" else ""
            ),
            raw_partial=self.raw_partial if self.config.policy == "raw_partial" else "",
            valid=bool(self.last_valid_output and self.config.policy == "last_valid"),
            errors=list(self.errors),
            metadata=dict(self.last_valid_metadata),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/director_ai/core/runtime/structured_recovery.py tests/test_structured_halt_recovery.py
git commit -m "Add structured recovery config types" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

---

### Task 2: JSON Checkpoint Recovery

**Files:**
- Modify: `src/director_ai/core/runtime/structured_recovery.py`
- Modify: `tests/test_structured_halt_recovery.py`

- [ ] **Step 1: Write failing JSON recovery tests**

Append:

```python
from director_ai.core.runtime.structured_recovery import StructuredRecoveryState


def test_json_recovery_keeps_last_valid_object_before_malformed_suffix() -> None:
    state = StructuredRecoveryState(StructuredRecoveryConfig(kind="json"))
    state.update('{"status": "ok"}')
    state.update('{"status": "ok"}{"status": ')

    result = state.finalise(halted_at=2)

    assert result.halted_at == 2
    assert result.valid is True
    assert result.last_valid_output == '{"status":"ok"}'
    assert result.raw_partial == ""
    assert result.metadata["json_root"] == "object"


def test_json_recovery_keeps_last_valid_array_before_mid_element_halt() -> None:
    state = StructuredRecoveryState(StructuredRecoveryConfig(kind="json"))
    state.update('[{"id": 1}]')
    state.update('[{"id": 1}, {"id":')

    result = state.finalise(halted_at=4)

    assert result.valid is True
    assert result.last_valid_output == '[{"id":1}]'
    assert result.metadata["json_root"] == "array"


def test_json_schema_rejection_does_not_replace_previous_checkpoint() -> None:
    config = StructuredRecoveryConfig(
        kind="json",
        json_schema={
            "type": "object",
            "required": ["status"],
            "properties": {"status": {"type": "string"}},
        },
    )
    state = StructuredRecoveryState(config)
    state.update('{"status": "ok"}')
    state.update('{"status": 42}')

    result = state.finalise(halted_at=3)

    assert result.valid is True
    assert result.last_valid_output == '{"status":"ok"}'
    assert any("schema" in error for error in result.errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py -q
```

Expected: JSON tests fail because `update()` does not parse or checkpoint.

- [ ] **Step 3: Implement JSON checkpointing**

Update `structured_recovery.py`:

```python
import json

from ..verification.json_verifier import verify_json
```

Replace `update()` and add helpers:

```python
    def update(self, text: str) -> None:
        self.raw_partial = text
        if self.config.kind == "json":
            self._update_json(text)

    def _update_json(self, text: str) -> None:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            self._remember_error(f"json_parse:{exc.msg if hasattr(exc, 'msg') else exc}")
            return
        verdict = verify_json(text, schema=self.config.json_schema)
        if not verdict.valid_json:
            self._remember_error(f"json_parse:{verdict.parse_error}")
            return
        if verdict.schema_valid is False or verdict.error_count:
            self._remember_error("json_schema:invalid")
            return
        self.last_valid_output = json.dumps(data, separators=(",", ":"), sort_keys=True)
        self.last_valid_metadata = {
            "json_root": "array" if isinstance(data, list) else type(data).__name__,
            "field_errors": 0,
        }

    def _remember_error(self, message: str) -> None:
        if not self.errors or self.errors[-1] != message:
            self.errors.append(message)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py -q
```

Expected: JSON recovery tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/director_ai/core/runtime/structured_recovery.py tests/test_structured_halt_recovery.py
git commit -m "Add JSON structured halt recovery" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

---

### Task 3: Tool-Call Checkpoint Recovery

**Files:**
- Modify: `src/director_ai/core/runtime/structured_recovery.py`
- Modify: `tests/test_structured_halt_recovery.py`

- [ ] **Step 1: Write failing tool-call recovery tests**

Append:

```python
TOOL_MANIFEST = {
    "book_flight": {
        "parameters": {
            "route": {"type": "object"},
            "passengers": {"type": "integer"},
        },
        "returns": "booking confirmation",
    },
}


def test_tool_call_recovery_preserves_nested_arguments() -> None:
    state = StructuredRecoveryState(
        StructuredRecoveryConfig(
            kind="tool_call",
            tool_manifest=TOOL_MANIFEST,
        )
    )
    state.update(
        '{"function_name":"book_flight","arguments":{"route":{"from":"ZRH","to":"PRG"},"passengers":2}}'
    )
    state.update('{"function_name":"book_flight","arguments":{"route":')

    result = state.finalise(halted_at=5)

    assert result.valid is True
    assert result.last_valid_output == (
        '{"arguments":{"passengers":2,"route":{"from":"ZRH","to":"PRG"}},'
        '"function_name":"book_flight"}'
    )
    assert result.metadata["function_name"] == "book_flight"


def test_tool_call_unknown_function_keeps_previous_valid_checkpoint() -> None:
    state = StructuredRecoveryState(
        StructuredRecoveryConfig(
            kind="tool_call",
            tool_manifest=TOOL_MANIFEST,
        )
    )
    state.update(
        '{"function_name":"book_flight","arguments":{"route":{"from":"ZRH","to":"PRG"},"passengers":2}}'
    )
    state.update('{"function_name":"wire_money","arguments":{"amount":1000}}')

    result = state.finalise(halted_at=6)

    assert result.valid is True
    assert "book_flight" in result.last_valid_output
    assert any("tool_call" in error for error in result.errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py -q
```

Expected: tool-call tests fail because `kind="tool_call"` has no parser path.

- [ ] **Step 3: Implement tool-call checkpointing**

Update imports:

```python
from ..verification.tool_call_verifier import verify_tool_call
```

Extend `update()`:

```python
        if self.config.kind == "tool_call":
            self._update_tool_call(text)
```

Add:

```python
    def _update_tool_call(self, text: str) -> None:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            self._remember_error(f"tool_call_parse:{exc.msg if hasattr(exc, 'msg') else exc}")
            return
        if not isinstance(data, dict):
            self._remember_error("tool_call:envelope_not_object")
            return
        function_name = data.get("function_name")
        arguments = data.get("arguments")
        if not isinstance(function_name, str) or not isinstance(arguments, dict):
            self._remember_error("tool_call:missing_function_name_or_arguments")
            return
        verdict = verify_tool_call(
            function_name=function_name,
            arguments=arguments,
            claimed_result=str(data.get("claimed_result", "")),
            manifest=self.config.tool_manifest,
            execution_log=(
                list(self.config.execution_log)
                if self.config.execution_log is not None
                else None
            ),
            score_fn=self.config.score_fn,
        )
        if (
            not verdict.function_exists
            or not verdict.arguments_valid
            or not verdict.result_plausible
            or verdict.fabrication_suspected
        ):
            self._remember_error(f"tool_call:invalid:{verdict.reason}")
            return
        self.last_valid_output = json.dumps(
            data,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.last_valid_metadata = {
            "function_name": function_name,
            "argument_count": len(arguments),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py tests/test_tool_call_verifier.py -q
```

Expected: recovery and existing tool-call verifier tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add src/director_ai/core/runtime/structured_recovery.py tests/test_structured_halt_recovery.py
git commit -m "Add tool-call structured halt recovery" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

---

### Task 4: Reasoning-Chain Checkpoint Recovery

**Files:**
- Modify: `src/director_ai/core/runtime/structured_recovery.py`
- Modify: `tests/test_structured_halt_recovery.py`

- [ ] **Step 1: Write failing reasoning-chain recovery tests**

Append:

```python
def test_reasoning_chain_recovery_keeps_last_valid_envelope() -> None:
    state = StructuredRecoveryState(
        StructuredRecoveryConfig(
            kind="reasoning_chain",
            reasoning_support_threshold=0.8,
        )
    )
    state.update(
        '{"steps":["The sky is blue","Therefore the sky has a visible colour"]}'
    )
    state.update('{"steps":["The sky is blue","Therefore')

    result = state.finalise(halted_at=7)

    assert result.valid is True
    assert result.last_valid_output == (
        '{"steps":["The sky is blue","Therefore the sky has a visible colour"]}'
    )
    assert result.metadata["steps_found"] >= 2


def test_reasoning_chain_invalid_update_keeps_previous_checkpoint() -> None:
    state = StructuredRecoveryState(StructuredRecoveryConfig(kind="reasoning_chain"))
    state.update('{"steps":["A implies B","Therefore B follows from A"]}')
    state.update('{"steps":["A implies B"]}')

    result = state.finalise(halted_at=4)

    assert result.valid is True
    assert "Therefore B follows" in result.last_valid_output
    assert any("reasoning" in error for error in result.errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py -q
```

Expected: reasoning tests fail because `kind="reasoning_chain"` has no parser path.

- [ ] **Step 3: Implement reasoning-chain checkpointing**

Update imports:

```python
from ..verification.reasoning_verifier import verify_reasoning_chain
```

Extend `update()`:

```python
        if self.config.kind == "reasoning_chain":
            self._update_reasoning_chain(text)
```

Add:

```python
    def _update_reasoning_chain(self, text: str) -> None:
        payload = text
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            steps = data["steps"]
            if not all(isinstance(step, str) and step.strip() for step in steps):
                self._remember_error("reasoning_chain:invalid_steps")
                return
            payload = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(steps))
        result = verify_reasoning_chain(
            payload,
            score_fn=self.config.score_fn,
            support_threshold=self.config.reasoning_support_threshold,
        )
        if result.steps_found < 2 or not result.chain_valid:
            self._remember_error("reasoning_chain:invalid")
            return
        self.last_valid_output = (
            json.dumps(data, separators=(",", ":"), sort_keys=True)
            if data is not None
            else text
        )
        self.last_valid_metadata = {
            "steps_found": result.steps_found,
            "issues_found": result.issues_found,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py tests/test_reasoning_verifier.py -q
```

Expected: recovery and existing reasoning verifier tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add src/director_ai/core/runtime/structured_recovery.py tests/test_structured_halt_recovery.py
git commit -m "Add reasoning-chain structured halt recovery" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

---

### Task 5: StreamingKernel Integration and Policies

**Files:**
- Modify: `src/director_ai/core/runtime/streaming.py`
- Modify: `tests/test_structured_halt_recovery.py`

- [ ] **Step 1: Write failing streaming integration tests**

Append:

```python
from director_ai.core.streaming import StreamingKernel


def test_streaming_json_recovery_attaches_last_valid_output_on_halt() -> None:
    kernel = StreamingKernel(hard_limit=0.5)
    config = StructuredRecoveryConfig(kind="json")
    scores = iter([0.8, 0.8, 0.2])

    session = kernel.stream_tokens(
        ['{"status":"ok"}', '{"status":', '"unsafe"'],
        lambda _text: next(scores),
        structured_recovery=config,
    )

    assert session.halted is True
    assert session.structured_recovery is not None
    assert session.structured_recovery.halted_at == session.halt_index
    assert session.structured_recovery.last_valid_output == '{"status":"ok"}'


def test_streaming_redacted_policy_suppresses_partial_payload_on_halt() -> None:
    kernel = StreamingKernel(hard_limit=0.5)
    config = StructuredRecoveryConfig(kind="json", policy="redacted")

    session = kernel.stream_tokens(
        ['{"status":"ok"}', '{"status":'],
        lambda text: 0.2 if text.endswith('{"status":') else 0.8,
        structured_recovery=config,
    )

    assert session.structured_recovery is not None
    assert session.structured_recovery.last_valid_output == ""
    assert session.structured_recovery.raw_partial == ""
    assert session.structured_recovery.valid is False


def test_streaming_raw_partial_policy_marks_payload_invalid() -> None:
    kernel = StreamingKernel(hard_limit=0.5)
    config = StructuredRecoveryConfig(kind="json", policy="raw_partial")

    session = kernel.stream_tokens(
        ['{"status":"ok"}', '{"status":'],
        lambda text: 0.2 if text.endswith('{"status":') else 0.8,
        structured_recovery=config,
    )

    assert session.structured_recovery is not None
    assert session.structured_recovery.last_valid_output == ""
    assert session.structured_recovery.raw_partial.endswith('{"status":')
    assert session.structured_recovery.valid is False


def test_unconfigured_streaming_recovery_preserves_plain_text_behaviour() -> None:
    kernel = StreamingKernel(hard_limit=0.5)
    session = kernel.stream_tokens(["Good ", "Bad "], lambda text: 0.2 if "Bad" in text else 0.8)

    assert session.halted is True
    assert session.structured_recovery is None
    assert session.output == "Good "
    assert kernel.stream_output(["Bad "], lambda _text: 0.2).startswith("[KERNEL INTERRUPT:")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py -q
```

Expected: `stream_tokens()` rejects the unexpected `structured_recovery` keyword or `StreamSession` has no field.

- [ ] **Step 3: Integrate recovery into StreamingKernel**

Update `streaming.py` imports:

```python
from .structured_recovery import (
    StructuredRecoveryConfig,
    StructuredRecoveryResult,
    StructuredRecoveryState,
)
```

Add to `StreamSession`:

```python
    structured_recovery: StructuredRecoveryResult | None = None
```

Add `structured_recovery` parameter to `stream_tokens()`:

```python
        structured_recovery: StructuredRecoveryConfig | None = None,
```

Create state after `session`:

```python
        recovery_state = (
            StructuredRecoveryState(structured_recovery)
            if structured_recovery is not None
            else None
        )
```

After `session.tokens.append(token)`, update the recovery state:

```python
            if recovery_state is not None:
                recovery_state.update("".join(session.tokens))
```

Inside `_finalize_halt()` after `session.halt_reason = reason`, finalise:

```python
            if recovery_state is not None:
                session.structured_recovery = recovery_state.finalise(
                    halted_at=session.halt_index,
                )
```

- [ ] **Step 4: Run integration tests to verify they pass**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py tests/test_streaming.py tests/test_coverage_streaming.py -q
```

Expected: structured recovery and existing sync streaming tests pass.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add src/director_ai/core/runtime/streaming.py tests/test_structured_halt_recovery.py
git commit -m "Wire structured recovery into streaming halts" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

---

### Task 6: Public Exports

**Files:**
- Modify: `src/director_ai/core/__init__.py`
- Modify: `src/director_ai/__init__.py`
- Modify: `tests/test_structured_halt_recovery.py`

- [ ] **Step 1: Write failing public export tests**

Append:

```python
def test_structured_recovery_types_are_public_core_exports() -> None:
    from director_ai.core import StructuredRecoveryConfig as CoreConfig
    from director_ai.core import StructuredRecoveryResult as CoreResult

    assert CoreConfig(kind="json").kind == "json"
    assert CoreResult(kind="json", policy="redacted", halted_at=1).halted_at == 1


def test_structured_recovery_types_are_lazy_package_exports() -> None:
    from director_ai import StructuredRecoveryConfig as PackageConfig
    from director_ai import StructuredRecoveryResult as PackageResult

    assert PackageConfig(kind="json").policy == "last_valid"
    assert PackageResult(kind="json", policy="redacted", halted_at=1).policy == "redacted"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py::test_structured_recovery_types_are_public_core_exports tests/test_structured_halt_recovery.py::test_structured_recovery_types_are_lazy_package_exports -q
```

Expected: import errors for public exports.

- [ ] **Step 3: Add exports**

In `src/director_ai/core/__init__.py`, import:

```python
from .runtime.structured_recovery import (
    StructuredRecoveryConfig,
    StructuredRecoveryResult,
)
```

Add to `__all__` near runtime symbols:

```python
    "StructuredRecoveryConfig",
    "StructuredRecoveryResult",
```

In `src/director_ai/__init__.py`, add lazy imports:

```python
    "StructuredRecoveryConfig": (".core", "StructuredRecoveryConfig"),
    "StructuredRecoveryResult": (".core", "StructuredRecoveryResult"),
```

- [ ] **Step 4: Run export tests**

Run:

```bash
./.venv/bin/pytest tests/test_structured_halt_recovery.py -q
```

Expected: all structured halt recovery tests pass.

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add src/director_ai/core/__init__.py src/director_ai/__init__.py tests/test_structured_halt_recovery.py
git commit -m "Expose structured recovery public types" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

---

### Task 7: Cookbook and Roadmap Closure

**Files:**
- Create: `docs-site/cookbook/halt-recovery-patterns.md`
- Modify: `mkdocs.yml`
- Modify: `ROADMAP.md`
- Test: existing docs build or targeted docs navigation check

- [ ] **Step 1: Write the cookbook page**

Create `docs-site/cookbook/halt-recovery-patterns.md`:

```markdown
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Halt Recovery Patterns

# Halt Recovery Patterns

Structured recovery is for parser-facing streams where a halt must not leave
downstream systems trying to parse unsafe partial text. Plain text streams keep
the normal interrupt behaviour unless a caller passes `StructuredRecoveryConfig`.

## Parser-safe recovery

Use `last_valid` when the downstream system expects JSON, tool-call envelopes,
or reasoning-chain envelopes:

```python
from director_ai.core import StreamingKernel, StructuredRecoveryConfig

kernel = StreamingKernel(hard_limit=0.5)
session = kernel.stream_tokens(
    token_generator,
    coherence_callback=score,
    structured_recovery=StructuredRecoveryConfig(
        kind="json",
        policy="last_valid",
        json_schema={"type": "object"},
    ),
)

if session.structured_recovery and session.structured_recovery.valid:
    payload = session.structured_recovery.last_valid_output
```

The recovery layer never completes braces, invents tool arguments, or generates
replacement reasoning. It only returns a checkpoint that was valid before halt.

## KB refresh

Use this when halt evidence points to stale or missing facts:

- inspect `session.safety_events[-1].evidence_refs`
- refresh or retract the source facts
- rerun the halted prompt in dry-run mode
- only lower thresholds after the source state is verified

## Threshold rollback

Use this when a recent tuning profile increases false halts:

- restore the previous profile overlay
- keep `structured_recovery.policy="last_valid"` while validating the rollback
- compare halt rate and false-positive feedback before promoting the profile

## Human review routing

Use `redacted` for regulated review queues where partial text must not be shown
until an operator approves evidence release. The reviewer receives halt metadata,
the token offset, and the safety event ID instead of the raw partial payload.

## Temporary policy fallback

Use temporary fallback only with expiry:

- switch high-risk structured endpoints to `redacted`
- route requests to human review or asynchronous completion
- keep the hard halt active
- remove the fallback once KB or threshold repair is validated
```

- [ ] **Step 2: Add docs navigation if required**

If `mkdocs.yml` lists cookbook pages explicitly, add:

```yaml
      - Halt Recovery Patterns: cookbook/halt-recovery-patterns.md
```

near the existing streaming halt cookbook entry. If `mkdocs.yml` uses automatic discovery, do not modify it.

- [ ] **Step 3: Mark roadmap items complete**

In `ROADMAP.md`, change the four unchecked items under "Halt Recovery and Structured Stream Resilience" to checked:

```markdown
- [x] Publish a halt recovery patterns cookbook covering KB refresh, threshold
  rollback, human review routing, and temporary policy fallback.
- [x] Add configurable `partial_output_on_halt` handling for JSON, tool-call, and
  reasoning-chain streams.
- [x] Emit the last valid structured chunk plus a `halted_at` marker so downstream
  parsers can recover cleanly from mid-stream halts.
- [x] Add structured-stream recovery tests for JSON objects, arrays, tool calls,
  nested tool arguments, and reasoning-chain envelopes.
```

- [ ] **Step 4: Run docs checks**

Run:

```bash
./.venv/bin/mkdocs build --strict
```

Expected: docs build succeeds. If this is too broad for local resources, run the repository's closest existing docs/nav test and record the limitation.

- [ ] **Step 5: Commit Task 7**

Run:

```bash
git add docs-site/cookbook/halt-recovery-patterns.md mkdocs.yml ROADMAP.md
git commit -m "Document structured halt recovery operations" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

---

### Task 8: Final Verification and Publication Readiness

**Files:**
- Verify all touched files
- No production code changes beyond planned files

- [ ] **Step 1: Run focused lint**

Run:

```bash
./.venv/bin/ruff check \
  src/director_ai/core/runtime/structured_recovery.py \
  src/director_ai/core/runtime/streaming.py \
  src/director_ai/core/__init__.py \
  src/director_ai/__init__.py \
  tests/test_structured_halt_recovery.py
```

Expected: `All checks passed!`

- [ ] **Step 2: Run focused format check**

Run:

```bash
./.venv/bin/ruff format --check \
  src/director_ai/core/runtime/structured_recovery.py \
  src/director_ai/core/runtime/streaming.py \
  src/director_ai/core/__init__.py \
  src/director_ai/__init__.py \
  tests/test_structured_halt_recovery.py
```

Expected: all files already formatted.

- [ ] **Step 3: Run focused tests**

Run:

```bash
./.venv/bin/pytest \
  tests/test_structured_halt_recovery.py \
  tests/test_streaming.py \
  tests/test_coverage_streaming.py \
  tests/test_json_verifier.py \
  tests/test_tool_call_verifier.py \
  tests/test_reasoning_verifier.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Run staged-diff policy checks before final commit or push**

Run:

```bash
git diff --check
git diff --cached --check
git status --short --branch
```

Expected: no whitespace errors; only intended tracked files changed.

- [ ] **Step 5: Record SNN stimulus**

Create a new timestamped stimulus after final verification:

```bash
date +%s
```

Then add a new file under `04_ARCANE_SAPIENCE/snn_stimuli/codex_<timestamp>.json` with:

```json
{"text":"DIRECTOR-AI implemented opt-in structured halt recovery for JSON, tool-call, and reasoning-chain streams with parser-safe policies, tests, docs, and roadmap closure.","source":"codex","project":"DIRECTOR-AI"}
```

Do not stage generated or unrelated existing files unless explicitly requested.

- [ ] **Step 6: Final commit if any verification-only docs/session files are intended**

If Task 8 adds only ignored SNN stimuli, do not create an extra git commit. If tracked files changed after Task 7, stage exact paths and commit:

```bash
git add <exact tracked paths>
git commit -m "Complete structured halt recovery verification" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

---

## Self-Review

Spec coverage:

- Opt-in only: Task 5 backwards compatibility test and `structured_recovery` optional parameter.
- Typed config: Tasks 1 and 6.
- Three policies: Task 5.
- JSON/tool/reasoning support: Tasks 2, 3, and 4.
- No auto-repair: Task 2 only checkpoints fully parseable JSON; docs state no synthesis.
- Cookbook: Task 7.
- Roadmap closure after implementation: Task 7.

Placeholder scan:

- No placeholder markers or incomplete implementation steps are present.
- Commands include expected results.
- File paths are explicit.

Type consistency:

- `StructuredRecoveryConfig`, `StructuredRecoveryResult`, and `StructuredRecoveryState` are consistently named.
- `StreamSession.structured_recovery` consistently stores `StructuredRecoveryResult | None`.
- Policies are consistently `last_valid`, `redacted`, and `raw_partial`.
