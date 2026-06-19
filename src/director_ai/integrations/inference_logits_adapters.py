# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — vLLM / TGI / llama.cpp logits-processor adapters

"""Wire the server-neutral :class:`InferenceServerHook` into real logits processors.

vLLM, TGI (Hugging Face ``LogitsProcessor``), and llama.cpp all expose the same
per-step contract — ``(token_ids, logits) -> logits`` called before sampling — so
a single adapter drives all three. At a claim boundary the adapter decodes the
text generated so far, scores it through the hook, and on a halt decision masks
every logit except the end-of-sequence token, which forces the server to stop on
its next step. This makes the pre-sampling guard a drop-in for self-hosted serving
rather than a library that only operates on the output stream after the fact.

The adapter is framework-light: ``logits`` only needs ``len()``, integer indexing,
and item assignment, which a Python ``list``, a NumPy array, and a torch tensor
all satisfy — so neither torch nor a server SDK is needed to use or test it. The
detokeniser is injected (``decode_fn``), so the adapter never owns a tokenizer.

Wiring (vLLM)::

    from director_ai.integrations.inference_server_hooks import build_inference_server_hook
    from director_ai.integrations.inference_logits_adapters import build_vllm_logits_processor

    hook = build_inference_server_hook("vllm", score_fn=my_coherence_score, hard_limit=0.4)
    processor = build_vllm_logits_processor(hook, decode_fn=tokenizer.decode, eos_token_id=tok.eos_token_id)
    # vLLM: SamplingParams(logits_processors=[processor])
"""

from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from typing import Any

from director_ai.core.runtime.streaming_gate import ends_claim
from director_ai.core.safety_event import SafetyEvent

from .inference_server_hooks import (
    InferenceHookDecision,
    InferenceHookRequest,
    InferenceServerHook,
)

__all__ = [
    "LogitsHaltProcessor",
    "build_vllm_logits_processor",
    "build_tgi_logits_processor",
    "build_llama_cpp_logits_processor",
]

DecodeFn = Callable[[Sequence[int]], str]
ClaimGate = Callable[[str], bool]
HaltSink = Callable[[SafetyEvent], None]

_NEG_INF = float("-inf")


def _force_halt(
    logits: MutableSequence[float], eos_token_id: int
) -> MutableSequence[float]:
    """Mask every logit except EOS so the server samples EOS and stops."""
    if not 0 <= eos_token_id < len(logits):
        raise ValueError(
            f"eos_token_id {eos_token_id} out of range for vocab size {len(logits)}"
        )
    for i in range(len(logits)):
        logits[i] = _NEG_INF
    logits[eos_token_id] = 0.0
    return logits


class LogitsHaltProcessor:
    """Server-neutral logits processor that halts generation via EOS masking.

    Scores the decoded text-so-far at claim boundaries through an
    :class:`InferenceServerHook`; on a halt decision it masks all logits except
    ``eos_token_id``. Between claim boundaries (the common case) it passes the
    logits through untouched, so the steady-state cost is one cheap boundary check
    per token, not a model call.
    """

    def __init__(
        self,
        hook: InferenceServerHook,
        decode_fn: DecodeFn,
        eos_token_id: int,
        *,
        claim_gate: ClaimGate | None = ends_claim,
        request_id: str = "",
        tenant_id: str = "",
        on_halt: HaltSink | None = None,
    ) -> None:
        if eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")
        self._hook = hook
        self._decode = decode_fn
        self._eos = int(eos_token_id)
        self._claim_gate = claim_gate
        self._request_id = request_id
        self._tenant_id = tenant_id
        self._on_halt = on_halt
        self._halted = False

    @property
    def halted(self) -> bool:
        """True once a halt has fired; the stream stays halted thereafter."""
        return self._halted

    def evaluate(
        self, token_ids: Sequence[int], logits: MutableSequence[float]
    ) -> tuple[MutableSequence[float], InferenceHookDecision | None]:
        """Apply the guard to one step; returns the (possibly masked) logits."""
        if self._halted:
            return _force_halt(logits, self._eos), None
        text = self._decode(token_ids)
        if self._claim_gate is not None and not self._claim_gate(text):
            return logits, None
        decision = self._hook.check(
            InferenceHookRequest(
                server=self._hook.server,
                accumulated_text=text,
                candidate_token="",  # nosec B106 - LLM generation token, not a credential
                token_id=self._eos,
                request_id=self._request_id,
                tenant_id=self._tenant_id,
            )
        )
        if not decision.allow:
            self._halted = True
            if self._on_halt is not None and decision.safety_event is not None:
                self._on_halt(decision.safety_event)
            return _force_halt(logits, self._eos), decision
        return logits, decision

    def __call__(
        self, token_ids: Sequence[int], logits: MutableSequence[float]
    ) -> MutableSequence[float]:
        """Logits-processor entry point: ``(token_ids, logits) -> logits``."""
        adjusted, _decision = self.evaluate(token_ids, logits)
        return adjusted


def build_vllm_logits_processor(
    hook: InferenceServerHook,
    decode_fn: DecodeFn,
    eos_token_id: int,
    **kwargs: Any,
) -> LogitsHaltProcessor:
    """Build a vLLM ``logits_processors`` entry from a ``"vllm"`` hook."""
    _require_server(hook, "vllm")
    return LogitsHaltProcessor(hook, decode_fn, eos_token_id, **kwargs)


def build_tgi_logits_processor(
    hook: InferenceServerHook,
    decode_fn: DecodeFn,
    eos_token_id: int,
    **kwargs: Any,
) -> LogitsHaltProcessor:
    """Build a TGI / Hugging Face ``LogitsProcessor`` from a ``"tgi"`` hook."""
    _require_server(hook, "tgi")
    return LogitsHaltProcessor(hook, decode_fn, eos_token_id, **kwargs)


def build_llama_cpp_logits_processor(
    hook: InferenceServerHook,
    decode_fn: DecodeFn,
    eos_token_id: int,
    **kwargs: Any,
) -> LogitsHaltProcessor:
    """Build a llama-cpp-python ``logits_processor`` from a ``"llama_cpp"`` hook."""
    _require_server(hook, "llama_cpp")
    return LogitsHaltProcessor(hook, decode_fn, eos_token_id, **kwargs)


def _require_server(hook: InferenceServerHook, expected: str) -> None:
    if hook.server != expected:
        raise ValueError(f"hook is for server {hook.server!r}, expected {expected!r}")
