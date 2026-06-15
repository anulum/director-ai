# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — inference logits-processor adapter tests

"""Offline tests for the vLLM / TGI / llama.cpp logits-processor adapters.

No torch and no server SDK: logits are plain lists, the detokeniser is a fake
that returns a fixed string, and the score function is a keyword heuristic — so
the boundary gating, EOS-masking halt, sticky halt state, audit sink, and the
per-server factory guards are all verified deterministically.
"""

from __future__ import annotations

import pytest

from director_ai.integrations.inference_logits_adapters import (
    LogitsHaltProcessor,
    build_llama_cpp_logits_processor,
    build_tgi_logits_processor,
    build_vllm_logits_processor,
)
from director_ai.integrations.inference_server_hooks import build_inference_server_hook

_EOS = 2


def _decoder(text: str):
    """A detokeniser that ignores ids and returns a fixed decoded text."""
    return lambda _ids: text


def _hook(server: str = "vllm"):
    # Low score for text containing "bad" -> halts below the 0.4 hard limit.
    return build_inference_server_hook(
        server,
        score_fn=lambda t: 0.1 if "bad" in t.lower() else 0.9,
        hard_limit=0.4,
    )


def _logits(n: int = 6) -> list[float]:
    return [1.0] * n


# --------------------------------------------------------------------------- #
# boundary gating + halt                                                      #
# --------------------------------------------------------------------------- #


def test_non_boundary_text_passes_through_unchanged():
    proc = LogitsHaltProcessor(_hook(), _decoder("this is bad"), _EOS)  # no terminator
    logits = _logits()
    out = proc(token_ids=[1, 2, 3], logits=logits)
    assert out == [1.0] * 6  # not a claim boundary -> never scored, untouched
    assert proc.halted is False


def test_boundary_safe_text_is_allowed():
    proc = LogitsHaltProcessor(_hook(), _decoder("All is well."), _EOS)
    out = proc(token_ids=[1], logits=_logits())
    assert out == [1.0] * 6
    assert proc.halted is False


def test_boundary_unsafe_text_halts_via_eos_mask():
    proc = LogitsHaltProcessor(_hook(), _decoder("this claim is bad."), _EOS)
    out = proc(token_ids=[1], logits=_logits())
    assert out[_EOS] == 0.0
    assert all(out[i] == float("-inf") for i in range(len(out)) if i != _EOS)
    assert proc.halted is True


def test_halt_is_sticky_across_subsequent_steps():
    proc = LogitsHaltProcessor(_hook(), _decoder("this is bad."), _EOS)
    proc(token_ids=[1], logits=_logits())
    assert proc.halted is True
    # Even a now-"safe" decode stays halted once tripped.
    proc2_logits = _logits()
    out = proc(token_ids=[1, 2], logits=proc2_logits)
    assert out[_EOS] == 0.0
    assert all(out[i] == float("-inf") for i in range(len(out)) if i != _EOS)


def test_on_halt_sink_receives_safety_event():
    events = []
    proc = LogitsHaltProcessor(
        _hook(), _decoder("totally bad."), _EOS, on_halt=events.append
    )
    proc(token_ids=[1], logits=_logits())
    assert len(events) == 1
    assert events[0].hook_scope == "inference_server"


def test_evaluate_returns_decision_for_inspection():
    proc = LogitsHaltProcessor(_hook(), _decoder("this is bad."), _EOS)
    _out, decision = proc.evaluate([1], _logits())
    assert decision is not None
    assert decision.allow is False


# --------------------------------------------------------------------------- #
# validation                                                                  #
# --------------------------------------------------------------------------- #


def test_negative_eos_rejected():
    with pytest.raises(ValueError, match="eos_token_id"):
        LogitsHaltProcessor(_hook(), _decoder("x."), -1)


def test_eos_out_of_range_raises_on_halt():
    proc = LogitsHaltProcessor(_hook(), _decoder("bad."), 999)
    with pytest.raises(ValueError, match="out of range"):
        proc(token_ids=[1], logits=_logits())


# --------------------------------------------------------------------------- #
# per-server factory guards                                                   #
# --------------------------------------------------------------------------- #


def test_each_factory_accepts_its_own_server():
    assert isinstance(
        build_vllm_logits_processor(_hook("vllm"), _decoder("x."), _EOS),
        LogitsHaltProcessor,
    )
    assert isinstance(
        build_tgi_logits_processor(_hook("tgi"), _decoder("x."), _EOS),
        LogitsHaltProcessor,
    )
    assert isinstance(
        build_llama_cpp_logits_processor(_hook("llama_cpp"), _decoder("x."), _EOS),
        LogitsHaltProcessor,
    )


def test_factory_rejects_mismatched_server():
    with pytest.raises(ValueError, match="expected 'vllm'"):
        build_vllm_logits_processor(_hook("tgi"), _decoder("x."), _EOS)
    with pytest.raises(ValueError, match="expected 'llama_cpp'"):
        build_llama_cpp_logits_processor(_hook("vllm"), _decoder("x."), _EOS)
