# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Input Normalisation Regression (KIMI2-J)
"""Regression tests for the KIMI2-J zero-width/confusable normalisation.

The GPU reproduction (2026-07-16) measured that a single zero-width space
(U+200B) inside an otherwise-true claim false-halts it — the invisible
character splits a word for the tokenizer and inflates the divergence. The NLI
inference layer now scrubs its text arguments before tokenisation. These tests
pin the scrub at the single ``_tokenize`` chokepoint and confirm the common
ASCII path is untouched.
"""

from __future__ import annotations

import threading
from typing import Any

from director_ai.core.scoring._nli_model_inference import (
    ModelInferenceMixin,
    _normalise_nli_arg,
    _normalise_nli_text,
)

_ZWSP = "​"


class TestNormaliseHelpers:
    def test_zero_width_space_is_stripped(self) -> None:
        assert _normalise_nli_text(f"app{_ZWSP}ears") == "appears"

    def test_plain_ascii_is_returned_unchanged(self) -> None:
        text = "Paris is the capital of France."
        assert _normalise_nli_text(text) is text

    def test_newlines_and_tabs_survive(self) -> None:
        assert _normalise_nli_text("a\nb\tc") == "a\nb\tc"

    def test_legitimate_non_ascii_letters_survive(self) -> None:
        # NFKC without confusable folding keeps real accented/non-Latin text.
        assert _normalise_nli_text("café") == "café"

    def test_arg_normalises_a_string(self) -> None:
        assert _normalise_nli_arg(f"Par{_ZWSP}is") == "Paris"

    def test_arg_normalises_a_list_of_strings(self) -> None:
        assert _normalise_nli_arg([f"a{_ZWSP}b", "cd"]) == ["ab", "cd"]

    def test_arg_passes_through_non_text(self) -> None:
        assert _normalise_nli_arg(42) == 42
        assert _normalise_nli_arg([]) == []
        assert _normalise_nli_arg([1, 2]) == [1, 2]


class _RecordingTokenizer:
    """Fake tokenizer that records the exact text it was handed."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((args, kwargs))
        return {"input_ids": [[0]]}


class _Holder(ModelInferenceMixin):
    """Minimal carrier exposing the mixin's ``_tokenize`` chokepoint."""

    def __init__(self, tokenizer: _RecordingTokenizer) -> None:
        self._tokenizer = tokenizer
        self._tokenizer_lock = threading.Lock()


class TestTokenizeChokepoint:
    def test_single_text_argument_is_scrubbed(self) -> None:
        tok = _RecordingTokenizer()
        holder = _Holder(tok)

        holder._tokenize(f"app{_ZWSP}ears", return_tensors="pt", truncation=True)

        assert tok.calls[0][0] == ("appears",)
        # kwargs are untouched.
        assert tok.calls[0][1] == {"return_tensors": "pt", "truncation": True}

    def test_premise_hypothesis_pair_is_scrubbed(self) -> None:
        tok = _RecordingTokenizer()
        holder = _Holder(tok)

        holder._tokenize(f"Par{_ZWSP}is", f"cap{_ZWSP}ital", padding=True)

        assert tok.calls[0][0] == ("Paris", "capital")

    def test_batched_lists_are_scrubbed(self) -> None:
        tok = _RecordingTokenizer()
        holder = _Holder(tok)

        holder._tokenize([f"a{_ZWSP}b", "cd"], ["ef", f"g{_ZWSP}h"])

        assert tok.calls[0][0] == (["ab", "cd"], ["ef", "gh"])
