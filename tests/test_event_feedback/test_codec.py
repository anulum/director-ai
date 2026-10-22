# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Event feedback protocol tests
"""Exercise strict public JSON decoding without normalising original identities."""

from __future__ import annotations

import pytest

from director_ai.agentic.event_feedback.codec import (
    FeedbackInvalidError,
    Json,
    array,
    canonical,
    decode,
    object_value,
    text,
)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"null",
        b"[]",
        b'{"a":1,"a":2}',
        b'{"a":{"b":1,"b":2}}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'{"x":1e999}',
        b'{"x":"\xff"}',
        b'{"x":"\\ud800"}',
        b'{"\\ud800":1}',
        b'{"nested":{"x":["\\udfff"]}}',
        b"{broken",
    ],
)
def test_decode_rejects_ambiguous_originals(raw: bytes) -> None:
    """Refuse malformed, duplicate, nonfinite or nonobject original bytes."""
    with pytest.raises(FeedbackInvalidError, match="could not be validated"):
        decode(raw)


def test_canonical_preserves_original_values() -> None:
    """Preserve whitespace, Unicode and numeric types inside member values."""
    raw = '{"z":"  é  ","a":false,"b":1.5}'.encode()
    assert canonical(decode(raw)) == '{"a":false,"b":1.5,"z":"  é  "}'.encode()
    assert text(" value ") == " value "
    assert object_value({"nested": []}) == {"nested": []}
    assert array([True, 1]) == [True, 1]
    assert canonical(decode(b'{"emoji":"\\ud83d\\ude00"}')) == '{"emoji":"😀"}'.encode()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "\ud800"])
def test_unencodable_values_refuse(value: Json) -> None:
    """Refuse lossy or nonfinite canonical representations."""
    with pytest.raises(FeedbackInvalidError):
        canonical(value)


@pytest.mark.parametrize("value", [None, False, 1, [], {}, " "])
def test_string_identity_is_not_coerced(value: Json) -> None:
    """Reject nonstring and blank identities."""
    with pytest.raises(FeedbackInvalidError):
        text(value)


@pytest.mark.parametrize("value", [None, False, "{}", []])
def test_mapping_shape_is_not_coerced(value: Json) -> None:
    """Reject alternate shapes instead of converting them."""
    with pytest.raises(FeedbackInvalidError):
        object_value(value)


@pytest.mark.parametrize("value", [None, False, "[]", {}])
def test_array_shape_is_not_coerced(value: Json) -> None:
    """Reject alternate iterable or scalar shapes."""
    with pytest.raises(FeedbackInvalidError):
        array(value)
