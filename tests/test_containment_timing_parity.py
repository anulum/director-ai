# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - containment timing parity tests

from __future__ import annotations

import hmac
from hashlib import sha256
from pathlib import Path

import pytest

from director_ai.core.containment import anchor as anchor_mod
from director_ai.core.containment.anchor import ContainmentAttestor

ROOT = Path(__file__).resolve().parents[1]
KEY = b"k" * 32


def _anchor_mac(key: bytes, payload: bytes) -> str:
    return hmac.new(key, payload, sha256).hexdigest()


def _flip_hex_digit(value: str) -> str:
    return "1" if value == "0" else "0"


def test_python_fallback_uses_fixed_length_compare_digest(monkeypatch) -> None:
    attestor = ContainmentAttestor(
        key=KEY,
        issuer="host",
        clock=lambda: 1_700_000_000,
    )
    anchor = attestor.mint(session_id="session", scope="sandbox")
    calls: list[tuple[int, int]] = []

    def fake_compare(left: str, right: str) -> bool:
        calls.append((len(left), len(right)))
        return left == right

    monkeypatch.setattr(anchor_mod, "_rust_anchor_mac", None)
    monkeypatch.setattr(anchor_mod.hmac, "compare_digest", fake_compare)

    assert anchor_mod._verify_anchor_mac(KEY, anchor.canonical_payload, anchor.mac)
    assert calls == [(64, 64)]


def test_rust_and_python_mac_paths_make_same_decisions(monkeypatch) -> None:
    attestor = ContainmentAttestor(
        key=KEY,
        issuer="host",
        clock=lambda: 1_700_000_000,
    )
    anchor = attestor.mint(session_id="session", scope="sandbox")
    mac_cases = [
        anchor.mac,
        _flip_hex_digit(anchor.mac[0]) + anchor.mac[1:],
        anchor.mac[:-1] + _flip_hex_digit(anchor.mac[-1]),
        "0" * 10,
    ]

    monkeypatch.setattr(anchor_mod, "_rust_anchor_mac", None)
    fallback = [
        anchor_mod._verify_anchor_mac(KEY, anchor.canonical_payload, mac)
        for mac in mac_cases
    ]

    def rust_stub(key: bytes, payload: bytes, mac_hex: str) -> bool:
        expected = _anchor_mac(key, payload)
        return len(mac_hex) == 64 and hmac.compare_digest(expected, mac_hex)

    monkeypatch.setattr(anchor_mod, "_rust_anchor_mac", rust_stub)
    accelerated = [
        anchor_mod._verify_anchor_mac(KEY, anchor.canonical_payload, mac)
        for mac in mac_cases
    ]

    assert accelerated == fallback == [True, False, False, False]


def test_rust_mac_exception_is_mandatory_failure(monkeypatch) -> None:
    attestor = ContainmentAttestor(
        key=KEY,
        issuer="host",
        clock=lambda: 1_700_000_000,
    )
    anchor = attestor.mint(session_id="session", scope="sandbox")

    monkeypatch.setattr(
        anchor_mod,
        "_rust_anchor_mac",
        lambda _key, _payload, _mac: (_ for _ in ()).throw(RuntimeError("ffi fail")),
    )

    with pytest.raises(RuntimeError, match="ffi"):
        anchor_mod._verify_anchor_mac(KEY, anchor.canonical_payload, anchor.mac)


def test_rust_mac_non_runtime_exception_is_mandatory_failure(monkeypatch) -> None:
    attestor = ContainmentAttestor(
        key=KEY,
        issuer="host",
        clock=lambda: 1_700_000_000,
    )
    anchor = attestor.mint(session_id="session", scope="sandbox")

    monkeypatch.setattr(
        anchor_mod,
        "_rust_anchor_mac",
        lambda _key, _payload, _mac: (_ for _ in ()).throw(ValueError("ffi fail")),
    )

    with pytest.raises(ValueError, match="ffi"):
        anchor_mod._verify_anchor_mac(KEY, anchor.canonical_payload, anchor.mac)


def test_rust_mac_arithmetic_exception_is_mandatory_failure(monkeypatch) -> None:
    attestor = ContainmentAttestor(
        key=KEY,
        issuer="host",
        clock=lambda: 1_700_000_000,
    )
    anchor = attestor.mint(session_id="session", scope="sandbox")

    monkeypatch.setattr(
        anchor_mod,
        "_rust_anchor_mac",
        lambda _key, _payload, _mac: (_ for _ in ()).throw(ArithmeticError("ffi fail")),
    )

    with pytest.raises(ArithmeticError, match="ffi"):
        anchor_mod._verify_anchor_mac(KEY, anchor.canonical_payload, anchor.mac)


def test_rust_mac_type_error_is_mandatory_failure(monkeypatch) -> None:
    attestor = ContainmentAttestor(
        key=KEY,
        issuer="host",
        clock=lambda: 1_700_000_000,
    )
    anchor = attestor.mint(session_id="session", scope="sandbox")

    monkeypatch.setattr(
        anchor_mod,
        "_rust_anchor_mac",
        lambda _key, _payload, _mac: (_ for _ in ()).throw(TypeError("ffi fail")),
    )

    with pytest.raises(TypeError, match="ffi"):
        anchor_mod._verify_anchor_mac(KEY, anchor.canonical_payload, anchor.mac)


def test_rust_containment_mac_helper_is_registered() -> None:
    source = (
        ROOT / "backfire-kernel" / "crates" / "backfire-ffi" / "src" / "safety_hooks.rs"
    ).read_text(encoding="utf-8")

    assert "rust_verify_reality_anchor_mac" in source
    assert "fixed_eq_32" in source
    assert ".fold(0_u8" in source
    assert "wrap_pyfunction!(rust_verify_reality_anchor_mac" in source
