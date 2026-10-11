# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Licence file input tests
"""Exercise malformed licence files through the public validation surfaces."""

import hashlib
import hmac
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from director_ai.core.license import (
    LicenseError,
    enforce_capability_tier,
    generate_license,
    load_license,
    validate_file,
)


@pytest.mark.parametrize("content", [b"[]", b"null", b"true", b"1", b'"licence"'])
def test_nonobject_json_returns_invalid(tmp_path: Path, content: bytes) -> None:
    """Valid JSON without an object root returns an invalid licence result.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the licence file read by validation.
    content : bytes
        UTF-8 JSON document whose root is not an object.
    """
    path = tmp_path / "licence.json"
    path.write_bytes(content)
    info = validate_file(path)
    assert not info.valid
    assert info.tier == "community"
    assert info.message == "License file must contain a JSON object"


def test_invalid_utf8_returns_invalid(tmp_path: Path) -> None:
    """An encoding error is reported without escaping the public API.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the licence file read by validation.
    """
    path = tmp_path / "licence.json"
    path.write_bytes(b"\xff")
    info = validate_file(path)
    assert not info.valid
    assert info.message == "Cannot read license file"


@pytest.mark.parametrize("content", [b"[]", b"\xff"])
def test_load_license_recovers_from_invalid_file(
    tmp_path: Path, content: bytes
) -> None:
    """The real loader recovers from shape and encoding failures in a subprocess.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the licence file read by validation.
    content : bytes
        Non-object JSON or invalid UTF-8 read by the actual child-process loader.
    """
    assert not (Path.home() / ".director-ai/license.json").exists(), (
        "Run this loader check without an operator's default licence"
    )
    path = tmp_path / "licence.json"
    path.write_bytes(content)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DIRECTOR_LICENSE_") and key != "DIRECTOR_AI_POLAR_ORG_ID"
    }
    environment["DIRECTOR_LICENSE_FILE"] = str(path)
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import json, sys; sys.dont_write_bytecode=True; "
        "sys.path.insert(0,sys.argv[1]); "
        "from director_ai.core.license import load_license; "
        "info=load_license(); "
        "print(json.dumps({'valid':info.valid,'tier':info.tier}))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"valid": True, "tier": "community"}


_INVALID_STRING_VALUES: tuple[object, ...] = (
    None,
    7,
    ["invalid"],
    {"invalid": True},
)
_INVALID_DEPLOYMENT_VALUES: tuple[object, ...] = (None, True, "2", 2.5, [], {})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in (
            "key",
            "tier",
            "licensee",
            "email",
            "issued",
            "expires",
            "signature",
            "ed25519_signature",
        )
        for value in _INVALID_STRING_VALUES
    ]
    + [("deployments", value) for value in _INVALID_DEPLOYMENT_VALUES],
)
def test_signed_file_rejects_wrong_field_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    """Malformed signed fields cannot crash validation or become active licences.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the licence file read by validation.
    monkeypatch : pytest.MonkeyPatch
        Fixture setting the synthetic signing and licence environment for this case.
    field : str
        Licence field replaced in the otherwise valid signed JSON payload.
    value : object
        Incorrectly typed JSON value supplied for that field.
    """
    secret = "licence-input-regression-only"
    monkeypatch.setenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC", "1")
    monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", secret)
    monkeypatch.setenv("DIRECTOR_LICENSE_PRIVATE_KEY", "")
    payload: dict[str, object] = dict(
        generate_license("pro", "Input regression", "licence@example.invalid")
    )
    payload[field] = value
    if field != "signature":
        unsigned = {key: item for key, item in payload.items() if key != "signature"}
        payload["signature"] = hmac.new(
            secret.encode(),
            json.dumps(unsigned, sort_keys=True).encode(),
            hashlib.sha256,
        ).hexdigest()
    path = tmp_path / "malformed-signed-licence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    info = validate_file(path)
    assert not info.valid
    assert info.tier == "community"
    assert info.message == f"Invalid type for license field: {field}"

    assert not (Path.home() / ".director-ai/license.json").exists(), (
        "Run this loader check without an operator's default licence"
    )
    monkeypatch.setenv("DIRECTOR_LICENSE_KEY", "")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_ORG_ID", "")
    monkeypatch.setenv("DIRECTOR_LICENSE_FILE", str(path))
    loaded = load_license()
    assert loaded.valid
    assert loaded.tier == "community"

    monkeypatch.setenv("DIRECTOR_ENFORCE_LICENSE_TIER", "1")
    with pytest.raises(LicenseError, match="community"):
        enforce_capability_tier("sector_policy")


def test_directory_is_not_a_licence_file(tmp_path: Path) -> None:
    """The real filesystem's directory read error returns an invalid result.

    Parameters
    ----------
    tmp_path : Path
        Directory passed as the licence path to exercise its native read failure.
    """
    info = validate_file(tmp_path)
    assert not info.valid
    assert info.message == "Cannot read license file"


def test_bad_asymmetric_signature_cannot_downgrade_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real valid legacy signature cannot rescue a bad asymmetric signature.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the licence file read by validation.
    monkeypatch : pytest.MonkeyPatch
        Fixture opting the synthetic signer into legacy migration for the comparison.
    """
    monkeypatch.setenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC", "1")
    secret = "licence-downgrade-regression-only"
    monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", secret)
    monkeypatch.setenv("DIRECTOR_LICENSE_PRIVATE_KEY", "")
    payload = generate_license("pro", "Input regression", "licence@example.invalid")
    path = tmp_path / "licence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_file(path).valid
    payload["ed25519_signature"] = "00"
    unsigned = {key: item for key, item in payload.items() if key != "signature"}
    payload["signature"] = hmac.new(
        secret.encode(), json.dumps(unsigned, sort_keys=True).encode(), hashlib.sha256
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    rejected = validate_file(path)
    assert not rejected.valid
    assert rejected.tier == "community"
    assert rejected.message


def test_integer_conversion_limit_returns_invalid(tmp_path: Path) -> None:
    """The real interpreter's JSON integer limit cannot escape validation.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the licence file read by validation.
    """
    limit = sys.get_int_max_str_digits()
    assert limit > 0, "Run this check with Python's integer conversion limit enabled"
    path = tmp_path / "oversized-integer.json"
    path.write_text('{"deployments":' + "9" * (limit + 1) + "}", encoding="utf-8")
    info = validate_file(path)
    assert not info.valid
    assert info.tier == "community"
    assert info.message == "Cannot read license file"


@pytest.mark.parametrize("enabled", [False, True])
def test_real_licence_file_enforces_capability_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """The real file-to-loader-to-gate chain honours rank and enforcement input.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the licence file read by validation.
    monkeypatch : pytest.MonkeyPatch
        Fixture setting the synthetic signing and licence environment for this case.
    enabled : bool
        Whether capability enforcement is enabled for the valid signed pro file.
    """
    assert not (Path.home() / ".director-ai/license.json").exists(), (
        "Run this loader check without an operator's default licence"
    )
    monkeypatch.setenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC", "1")
    monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", "licence-gate-regression-only")
    monkeypatch.setenv("DIRECTOR_LICENSE_PRIVATE_KEY", "")
    monkeypatch.setenv("DIRECTOR_LICENSE_KEY", "")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_ORG_ID", "")
    monkeypatch.setenv("DIRECTOR_ENFORCE_LICENSE_TIER", "1" if enabled else "0")
    payload = generate_license("pro", "Gate regression", "licence@example.invalid")
    path = tmp_path / "licence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("DIRECTOR_LICENSE_FILE", str(path))
    loaded = load_license()
    assert loaded.valid
    assert loaded.tier == "pro"
    enforce_capability_tier("sector_policy", minimum="pro")
    if enabled:
        with pytest.raises(LicenseError, match="enterprise"):
            enforce_capability_tier("enterprise_policy", minimum="enterprise")
    else:
        enforce_capability_tier("enterprise_policy", minimum="enterprise")


@pytest.mark.parametrize("signature", ["é", "😀", "\ud800", "\udfff"])
def test_nonascii_signature_cannot_crash_licence_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signature: str
) -> None:
    """Unicode signature inputs fail closed through the real licence chain.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the licence file read by validation.
    monkeypatch : pytest.MonkeyPatch
        Fixture setting the synthetic signing and licence environment for this case.
    signature : str
        Non-ASCII or surrogate text replacing the valid HMAC signature.
    """
    assert not (Path.home() / ".director-ai/license.json").exists(), (
        "Run this loader check without an operator's default licence"
    )
    monkeypatch.setenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC", "1")
    monkeypatch.setenv("DIRECTOR_LICENSE_SIGNING_KEY", "unicode-input-regression-only")
    monkeypatch.setenv("DIRECTOR_LICENSE_PRIVATE_KEY", "")
    monkeypatch.setenv("DIRECTOR_LICENSE_KEY", "")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_ORG_ID", "")
    payload = generate_license("pro", "Unicode regression", "licence@example.invalid")
    path = tmp_path / "unicode-signature.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_file(path).valid
    payload["signature"] = signature
    path.write_text(json.dumps(payload), encoding="utf-8")
    info = validate_file(path)
    assert not info.valid
    assert info.tier == "community"
    assert info.message == "Invalid license signature"
    monkeypatch.setenv("DIRECTOR_LICENSE_FILE", str(path))
    loaded = load_license()
    assert loaded.valid
    assert loaded.tier == "community"
    monkeypatch.setenv("DIRECTOR_ENFORCE_LICENSE_TIER", "1")
    with pytest.raises(LicenseError, match="community"):
        enforce_capability_tier("sector_policy")


@pytest.mark.parametrize("unsearchable", [False, True])
@pytest.mark.skipif(
    os.name != "posix",
    reason="Requires real POSIX user lookup and directory permissions",
)
def test_unresolvable_path_recovers_through_licence_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsearchable: bool
) -> None:
    """Native user lookup and directory permissions cannot crash the loader.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the licence file read by validation.
    monkeypatch : pytest.MonkeyPatch
        Fixture selecting the invalid file path and enabling capability enforcement.
    unsearchable : bool
        True selects denied directory traversal; False selects an unknown tilde user.
    """
    assert not (Path.home() / ".director-ai/license.json").exists(), (
        "Run this loader check without an operator's default licence"
    )
    parent = tmp_path / "restricted"
    parent.mkdir()
    path = parent / "licence.json"
    path.write_text("{}", encoding="utf-8")
    if unsearchable:
        assert os.geteuid() != 0, (
            "Run the real permission check as an unprivileged user"
        )
        parent.chmod(0)
        location = str(path)
    else:
        import pwd

        username = "director-licence-regression-no-such-user"
        with pytest.raises(KeyError):
            pwd.getpwnam(username)
        location = f"~{username}/licence.json"
    monkeypatch.setenv("DIRECTOR_LICENSE_KEY", "")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_ORG_ID", "")
    monkeypatch.setenv("DIRECTOR_LICENSE_FILE", location)
    monkeypatch.setenv("DIRECTOR_ENFORCE_LICENSE_TIER", "1")
    try:
        info = validate_file(location)
        assert not info.valid
        assert info.message == "Cannot read license file"
        loaded = load_license()
        assert loaded.valid
        assert loaded.tier == "community"
        with pytest.raises(LicenseError, match="community"):
            enforce_capability_tier("sector_policy")
    finally:
        parent.chmod(0o700)


@pytest.mark.parametrize("valid_default", [False, True])
def test_real_default_file_after_invalid_environment_inputs(
    tmp_path: Path, valid_default: bool
) -> None:
    """An isolated process uses actual HOME discovery and default-file fallback.

    Parameters
    ----------
    tmp_path : Path
        Directory holding the isolated child home and invalid explicit licence file.
    valid_default : bool
        Whether the child writes a signed pro licence at the real default location.
    """
    home = tmp_path / "home"
    home.mkdir()
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DIRECTOR_LICENSE_") and key != "DIRECTOR_AI_POLAR_ORG_ID"
    }
    environment.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "DIRECTOR_LICENSE_KEY": "invalid-bare-key",
            "DIRECTOR_LICENSE_FILE": str(invalid),
            "DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC": "1",
            "DIRECTOR_LICENSE_SIGNING_KEY": "default-file-regression-only",
            "DIRECTOR_ENFORCE_LICENSE_TIER": "1",
        }
    )
    source = Path(__file__).resolve().parents[1] / "src"
    code = """
import json, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage
    coverage.process_startup()
sys.path.insert(0, sys.argv[1])
from director_ai.core.license import (
    LicenseError, enforce_capability_tier, generate_license, load_license,
)
home = Path.home()
assert home == Path(sys.argv[2])
default = home / '.director-ai' / 'license.json'
default.parent.mkdir()
valid_default = sys.argv[3] == '1'
payload = generate_license('pro', 'Default regression', 'licence@example.invalid')
default.write_text(json.dumps(payload if valid_default else []), encoding='utf-8')
info = load_license()
assert info.valid
assert info.tier == ('pro' if valid_default else 'community')
if valid_default:
    enforce_capability_tier('sector_policy')
else:
    try:
        enforce_capability_tier('sector_policy')
    except LicenseError:
        pass
    else:
        raise AssertionError('Community licence unlocked a paid capability')
print(json.dumps({'valid': info.valid, 'tier': info.tier}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(source),
            str(home),
            str(int(valid_default)),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "valid": True,
        "tier": "pro" if valid_default else "community",
    }


def test_legacy_file_is_denied_without_explicit_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid legacy file cannot activate a paid tier under the default policy.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the licence file read by validation.
    monkeypatch : pytest.MonkeyPatch
        Fixture minting the test file under migration and then restoring default refusal.
    """
    assert not (Path.home() / ".director-ai/license.json").exists(), (
        "Run this loader check without an operator's default licence"
    )
    monkeypatch.setenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC", "1")
    monkeypatch.setenv(
        "DIRECTOR_LICENSE_SIGNING_KEY", "migration-policy-regression-only"
    )
    monkeypatch.setenv("DIRECTOR_LICENSE_PRIVATE_KEY", "")
    monkeypatch.setenv("DIRECTOR_LICENSE_KEY", "")
    monkeypatch.setenv("DIRECTOR_AI_POLAR_ORG_ID", "")
    payload = generate_license("pro", "Migration regression", "licence@example.invalid")
    path = tmp_path / "legacy-licence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_file(path).valid
    monkeypatch.delenv("DIRECTOR_LICENSE_ALLOW_LEGACY_HMAC")
    info = validate_file(path)
    assert not info.valid
    assert "rejected by default" in info.message
    monkeypatch.setenv("DIRECTOR_LICENSE_FILE", str(path))
    assert load_license().tier == "community"
    monkeypatch.setenv("DIRECTOR_ENFORCE_LICENSE_TIER", "1")
    with pytest.raises(LicenseError, match="community"):
        enforce_capability_tier("sector_policy")
    with pytest.raises(RuntimeError, match="PRIVATE_KEY not set"):
        generate_license("pro", "Migration regression", "licence@example.invalid")
