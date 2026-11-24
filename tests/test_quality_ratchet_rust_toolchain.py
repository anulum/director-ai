# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the native quality job's pinned rustup and Rust reader inputs
"""Pin the rustup release and the reader's dependencies the rust-docs capture needs.

The rust-docs identity hashes the ``cargo`` and ``rustc`` executables the ratchet
runs. Through rustup those are its proxies, which are the rustup binary itself, so
the required job must install the exact rustup release the baseline was recorded
with. The reader builds with ``--offline``, so the job must fetch its locked
dependencies first.
"""

from __future__ import annotations

import re
from pathlib import Path

from tools.audit_ci_workflow_modularity import read_workflow
from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_rust import RUST_VERSION

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "tools/install_rustup.sh"
INSTALL = "bash tools/install_rustup.sh"
FETCH = "cargo fetch --locked --manifest-path tools/quality_ratchet_rust/Cargo.toml"


def _job() -> list[dict[str, object]]:
    jobs = object_mapping(
        read_workflow(ROOT / ".github/workflows/ci-quality.yml")["jobs"]
    )
    raw = object_mapping(jobs["native-quality"])["steps"]
    assert isinstance(raw, list)
    return [object_mapping(step) for step in raw]


def _assignment(name: str) -> str:
    match = re.search(rf"^{name}=(\S+)$", INSTALLER.read_text(encoding="utf-8"), re.M)
    assert match is not None, name
    return match.group(1)


def test_the_quality_job_installs_the_pin_then_fetches_before_the_ratchet() -> None:
    """Install the pinned rustup, fetch the reader's lock, then run the ratchet."""
    job = _job()
    uses = [str(step.get("uses", "")) for step in job]
    assert not any("rust-toolchain" in use for use in uses)
    install = [i for i, step in enumerate(job) if step.get("run") == INSTALL]
    fetch = [i for i, step in enumerate(job) if step.get("run") == FETCH]
    ratchet = [
        i for i, step in enumerate(job) if '"$QUALITY_PYTHON"' in str(step.get("run"))
    ]
    assert len(install) == 1 and len(fetch) == 1 and ratchet
    assert install[0] < fetch[0] < min(ratchet)
    for index in (install[0], fetch[0]):
        assert "if" not in job[index]
        assert not job[index].get("continue-on-error", False)


def test_the_installer_pins_the_reader_toolchain_and_one_rustup_release() -> None:
    """Install the rust-docs toolchain under one verified rustup release."""
    assert _assignment("toolchain") == RUST_VERSION
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", _assignment("rustup_version"))
    assert re.fullmatch(r"[0-9a-f]{64}", _assignment("rustup_sha256"))
    assert int(_assignment("rustup_bytes")) > 0


def test_the_installer_refuses_unless_every_proxy_is_the_verified_binary() -> None:
    """Check the download and then each installed proxy against the one digest."""
    text = INSTALLER.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert "https://static.rust-lang.org/rustup/archive/${rustup_version}/" in text
    assert text.count("sha256sum --check --strict -") == 2
    assert re.search(
        r"for proxy in rustup cargo rustc; do\n\s+printf '%s  %s\\n' \"\$rustup_sha256\"",
        text,
    )
    assert "--default-toolchain none" in text
    assert '"${cargo_home}/bin/rustup" default "$toolchain"' in text
    assert "|| true" not in text
