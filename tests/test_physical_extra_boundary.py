# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# DIRECTOR-AI — physical extra boundary contracts

"""Check physical-extra declarations, installation hints and public vendor guidance."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_physical_extra_declares_pip_runtime_boundary() -> None:
    """Declare only the pip-installable MuJoCo dependency in the physical extra."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    physical = pyproject["project"]["optional-dependencies"]["physical"]

    assert physical == ["mujoco>=3.2,<4"]


def test_adapter_install_hints_use_single_physical_extra() -> None:
    """Use the unified physical extra in the actual adapter installation hints."""
    adapters = (
        ROOT / "src" / "director_ai" / "core" / "cyber_physical" / "adapters.py"
    ).read_text()

    assert "director-ai[physical]" in adapters
    assert "director-ai[ros2]" not in adapters
    assert "director-ai[mujoco]" not in adapters
    assert "director-ai[carla]" not in adapters


def test_physical_docs_cover_vendor_runtime_isolation() -> None:
    """Document the physical extra and isolated ROS 2 and CARLA vendor runtimes."""
    docs = (ROOT / "docs-site" / "api" / "cyber-physical.md").read_text()

    assert "director-ai[physical]" in docs
    assert "ROS 2" in docs
    assert "CARLA" in docs
    assert "isolated physical runtime" in docs
