# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - physical extra boundary tests

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_physical_extra_declares_pip_runtime_boundary() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    physical = pyproject["project"]["optional-dependencies"]["physical"]

    assert physical == ["mujoco>=3.2,<4"]


def test_adapter_install_hints_use_single_physical_extra() -> None:
    adapters = (
        ROOT / "src" / "director_ai" / "core" / "cyber_physical" / "adapters.py"
    ).read_text()

    assert "director-ai[physical]" in adapters
    assert "director-ai[ros2]" not in adapters
    assert "director-ai[mujoco]" not in adapters
    assert "director-ai[carla]" not in adapters


def test_physical_docs_cover_vendor_runtime_isolation() -> None:
    docs = (ROOT / "docs-site" / "api" / "cyber-physical.md").read_text()

    assert "director-ai[physical]" in docs
    assert "ROS 2" in docs
    assert "CARLA" in docs
    assert "isolated physical runtime" in docs
