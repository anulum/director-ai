# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_requirements_directory_is_explicitly_marked_non_package_uv_project() -> None:
    """Keep GitHub's uv dependency grapher from failing on requirements/."""

    marker = ROOT / "requirements" / "pyproject.toml"
    data = tomllib.loads(marker.read_text(encoding="utf-8"))

    assert data["project"]["name"] == "director-ai-requirements-graph"
    assert data["project"]["dependencies"] == []
    assert data["tool"]["uv"]["package"] is False
