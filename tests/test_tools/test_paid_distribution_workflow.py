# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — paid distribution workflow contract

"""Verify the inactive paid index stays disconnected from release automation."""

from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "publish-paid.yml"
)


@pytest.mark.parametrize(
    "retired_route",
    ["pypi.remanentia.com", "PRIVATE_INDEX_PUBLISHER_TOKEN", "twine upload"],
)
def test_paid_workflow_cannot_publish_to_the_retired_index(retired_route: str) -> None:
    """Reject the old endpoint, credential binding and index upload command."""
    assert retired_route not in WORKFLOW.read_text(encoding="utf-8")
