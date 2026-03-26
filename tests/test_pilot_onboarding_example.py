# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "pilot_onboarding.py"
    spec = importlib.util.spec_from_file_location("pilot_onboarding_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_store_retrieves_relevant_context():
    module = _load_example_module()
    store = module.build_demo_store()

    context = store.retrieve_context("How much does the Team plan cost?")

    assert context is not None
    assert "$19/user/month" in context
