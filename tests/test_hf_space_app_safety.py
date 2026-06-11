# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Hugging Face Space app safety tests

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "demo" / "app.py"


def _load_app_module() -> types.ModuleType:
    fake_gradio = types.ModuleType("gradio")
    fake_gradio_module: Any = fake_gradio
    fake_gradio_module.Blocks = object
    fake_gradio_module.themes = types.SimpleNamespace(Soft=lambda: object())
    monkey_modules = {"gradio": fake_gradio}
    original_modules = {name: sys.modules.get(name) for name in monkey_modules}
    sys.modules.update(monkey_modules)
    try:
        spec = importlib.util.spec_from_file_location("director_ai_hf_space_app", APP)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_score_response_escapes_user_controlled_markdown_surfaces() -> None:
    module = _load_app_module()
    malicious_fact = '<img src=x onerror="alert(1)">'

    _badge, details, _bar, context = module.score_response(
        f"payload: {malicious_fact}",
        "payload",
        malicious_fact,
    )

    assert malicious_fact not in details
    assert malicious_fact not in context
    assert "&lt;img" in context
