# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Task-scoring accelerator binding contracts

"""Contract tests for the canonical Rust task-scoring accelerator binding.

``director_ai.core.scoring._task_accel`` is the single module through
which the task-scoring paths resolve their Rust fast lane — the
task-lane sibling of ``_nli_accel``, with the two shared kernel
functions bound independently per lane. These tests pin that contract:
the flag and all four accelerator names live there, and patching the
binding module — one place — switches ``_task_scoring`` between the
accelerated branch and the pure-Python floor.
"""

from __future__ import annotations

import director_ai.core.scoring._task_accel as task_accel
import director_ai.core.scoring._task_scoring as task_scoring_mod
from director_ai.core.scoring._task_scoring import detect_task_type

_ACCEL_NAMES = (
    "rust_coverage_from_divergences",
    "rust_detect_task_type",
    "rust_split_sentences",
    "rust_sum_i64",
)


class TestBindingSurface:
    def test_flag_is_boolean_and_all_names_are_callable(self):
        assert isinstance(task_accel._RUST_TASK, bool)
        for name in _ACCEL_NAMES:
            assert callable(getattr(task_accel, name))

    def test_module_all_names_the_binding_contract(self):
        assert set(task_accel.__all__) == {"_RUST_TASK", *_ACCEL_NAMES}

    def test_task_scoring_consumes_the_binding_module(self):
        assert task_scoring_mod._task_accel is task_accel
        for name in ("_RUST_TASK", *_ACCEL_NAMES):
            assert not hasattr(task_scoring_mod, name)

    def test_lane_is_independent_of_the_nli_binding(self):
        import director_ai.core.scoring._nli_accel as nli_accel

        assert task_accel is not nli_accel
        assert task_accel.rust_split_sentences.__module__ != "director_ai"


class TestCanonicalPatchPoint:
    def test_disabling_the_flag_routes_detection_to_the_python_floor(self, monkeypatch):
        def _explode(_prompt, _response):
            raise AssertionError("accelerated branch must not run")

        monkeypatch.setattr(task_accel, "_RUST_TASK", False)
        monkeypatch.setattr(task_accel, "rust_detect_task_type", _explode)
        assert detect_task_type("User: hi\nAssistant: hello\nUser: bye") == "dialogue"

    def test_enabling_the_flag_dispatches_through_the_binding(self, monkeypatch):
        monkeypatch.setattr(task_accel, "_RUST_TASK", True)
        monkeypatch.setattr(task_accel, "rust_sum_i64", lambda values: 41 + len(values))
        assert task_scoring_mod._sum_int([0]) == 42
