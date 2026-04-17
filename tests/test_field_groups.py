# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.ui._field_groups``.

Covers field classification, widget inference, group completeness,
introspection of DirectorConfig, and edge cases.
"""

from __future__ import annotations

from director_ai.ui._field_groups import (
    FIELD_GROUPS,
    FieldMeta,
    _classify_field,
    _infer_widget,
    get_field_groups,
)

# ── Field classification ───────────────────────────────────────────────


class TestClassifyField:
    def test_scoring_fields(self):
        assert _classify_field("coherence_threshold") == "Scoring"
        assert _classify_field("scorer_backend") == "Scoring"

    def test_nli_fields(self):
        assert _classify_field("nli_model") == "NLI Model"
        assert _classify_field("nli_max_length") == "NLI Model"

    def test_retrieval_fields(self):
        assert _classify_field("vector_backend") == "Retrieval"
        assert _classify_field("hybrid_retrieval") == "Retrieval"
        assert _classify_field("parent_child_enabled") == "Retrieval"
        assert _classify_field("hyde_enabled") == "Retrieval"
        assert _classify_field("query_decomposition_enabled") == "Retrieval"
        assert _classify_field("multi_vector_enabled") == "Retrieval"

    def test_injection_fields(self):
        assert _classify_field("injection_detection_enabled") == "Injection Detection"
        assert _classify_field("injection_threshold") == "Injection Detection"

    def test_enterprise_fields(self):
        assert _classify_field("redis_url") == "Enterprise"
        assert _classify_field("cache_size") == "Enterprise"

    def test_security_fields(self):
        assert _classify_field("sanitize_inputs") == "Security"
        assert _classify_field("redact_pii") == "Security"

    def test_unknown_goes_to_other(self):
        assert _classify_field("completely_unknown_xyz") == "Other"


# ── Widget inference ───────────────────────────────────────────────────


class TestWidgetInference:
    def test_bool_toggle(self):
        assert _infer_widget(bool) == "toggle"

    def test_int_number(self):
        assert _infer_widget(int) == "number"

    def test_float_slider(self):
        assert _infer_widget(float) == "slider"

    def test_str_text(self):
        assert _infer_widget(str) == "text"

    def test_unknown_type_text(self):
        assert _infer_widget(list) == "text"


# ── FieldMeta ──────────────────────────────────────────────────────────


class TestFieldMeta:
    def test_to_dict(self):
        m = FieldMeta("test", "Scoring", float, 0.5, "slider", "desc")
        d = m.to_dict()
        assert d["name"] == "test"
        assert d["group"] == "Scoring"
        assert d["type"] == "float"
        assert d["default"] == 0.5
        assert d["widget"] == "slider"

    def test_slots(self):
        m = FieldMeta("x", "g", str, "", "text")
        assert hasattr(m, "__slots__")


# ── get_field_groups introspection ─────────────────────────────────────


class TestGetFieldGroups:
    def test_returns_dict(self):
        groups = get_field_groups()
        assert isinstance(groups, dict)

    def test_scoring_group_exists(self):
        groups = get_field_groups()
        assert "Scoring" in groups
        assert len(groups["Scoring"]) > 0

    def test_retrieval_group_has_new_fields(self):
        groups = get_field_groups()
        names = [f["name"] for f in groups.get("Retrieval", [])]
        assert "parent_child_enabled" in names
        assert "hyde_enabled" in names
        assert "multi_vector_enabled" in names

    def test_all_fields_have_required_keys(self):
        groups = get_field_groups()
        for _group_name, field_list in groups.items():
            for f in field_list:
                assert "name" in f
                assert "group" in f
                assert "type" in f
                assert "widget" in f

    def test_total_field_count(self):
        groups = get_field_groups()
        total = sum(len(fields) for fields in groups.values())
        # DirectorConfig has 100+ fields
        assert total > 50

    def test_no_empty_groups_in_predefined(self):
        groups = get_field_groups()
        # At least Scoring and Retrieval should be non-empty
        assert len(groups.get("Scoring", [])) > 0
        assert len(groups.get("Retrieval", [])) > 0


# ── FIELD_GROUPS constant ─────────────────────────────────────────────


class TestFieldGroupsConstant:
    def test_has_expected_groups(self):
        assert "Scoring" in FIELD_GROUPS
        assert "Retrieval" in FIELD_GROUPS
        assert "Enterprise" in FIELD_GROUPS
        assert "Other" in FIELD_GROUPS

    def test_other_is_last(self):
        assert FIELD_GROUPS[-1] == "Other"
