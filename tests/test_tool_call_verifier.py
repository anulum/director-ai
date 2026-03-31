# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for tool call verification pipeline.

Covers: tool call schema validation, argument verification, hallucinated
tool detection, missing parameter detection, pipeline integration with
CoherenceScorer, and performance documentation.
"""

from __future__ import annotations

from director_ai.core.verification.tool_call_verifier import verify_tool_call

MANIFEST = {
    "get_weather": {
        "description": "Get current weather for a city",
        "parameters": {"city": {"type": "string"}},
        "returns": "Weather data with temperature and conditions",
    },
    "search_database": {
        "description": "Search customer database by query",
        "parameters": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "required": False},
        },
        "returns": "List of matching customer records",
    },
}


class TestFunctionExistence:
    def test_existing_function(self):
        r = verify_tool_call("get_weather", {"city": "Prague"}, manifest=MANIFEST)
        assert r.function_exists is True

    def test_nonexistent_function(self):
        r = verify_tool_call("get_stock_price", {"symbol": "AAPL"}, manifest=MANIFEST)
        assert r.function_exists is False
        assert "not in manifest" in r.reason

    def test_no_manifest(self):
        r = verify_tool_call("anything", {"x": 1})
        assert r.function_exists is True


class TestArgumentValidation:
    def test_correct_arguments(self):
        r = verify_tool_call("get_weather", {"city": "Prague"}, manifest=MANIFEST)
        assert r.arguments_valid is True

    def test_wrong_argument_type(self):
        r = verify_tool_call("get_weather", {"city": 123}, manifest=MANIFEST)
        assert r.arguments_valid is False
        assert any(v.verdict == "invalid_type" for v in r.verdicts)

    def test_missing_required_argument(self):
        r = verify_tool_call("search_database", {}, manifest=MANIFEST)
        assert r.arguments_valid is False
        assert any(v.verdict == "missing" for v in r.verdicts)

    def test_optional_argument_missing(self):
        r = verify_tool_call("search_database", {"query": "test"}, manifest=MANIFEST)
        assert r.arguments_valid is True

    def test_extra_argument(self):
        r = verify_tool_call(
            "get_weather", {"city": "Prague", "units": "metric"}, manifest=MANIFEST
        )
        assert any(v.verdict == "extra" for v in r.verdicts)

    def test_boolean_not_number(self):
        manifest = {
            "charge": {
                "parameters": {"amount": {"type": "number", "required": True}},
            }
        }
        r = verify_tool_call("charge", {"amount": True}, manifest=manifest)
        assert r.arguments_valid is False
        assert any(v.verdict == "invalid_type" for v in r.verdicts)

    def test_boolean_not_integer(self):
        manifest = {
            "count": {
                "parameters": {"n": {"type": "integer", "required": True}},
            }
        }
        r = verify_tool_call("count", {"n": False}, manifest=manifest)
        assert r.arguments_valid is False
        assert any(v.verdict == "invalid_type" for v in r.verdicts)


class TestFabricationDetection:
    def test_matching_log_entry(self):
        log = [
            {
                "function": "get_weather",
                "arguments": {"city": "Prague"},
                "result": "sunny 22C",
            }
        ]
        r = verify_tool_call(
            "get_weather",
            {"city": "Prague"},
            claimed_result="sunny 22C",
            manifest=MANIFEST,
            execution_log=log,
        )
        assert r.fabrication_suspected is False

    def test_no_log_entry(self):
        r = verify_tool_call(
            "get_weather",
            {"city": "Prague"},
            claimed_result="sunny 22C",
            manifest=MANIFEST,
            execution_log=[],
        )
        assert r.fabrication_suspected is True
        assert "fabrication" in r.reason.lower()

    def test_mismatched_result(self):
        log = [
            {
                "function": "get_weather",
                "arguments": {"city": "Prague"},
                "result": "rainy 10C",
            }
        ]
        r = verify_tool_call(
            "get_weather",
            {"city": "Prague"},
            claimed_result="sunny 22C",
            manifest=MANIFEST,
            execution_log=log,
        )
        assert r.fabrication_suspected is True

    def test_different_arguments_in_log(self):
        log = [
            {
                "function": "get_weather",
                "arguments": {"city": "Berlin"},
                "result": "cloudy",
            }
        ]
        r = verify_tool_call(
            "get_weather",
            {"city": "Prague"},
            claimed_result="sunny",
            manifest=MANIFEST,
            execution_log=log,
        )
        assert "different arguments" in r.reason


class TestResultPlausibility:
    def test_plausible_result(self):
        r = verify_tool_call(
            "get_weather",
            {"city": "Prague"},
            claimed_result="Temperature 22C, sunny",
            manifest=MANIFEST,
            score_fn=lambda p, h: 0.2,
        )
        assert r.result_plausible is True

    def test_implausible_result(self):
        r = verify_tool_call(
            "get_weather",
            {"city": "Prague"},
            claimed_result="The meaning of life is 42",
            manifest=MANIFEST,
            score_fn=lambda p, h: 0.9,
        )
        assert r.result_plausible is False
