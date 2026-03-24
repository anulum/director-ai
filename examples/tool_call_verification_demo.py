# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tool call verification demo — detect fabricated API results."""

from director_ai import verify_tool_call

# Define your tool manifest
manifest = {
    "get_weather": {
        "description": "Get current weather for a city",
        "parameters": {"city": {"type": "string"}},
        "returns": "Weather data with temperature and conditions",
    },
    "search_orders": {
        "description": "Search customer orders by query",
        "parameters": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "required": False},
        },
        "returns": "List of matching order records",
    },
}

# Valid tool call
result = verify_tool_call("get_weather", {"city": "Prague"}, manifest=manifest)
print(
    f"Function exists: {result.function_exists}, Args valid: {result.arguments_valid}"
)

# Nonexistent function (agent hallucinated an API)
result = verify_tool_call("get_stock_price", {"symbol": "AAPL"}, manifest=manifest)
print(f"\nHallucinated function: exists={result.function_exists}")
print(f"  Reason: {result.reason}")

# Wrong argument type
result = verify_tool_call("get_weather", {"city": 42}, manifest=manifest)
print(f"\nWrong type: args_valid={result.arguments_valid}")
for v in result.verdicts:
    if v.verdict != "valid":
        print(f"  {v.path}: {v.verdict} — {v.reason}")

# Fabrication detection with execution log
execution_log = [
    {
        "function": "get_weather",
        "arguments": {"city": "Berlin"},
        "result": "cloudy 10C",
    },
]
result = verify_tool_call(
    "get_weather",
    {"city": "Prague"},
    claimed_result="sunny 22C",
    manifest=manifest,
    execution_log=execution_log,
)
print(f"\nFabrication check: suspected={result.fabrication_suspected}")
print(f"  Reason: {result.reason}")
