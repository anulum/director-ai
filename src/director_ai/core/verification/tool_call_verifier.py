# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tool call verification — detect fabricated API results in agentic workflows.

Verifies that an agent's tool/function call is legitimate:
1. Function exists in the provided tool manifest
2. Arguments match expected types from the manifest
3. Claimed result is plausible given the function description
4. Cross-references against an execution log to detect fabrication

Usage::

    manifest = {
        "get_weather": {
            "description": "Get current weather for a city",
            "parameters": {"city": {"type": "string"}},
            "returns": "Weather data with temperature and conditions",
        }
    }

    result = verify_tool_call(
        function_name="get_weather",
        arguments={"city": "Prague"},
        claimed_result='{"temp": 22, "conditions": "sunny"}',
        manifest=manifest,
    )
"""

from __future__ import annotations

from .types import FieldVerdict, ToolCallResult

__all__ = ["verify_tool_call"]

_JSON_SCHEMA_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def verify_tool_call(
    function_name: str,
    arguments: dict,
    claimed_result: str = "",
    manifest: dict | None = None,
    execution_log: list[dict] | None = None,
    score_fn=None,
) -> ToolCallResult:
    """Verify a tool/function call against a manifest and optional execution log.

    Parameters
    ----------
    function_name : str
        The function the agent claims to have called.
    arguments : dict
        Arguments passed to the function.
    claimed_result : str
        The result the agent claims the function returned.
    manifest : dict | None
        Tool inventory: ``{func_name: {"description": ..., "parameters": ..., "returns": ...}}``.
    execution_log : list[dict] | None
        Actual execution records: ``[{"function": ..., "arguments": ..., "result": ...}]``.
    score_fn : callable | None
        ``score_fn(premise, hypothesis) -> float`` for result plausibility scoring.

    Returns
    -------
    ToolCallResult
    """
    verdicts: list[FieldVerdict] = []
    function_exists = True
    arguments_valid = True
    result_plausible = True
    fabrication_suspected = False
    reason_parts: list[str] = []

    # 1. Function existence
    if manifest is not None:
        if function_name not in manifest:
            function_exists = False
            reason_parts.append(f"Function '{function_name}' not in manifest")
            verdicts.append(
                FieldVerdict(
                    path="function_name",
                    value=function_name,
                    verdict="invalid_value",
                    reason=f"Not found in manifest. Available: {list(manifest.keys())[:10]}",
                )
            )
        else:
            func_spec = manifest[function_name]
            param_spec = func_spec.get("parameters", {})

            # 2. Argument validation
            for param_name, param_def in param_spec.items():
                if param_name not in arguments:
                    if param_def.get("required", True):
                        arguments_valid = False
                        verdicts.append(
                            FieldVerdict(
                                path=f"arguments.{param_name}",
                                value="",
                                verdict="missing",
                                reason=f"Required parameter '{param_name}' not provided",
                            )
                        )
                else:
                    val = arguments[param_name]
                    expected_type = param_def.get("type")
                    if expected_type:
                        py_type = _JSON_SCHEMA_TYPE_MAP.get(expected_type)
                        is_wrong = py_type and not isinstance(val, py_type)  # type: ignore[arg-type]
                        if expected_type in ("integer", "number") and isinstance(
                            val, bool
                        ):
                            is_wrong = True
                        if is_wrong:
                            arguments_valid = False
                            verdicts.append(
                                FieldVerdict(
                                    path=f"arguments.{param_name}",
                                    value=str(val),
                                    verdict="invalid_type",
                                    reason=f"Expected {expected_type}, got {type(val).__name__}",
                                )
                            )
                        else:
                            verdicts.append(
                                FieldVerdict(
                                    path=f"arguments.{param_name}",
                                    value=str(val),
                                    verdict="valid",
                                )
                            )

            # Check for unexpected arguments
            known_params = set(param_spec.keys())
            for arg_name in arguments:
                if arg_name not in known_params:
                    verdicts.append(
                        FieldVerdict(
                            path=f"arguments.{arg_name}",
                            value=str(arguments[arg_name]),
                            verdict="extra",
                            reason=f"Unexpected argument '{arg_name}'",
                        )
                    )

    # 3. Result plausibility via NLI
    if (
        score_fn is not None
        and claimed_result
        and manifest
        and function_name in manifest
    ):
        func_desc = manifest[function_name].get("returns", "")
        if func_desc:
            premise = f"The function {function_name} returns: {func_desc}"
            div = score_fn(premise, claimed_result)
            result_plausible = div < 0.6
            verdicts.append(
                FieldVerdict(
                    path="result",
                    value=claimed_result[:200],
                    verdict="valid" if result_plausible else "invalid_value",
                    reason=f"Result plausibility divergence: {div:.2f}",
                )
            )

    # 4. Fabrication detection via execution log
    if execution_log is not None:
        matching = [
            entry for entry in execution_log if entry.get("function") == function_name
        ]
        if not matching:
            fabrication_suspected = True
            reason_parts.append(
                f"No execution log entry for '{function_name}' — possible fabrication"
            )
        else:
            best_match = None
            for entry in matching:
                if entry.get("arguments") == arguments:
                    best_match = entry
                    break
            if best_match is None:
                reason_parts.append(
                    f"Function '{function_name}' was called but with different arguments"
                )
            elif claimed_result and best_match.get("result") != claimed_result:
                fabrication_suspected = True
                reason_parts.append(
                    "Claimed result does not match execution log result"
                )

    return ToolCallResult(
        function_exists=function_exists,
        arguments_valid=arguments_valid,
        result_plausible=result_plausible,
        fabrication_suspected=fabrication_suspected,
        verdicts=verdicts,
        reason="; ".join(reason_parts),
    )
