# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — guard-decision latency vs NeMo Guardrails / Guardrails AI

"""Per-call guard-decision latency: Director-AI vs NeMo Guardrails / Guardrails AI.

This measures *framework overhead to make a local guard decision* — not LLM
latency — so the comparison is honest about what each tool does locally:

* **Director-AI** (``CoherenceScorer.review``, heuristic ``use_nli=False``):
  makes a real grounding decision locally, no model download, no LLM call.
* **Guardrails AI** (``Guard().parse``): the framework's local parse/validate
  overhead. Grounding/hallucination validators live on the Hub and themselves
  call an LLM, so the local-only number is the framework baseline, not a
  grounding decision.
* **NeMo Guardrails**: its rails require a configured LLM to ``generate``; there
  is **no local grounding decision** to time, so per-call latency in production
  is dominated by an LLM round-trip. We record config-load overhead and flag the
  architectural difference rather than fabricate a comparable local number.

The competitor libraries are heavy and conflict with the core lock, so they live
in a separate virtualenv; this orchestrator times Director-AI in-process and
shells the competitor timings into that venv (``--competitor-python`` or
``DIRECTOR_COMPETITOR_PYTHON``; default ``…/_scratch/venv-bench-competitors``).
Each competitor stage degrades to ``available: false`` with a reason when its
library or venv is absent, so the bench never fabricates a number.

Run::

    python -m benchmarks.competitor_latency_bench --repeats 200
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics

# Benchmark-only subprocess runner; snippets are fixed in this module.
import subprocess  # nosec B404
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"

_DEFAULT_COMPETITOR_PYTHON = (
    "/media/anulum/GOTM/_scratch/venv-bench-competitors/bin/python"
)

# Representative grounded prompt/response pairs (no network, no fabricated data).
WORKLOAD: list[tuple[str, str]] = [
    (
        "What is the boiling point of water?",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    ),
    (
        "How fast is light?",
        "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
    ),
    (
        "Describe DNA base pairing.",
        "Adenine pairs with thymine and guanine pairs with cytosine in DNA.",
    ),
    (
        "What is Earth's gravity?",
        "Earth's surface gravitational acceleration is about 9.81 metres per second squared.",
    ),
    (
        "Summarise photosynthesis.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight in chloroplasts.",
    ),
]


def summarise(samples_ms: list[float]) -> dict:
    """p50 / p95 / mean / min / max for a list of per-call millisecond timings."""
    if not samples_ms:
        return {"n": 0}
    ordered = sorted(samples_ms)
    return {
        "n": len(ordered),
        "p50_ms": round(statistics.median(ordered), 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "mean_ms": round(statistics.fmean(ordered), 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
    }


def time_director_ai(repeats: int) -> dict:
    """Time the local heuristic guard decision (no NLI model, no LLM, no network)."""
    from director_ai.core import CoherenceScorer, GroundTruthStore

    store = GroundTruthStore()
    for prompt, response in WORKLOAD:
        store.add(prompt, response)
    scorer = CoherenceScorer(threshold=0.3, ground_truth_store=store, use_nli=False)

    # Warmup (exclude first-call import/JIT effects from the measured samples).
    for prompt, response in WORKLOAD:
        scorer.review(prompt, response)

    samples: list[float] = []
    for _ in range(repeats):
        for prompt, response in WORKLOAD:
            t0 = time.perf_counter()
            scorer.review(prompt, response)
            samples.append((time.perf_counter() - t0) * 1000)
    return {
        "framework": "director-ai",
        "operation": "CoherenceScorer.review (heuristic, local grounding decision, no LLM)",
        "available": True,
        "makes_local_grounding_decision": True,
        **summarise(samples),
    }


# Self-contained snippets run inside the competitor venv (must NOT import
# director_ai or this package — that venv has neither).
_GUARDRAILS_SNIPPET = """
import json, time, statistics
WORKLOAD = {workload}
REPEATS = {repeats}
try:
    from guardrails import Guard
except Exception as exc:  # pragma: no cover - exercised only without the lib
    print(json.dumps({{"available": False, "reason": f"import failed: {{exc}}"}}))
    raise SystemExit(0)
g = Guard()
for _p, r in WORKLOAD:
    g.parse(llm_output=r)
samples = []
for _ in range(REPEATS):
    for _p, r in WORKLOAD:
        t0 = time.perf_counter()
        g.parse(llm_output=r)
        samples.append((time.perf_counter() - t0) * 1000)
ordered = sorted(samples)
print(json.dumps({{
    "available": True,
    "n": len(ordered),
    "p50_ms": round(statistics.median(ordered), 4),
    "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
    "mean_ms": round(statistics.fmean(ordered), 4),
    "min_ms": round(ordered[0], 4),
    "max_ms": round(ordered[-1], 4),
}}))
"""

_NEMO_SNIPPET = """
import json, time, statistics
REPEATS = {repeats}
try:
    from nemoguardrails import RailsConfig
except Exception as exc:  # pragma: no cover - exercised only without the lib
    print(json.dumps({{"available": False, "reason": f"import failed: {{exc}}"}}))
    raise SystemExit(0)
# NeMo rails need a configured LLM to generate(); with no model there is no local
# guard decision to time. We record config-load overhead and flag the difference.
samples = []
for _ in range(REPEATS):
    t0 = time.perf_counter()
    RailsConfig.from_content(yaml_content="models: []\\n")
    samples.append((time.perf_counter() - t0) * 1000)
ordered = sorted(samples)
print(json.dumps({{
    "available": True,
    "n": len(ordered),
    "p50_ms": round(statistics.median(ordered), 4),
    "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
    "mean_ms": round(statistics.fmean(ordered), 4),
    "min_ms": round(ordered[0], 4),
    "max_ms": round(ordered[-1], 4),
}}))
"""


def _run_competitor(python_exe: str, snippet: str) -> dict:
    """Run a self-contained timing snippet in the competitor venv; parse its JSON."""
    if not Path(python_exe).exists():
        return {
            "available": False,
            "reason": f"competitor python not found: {python_exe}",
        }
    try:
        env = os.environ.copy()
        env.update(
            {
                "GUARDRAILS_DISABLE_TELEMETRY": "true",
                "OTEL_SDK_DISABLED": "true",
            }
        )
        # Fixed snippet body with shell=False; python_exe is a benchmark input.
        proc = subprocess.run(  # nosec B603
            [python_exe, "-c", snippet],
            capture_output=True,
            env=env,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"subprocess error: {exc}"}
    if proc.returncode != 0:
        return {
            "available": False,
            "reason": f"exit {proc.returncode}: {proc.stderr[-300:]}",
        }
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {
            "available": False,
            "reason": f"unparsable output: {proc.stdout[-300:]}",
        }


def time_guardrails_ai(repeats: int, python_exe: str) -> dict:
    payload = _run_competitor(
        python_exe,
        _GUARDRAILS_SNIPPET.format(workload=repr(WORKLOAD), repeats=repeats),
    )
    return {
        "framework": "guardrails-ai",
        "operation": "Guard().parse (framework parse overhead; grounding validators are Hub/LLM-backed)",
        "makes_local_grounding_decision": False,
        **payload,
    }


def time_nemo(repeats: int, python_exe: str) -> dict:
    payload = _run_competitor(python_exe, _NEMO_SNIPPET.format(repeats=repeats))
    return {
        "framework": "nemo-guardrails",
        "operation": "RailsConfig.from_content (config-load overhead; rails require an LLM to decide)",
        "makes_local_grounding_decision": False,
        "note": "NeMo rails are LLM-bound: production per-call latency is dominated by an LLM round-trip.",
        **payload,
    }


def _load_average() -> dict[str, float] | None:
    """Return the Unix load average when the host exposes it.

    Benchmark artifacts need to be auditable as local regression evidence, not
    claim-grade isolated lab results. Recording host load with the result makes
    that boundary explicit and gives future runs enough context to spot obvious
    workstation contention.
    """

    try:
        one, five, fifteen = os.getloadavg()
    except (AttributeError, OSError):
        return None
    return {"1m": round(one, 4), "5m": round(five, 4), "15m": round(fifteen, 4)}


def _read_first_line(path: Path) -> str | None:
    """Read a short host metadata file when present."""

    try:
        return path.read_text(encoding="utf-8").strip().splitlines()[0]
    except (OSError, IndexError):
        return None


def _cpu_model() -> str | None:
    """Return the Linux CPU model name when available."""

    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        return None
    return None


def _runtime_metadata(repeats: int, competitor_python: str) -> dict[str, Any]:
    """Return reproducibility metadata for the current local benchmark run."""

    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "hardware_vendor": _read_first_line(Path("/sys/class/dmi/id/sys_vendor")),
        "hardware_model": _read_first_line(Path("/sys/class/dmi/id/product_name")),
        "baseboard_vendor": _read_first_line(Path("/sys/class/dmi/id/board_vendor")),
        "baseboard_model": _read_first_line(Path("/sys/class/dmi/id/board_name")),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "load_average": _load_average(),
        "benchmark_isolation": "non_isolated_local_regression",
        "competitor_python": competitor_python,
        "command": [
            sys.executable,
            "-m",
            "benchmarks.competitor_latency_bench",
            "--repeats",
            str(repeats),
            "--competitor-python",
            competitor_python,
        ],
    }


def run_benchmark(repeats: int, competitor_python: str) -> dict[str, Any]:
    return {
        "benchmark": "competitor_guard_latency",
        "schema_version": 2,
        "workload_pairs": len(WORKLOAD),
        "repeats": repeats,
        "metadata": _runtime_metadata(repeats, competitor_python),
        "workload": [
            {"prompt": prompt, "response": response} for prompt, response in WORKLOAD
        ],
        "frameworks": [
            time_director_ai(repeats),
            time_guardrails_ai(repeats, competitor_python),
            time_nemo(repeats, competitor_python),
        ],
        "caveat": (
            "Measures local framework overhead, not LLM latency. Only Director-AI "
            "makes a grounding decision locally; Guardrails AI grounding validators "
            "and all NeMo rails require an LLM call, adding its full round-trip on "
            "top of the numbers shown."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--competitor-python",
        default=os.environ.get(
            "DIRECTOR_COMPETITOR_PYTHON", _DEFAULT_COMPETITOR_PYTHON
        ),
    )
    args = parser.parse_args()

    result = run_benchmark(args.repeats, args.competitor_python)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "competitor_guard_latency.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nGuard-decision latency ({args.repeats} repeats x {len(WORKLOAD)} pairs):")
    for fw in result["frameworks"]:
        if fw.get("available"):
            print(
                f"  {fw['framework']:<16} p50={fw['p50_ms']:>8.3f}ms  "
                f"p95={fw['p95_ms']:>8.3f}ms  mean={fw['mean_ms']:>8.3f}ms  "
                f"local_decision={fw['makes_local_grounding_decision']}"
            )
        else:
            print(f"  {fw['framework']:<16} unavailable: {fw.get('reason', '?')}")
    print(f"\n  {result['caveat']}")
    print(f"  saved -> {out}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
