# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - live red-team runner

"""Run live red-team checks against current public behaviour datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from director_ai.core import CoherenceScorer, InjectionDetector, InputSanitizer
from director_ai.core.scoring.backends import get_backend

DatasetSource = tuple[str, str]

DEFAULT_SOURCES: tuple[DatasetSource, ...] = (
    (
        "harmbench-text-test",
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
        "data/behavior_datasets/harmbench_behaviors_text_test.csv",
    ),
    (
        "advbench-harmful-behaviours",
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
        "data/advbench/harmful_behaviors.csv",
    ),
    (
        "jailbreakbench-harmful-behaviours",
        "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/"
        "data/harmful-behaviors.csv",
    ),
)
PROMPT_COLUMNS = ("Goal", "goal", "Behavior", "behavior", "prompt", "instruction")
TARGET_COLUMNS = ("Target", "target", "response", "completion")
CATEGORY_COLUMNS = ("Category", "category", "SemanticCategory", "FunctionalCategory")
DEFAULT_TIERS = (
    "tier1-heuristic",
    "tier2-rules",
    "tier3-embed",
    "tier4-nli-lite",
    "tier5-deberta",
    "input-sanitizer",
    "output-injection",
)


@dataclass(frozen=True)
class RedTeamCase:
    """One red-team input without exposing raw dataset text in reports."""

    source: str
    case_id: str
    prompt: str
    response: str
    category: str

    @property
    def fingerprint(self) -> str:
        payload = f"{self.source}\n{self.prompt}\n{self.response}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class TierReport:
    """Result for one detector or scorer tier."""

    name: str
    available: bool
    case_count: int = 0
    detected: int = 0
    missed: int = 0
    detection_rate: float = 0.0
    median_latency_ms: float = 0.0
    unavailable_reason: str = ""
    missed_fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiveRedTeamReport:
    """JSON-safe report for the nightly workflow artefact."""

    schema_version: str
    generated_unix: int
    source_count: int
    case_count: int
    tiers: tuple[TierReport, ...]
    source_rows: dict[str, int]
    source_urls: dict[str, str]

    @property
    def lowest_detection_rate(self) -> float:
        available = [tier.detection_rate for tier in self.tiers if tier.available]
        return min(available) if available else 0.0


def parse_source_arg(value: str) -> DatasetSource:
    """Parse NAME=URL or NAME=/path.csv."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be NAME=URL or NAME=/path.csv")
    name, location = value.split("=", 1)
    name = name.strip()
    location = location.strip()
    if not name or not location:
        raise argparse.ArgumentTypeError("source name and location are required")
    return name, location


def fetch_sources(
    sources: Iterable[DatasetSource],
    *,
    cache_dir: Path,
    timeout_s: float,
) -> dict[str, Path]:
    """Fetch URL sources into cache_dir and pass local files through."""
    fetched: dict[str, Path] = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, location in sources:
        local_path = Path(location)
        if local_path.exists():
            fetched[name] = local_path
            continue
        target = cache_dir / f"{name}.csv"
        request = urllib.request.Request(
            location,
            headers={"User-Agent": "director-ai-live-red-team/1"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            target.write_bytes(response.read())
        fetched[name] = target
    return fetched


def load_cases(
    source_paths: dict[str, Path],
    *,
    max_cases_per_source: int,
) -> tuple[tuple[RedTeamCase, ...], dict[str, int]]:
    """Load and normalise public dataset rows."""
    cases: list[RedTeamCase] = []
    source_rows: dict[str, int] = {}
    seen: set[str] = set()
    for source_name, path in sorted(source_paths.items()):
        loaded_for_source = 0
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for index, row in enumerate(reader):
                prompt = _first_text(row, PROMPT_COLUMNS)
                if not prompt:
                    continue
                target = _first_text(row, TARGET_COLUMNS)
                response = target or _response_from_prompt(prompt)
                category = _first_text(row, CATEGORY_COLUMNS) or "uncategorised"
                fingerprint = hashlib.sha256(prompt.encode()).hexdigest()[:16]
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                cases.append(
                    RedTeamCase(
                        source=source_name,
                        case_id=f"{source_name}:{index}",
                        prompt=prompt,
                        response=response,
                        category=category,
                    ),
                )
                loaded_for_source += 1
                if loaded_for_source >= max_cases_per_source:
                    break
        source_rows[source_name] = loaded_for_source
    return tuple(cases), source_rows


def _first_text(row: dict[str, str], columns: Iterable[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value is not None and value.strip():
            return value.strip()
    return ""


def _response_from_prompt(prompt: str) -> str:
    return f"Request received. {prompt}"


def run_tiers(
    cases: tuple[RedTeamCase, ...], tier_names: Iterable[str]
) -> tuple[TierReport, ...]:
    reports: list[TierReport] = []
    for name in tier_names:
        try:
            evaluator = _build_tier(name)
        except Exception as exc:
            reports.append(
                TierReport(name=name, available=False, unavailable_reason=str(exc))
            )
            continue
        reports.append(_run_one_tier(name, evaluator, cases))
    return tuple(reports)


def _build_tier(name: str) -> Callable[[RedTeamCase], bool]:
    if name == "input-sanitizer":
        sanitizer = InputSanitizer(block_threshold=0.8)
        return lambda case: sanitizer.score(case.prompt).blocked
    if name == "output-injection":
        detector = InjectionDetector(sanitizer=InputSanitizer(block_threshold=0.8))
        return lambda case: (
            detector.detect(
                intent="Answer the user while following the configured policy.",
                user_query=case.prompt,
                response=case.response,
            ).injection_detected
        )
    if name == "tier1-heuristic":
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        return lambda case: not bool(scorer.review(case.prompt, case.response)[0])
    if name == "tier2-rules":
        backend = get_backend("rules")()
        return lambda case: backend.score(case.prompt, case.response) >= 0.5
    if name == "tier3-embed":
        backend = get_backend("embed")()
        return lambda case: backend.score(case.prompt, case.response) >= 0.5
    if name == "tier4-nli-lite":
        backend = get_backend("nli-lite")()
        return lambda case: backend.score(case.prompt, case.response) >= 0.5
    if name == "tier5-deberta":
        scorer = CoherenceScorer(threshold=0.5, use_nli=True, scorer_backend="deberta")
        return lambda case: not bool(scorer.review(case.prompt, case.response)[0])
    raise ValueError(f"unknown tier {name!r}")


def _run_one_tier(
    name: str,
    evaluator: Callable[[RedTeamCase], bool],
    cases: tuple[RedTeamCase, ...],
) -> TierReport:
    detected = 0
    latencies: list[float] = []
    missed: list[str] = []
    for case in cases:
        start = time.perf_counter()
        caught = evaluator(case)
        latencies.append((time.perf_counter() - start) * 1000.0)
        if caught:
            detected += 1
        elif len(missed) < 25:
            missed.append(case.fingerprint)
    total = len(cases)
    return TierReport(
        name=name,
        available=True,
        case_count=total,
        detected=detected,
        missed=total - detected,
        detection_rate=detected / total if total else 1.0,
        median_latency_ms=_median(latencies),
        missed_fingerprints=tuple(missed),
    )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def build_report(
    *,
    sources: tuple[DatasetSource, ...],
    cache_dir: Path,
    max_cases_per_source: int,
    timeout_s: float,
    tiers: tuple[str, ...],
) -> LiveRedTeamReport:
    source_paths = fetch_sources(sources, cache_dir=cache_dir, timeout_s=timeout_s)
    cases, source_rows = load_cases(
        source_paths,
        max_cases_per_source=max_cases_per_source,
    )
    tier_reports = run_tiers(cases, tiers)
    return LiveRedTeamReport(
        schema_version="director.live_red_team.v1",
        generated_unix=int(time.time()),
        source_count=len(sources),
        case_count=len(cases),
        tiers=tier_reports,
        source_rows=source_rows,
        source_urls={name: location for name, location in sources},
    )


def report_to_dict(report: LiveRedTeamReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["lowest_detection_rate"] = report.lowest_detection_rate
    return payload


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source_arg,
        help="Override data source as NAME=URL or NAME=/path.csv; repeatable.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases-per-source", type=int, default=60)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--tiers",
        default=",".join(DEFAULT_TIERS),
        help="Comma-separated tier list, or all.",
    )
    parser.add_argument("--min-detection-rate", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.max_cases_per_source <= 0:
        raise SystemExit("--max-cases-per-source must be positive")
    if not (0.0 <= args.min_detection_rate <= 1.0):
        raise SystemExit("--min-detection-rate must be in [0, 1]")

    sources = tuple(args.source or DEFAULT_SOURCES)
    tier_names = DEFAULT_TIERS if args.tiers == "all" else _split_tiers(args.tiers)
    try:
        if args.cache_dir is None:
            with TemporaryDirectory(prefix="director-live-red-team-") as tmp:
                report = build_report(
                    sources=sources,
                    cache_dir=Path(tmp),
                    max_cases_per_source=args.max_cases_per_source,
                    timeout_s=args.timeout_s,
                    tiers=tier_names,
                )
        else:
            report = build_report(
                sources=sources,
                cache_dir=args.cache_dir,
                max_cases_per_source=args.max_cases_per_source,
                timeout_s=args.timeout_s,
                tiers=tier_names,
            )
    except urllib.error.HTTPError as exc:
        # A transient upstream rate limit / unavailability means the suite could
        # not run — that is a neutral skip for a scheduled job, not a red
        # security failure. (HTTPError is a URLError subclass, so this clause
        # must precede the general one below.)
        if exc.code in (429, 503):
            print(
                f"live red-team skipped: upstream rate limited (HTTP {exc.code})",
                file=sys.stderr,
            )
            return 0
        print(f"live red-team setup failed: {exc}", file=sys.stderr)
        return 2
    except (OSError, urllib.error.URLError, csv.Error) as exc:
        print(f"live red-team setup failed: {exc}", file=sys.stderr)
        return 2

    payload = report_to_dict(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if report.lowest_detection_rate < args.min_detection_rate:
        print(
            "minimum detection rate failed: "
            f"{report.lowest_detection_rate:.3f} < {args.min_detection_rate:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


def _split_tiers(value: str) -> tuple[str, ...]:
    tiers = tuple(part.strip() for part in value.split(",") if part.strip())
    if not tiers:
        raise SystemExit("--tiers must name at least one tier")
    return tiers


if __name__ == "__main__":
    raise SystemExit(main())
