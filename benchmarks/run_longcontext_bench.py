# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — long-context hallucination-detection sweep (WCS-1)
"""Evidence-composition × aggregation sweep for long-context detection.

WCS-1 (``docs/internal/BACKLOG.md``): at threshold 0.5 the tracked _200
baseline catches 84.5 % on HaluEval QA but only 12 % on summarisation and
4.5 % on dialogue (``benchmarks/results/judge_bench_nli_only_200.json``).
The 2026-07-14 diagnosis found three mechanical causes — D1 dialogue
premises omit the ``knowledge`` field, D2 summarisation premises truncate
to a 3 000-char prefix while 60 % of documents are longer, D3 coverage-mean
aggregation dilutes single-fact-swap responses. This harness measures the
fix families without a new model.

Design (the WCA-2 pattern): the expensive claim×evidence score matrix is
computed ONCE per checker (``--matrix-out``), and every sweep dimension is
an offline re-aggregation of that matrix (``--sweep-from``):

- **evidence** — summarisation: ``prefix3000`` (production today, D2
  baseline) / ``fulldoc`` / ``anchored@5`` (claim-anchored top-k source
  sentences via :func:`benchmarks._common.select_relevant_passages`);
  dialogue: ``history`` (production today, D1 baseline) /
  ``knowledge+history``.
- **aggregation** — per-response support from per-claim supports:
  ``min`` (weakest link) / ``mean`` / ``low2mean`` (mean of two weakest) /
  ``coverage`` (fraction of claims supported at 0.5).
- **checker** — ``factcg`` (production default) / ``minicheck``.

The headline metric is **catch at MATCHED per-task FPR** (summarisation
0.025, dialogue 0.045 — the tracked baseline's operating point), plus the
oracle balanced-accuracy threshold so the read is not a one-cut-point
artefact. The WCS-1 gate needs a task-routed config to beat the baseline
on BOTH HaluEval and a second long-context set before any default flips —
this harness is the HaluEval half.

The scoring and sweep logic is pure and takes an injected ``predictor`` /
``rows`` / ``splitter`` so it is exercised offline with fakes; the heavy
models load only in :func:`build_predictor` / :func:`main` on the metered
run (CEO spend gate).

Usage::

    python benchmarks/run_longcontext_bench.py \
        --samples 200 --checker factcg \
        --matrix-out benchmarks/results/longcontext_matrix_factcg.json \
        --out benchmarks/results/longcontext_sweep_factcg.json

    # offline re-sweep, no model, no GPU
    python benchmarks/run_longcontext_bench.py \
        --sweep-from benchmarks/results/longcontext_matrix_factcg.json \
        --out benchmarks/results/longcontext_sweep_factcg.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("DirectorAI")

#: Evidence variants per task; the FIRST entry is the production baseline.
TASK_VARIANTS: dict[str, tuple[str, ...]] = {
    "summarization": ("prefix3000", "fulldoc", "anchored@5"),
    "dialogue": ("history", "knowledge+history"),
}

#: Per-response aggregations over per-claim support scores.
AGGREGATIONS: tuple[str, ...] = ("min", "mean", "low2mean", "coverage")

#: Matched-FPR targets from the tracked _200 baseline
#: (benchmarks/results/judge_bench_nli_only_200.json, threshold 0.5).
MATCHED_FPR: dict[str, float] = {"summarization": 0.025, "dialogue": 0.045}

#: Baseline catch rates at the matched FPR (same artefact) for the delta read.
BASELINE_CATCH: dict[str, float] = {"summarization": 0.12, "dialogue": 0.045}

#: Claim-level support cut used by the ``coverage`` aggregation.
COVERAGE_CUT: float = 0.5

#: Candidate thresholds for the oracle balanced-accuracy sweep.
_SWEEP_THRESHOLDS: tuple[float, ...] = tuple(i / 100 for i in range(5, 100, 5))

_PREFIX_CHARS = 3000


class Predictor(Protocol):
    """Minimal checker interface: P(supported) for a (premise, claim)."""

    def score(self, premise: str, hypothesis: str) -> float: ...


# ── Rows ─────────────────────────────────────────────────────────────────


@dataclass
class Row:
    """One (evidence, response, label) case of a HaluEval long-context task."""

    task: str
    doc: str
    history: str
    response: str
    hallucinated: bool


def rows_from_halueval(task: str, samples: Sequence[dict[str, Any]]) -> list[Row]:
    """Build paired rows (right + hallucinated response) for *task*.

    Dialogue rows carry BOTH ``knowledge`` (as ``doc``) and
    ``dialogue_history`` (as ``history``) so the evidence variants can
    compose either premise — the D1 finding is exactly that the production
    path drops ``knowledge`` whenever a history exists.
    """
    if task not in TASK_VARIANTS:
        raise ValueError(f"unsupported task {task!r}")
    rows: list[Row] = []
    for sample in samples:
        if task == "summarization":
            doc = sample.get("document", "") or ""
            for key, is_bad in (
                ("right_summary", False),
                ("hallucinated_summary", True),
            ):
                resp = sample.get(key, "") or ""
                if doc and resp:
                    rows.append(Row(task, doc, "", resp, is_bad))
        else:
            doc = sample.get("knowledge", "") or ""
            history = sample.get("dialogue_history", "") or ""
            for key, is_bad in (
                ("right_response", False),
                ("hallucinated_response", True),
            ):
                resp = sample.get(key, "") or ""
                if (doc or history) and resp:
                    rows.append(Row(task, doc, history, resp, is_bad))
    return rows


# ── Evidence composition ─────────────────────────────────────────────────


def build_premise(row: Row, variant: str, claim: str) -> str:
    """Compose the evidence premise for one (row, variant, claim).

    ``anchored@k`` selects the k source sentences most lexically relevant
    to *claim* over the WHOLE document (the D2 fix family); an empty
    selection degrades to the production prefix so the claim is always
    scored against something.
    """
    if variant == "prefix3000":
        return row.doc[:_PREFIX_CHARS]
    if variant == "fulldoc":
        return row.doc
    if variant.startswith("anchored@"):
        from benchmarks._common import select_relevant_passages

        top_k = int(variant.split("@", 1)[1])
        passages = select_relevant_passages(row.doc, claim, top_k=top_k)
        return "\n".join(passages) if passages else row.doc[:_PREFIX_CHARS]
    if variant == "history":
        return row.history
    if variant == "knowledge+history":
        return f"{row.doc}\n{row.history}" if row.doc else row.history
    raise ValueError(f"unknown evidence variant {variant!r}")


# ── Score matrix ─────────────────────────────────────────────────────────


@dataclass
class MatrixEntry:
    """Per-claim support scores for one row under every evidence variant."""

    task: str
    hallucinated: bool
    scores: dict[str, list[float]] = field(default_factory=dict)


def score_matrix(
    rows: Sequence[Row],
    predictor: Predictor,
    *,
    splitter: Callable[[str], list[str]],
    variants: dict[str, tuple[str, ...]] | None = None,
) -> list[MatrixEntry]:
    """Score every (row, variant, claim) triple once.

    Predictor calls are memoised on the exact ``(premise, claim)`` pair, so
    shared premises (prefix/fulldoc/history variants, and the right/
    hallucinated response pair of one sample) hit the checker once.
    """
    variants = variants or TASK_VARIANTS
    cache: dict[tuple[str, str], float] = {}

    def cached_score(premise: str, claim: str) -> float:
        key = (premise, claim)
        if key not in cache:
            cache[key] = float(predictor.score(premise, claim))
        return cache[key]

    matrix: list[MatrixEntry] = []
    for row in rows:
        claims = [c for c in splitter(row.response) if c.strip()] or [row.response]
        entry = MatrixEntry(task=row.task, hallucinated=row.hallucinated)
        for variant in variants[row.task]:
            entry.scores[variant] = [
                cached_score(build_premise(row, variant, claim), claim)
                for claim in claims
            ]
        matrix.append(entry)
    logger.info(
        "score_matrix: %d rows, %d unique predictor calls", len(matrix), len(cache)
    )
    return matrix


def matrix_to_json(matrix: Sequence[MatrixEntry]) -> list[dict[str, Any]]:
    return [
        {"task": e.task, "hallucinated": e.hallucinated, "scores": e.scores}
        for e in matrix
    ]


def matrix_from_json(data: Sequence[dict[str, Any]]) -> list[MatrixEntry]:
    return [
        MatrixEntry(
            task=d["task"],
            hallucinated=bool(d["hallucinated"]),
            scores={k: [float(x) for x in v] for k, v in d["scores"].items()},
        )
        for d in data
    ]


# ── Aggregation + operating points ───────────────────────────────────────


def aggregate(scores: Sequence[float], agg: str) -> float:
    """Collapse per-claim supports into one response support in [0, 1]."""
    if not scores:
        raise ValueError("aggregate() needs at least one claim score")
    if agg == "min":
        return min(scores)
    if agg == "mean":
        return sum(scores) / len(scores)
    if agg == "low2mean":
        low = sorted(scores)[:2]
        return sum(low) / len(low)
    if agg == "coverage":
        return sum(1 for s in scores if s >= COVERAGE_CUT) / len(scores)
    raise ValueError(f"unknown aggregation {agg!r}")


def threshold_at_matched_fpr(good_scores: Sequence[float], target_fpr: float) -> float:
    """Largest threshold flagging at most *target_fpr* of good responses.

    A response is flagged hallucinated iff ``support < threshold``, so the
    threshold is the (⌊target·n⌋)-th smallest good score — every good score
    strictly below it is a false positive, and there are at most ⌊target·n⌋
    of those by construction.
    """
    if not good_scores:
        raise ValueError("threshold_at_matched_fpr() needs good scores")
    allowed = int(target_fpr * len(good_scores))
    return sorted(good_scores)[allowed]


def _balanced_accuracy(
    good_scores: Sequence[float], bad_scores: Sequence[float], threshold: float
) -> float:
    if not good_scores or not bad_scores:
        return 0.0
    tnr = sum(1 for s in good_scores if s >= threshold) / len(good_scores)
    tpr = sum(1 for s in bad_scores if s < threshold) / len(bad_scores)
    return (tpr + tnr) / 2


def operating_points(
    good_scores: Sequence[float],
    bad_scores: Sequence[float],
    target_fpr: float,
) -> dict[str, float]:
    """Matched-FPR catch + oracle balanced accuracy for one config."""
    t = threshold_at_matched_fpr(good_scores, target_fpr)
    catch = sum(1 for s in bad_scores if s < t) / max(len(bad_scores), 1)
    fpr = sum(1 for s in good_scores if s < t) / max(len(good_scores), 1)
    oracle_t, oracle_ba = max(
        (
            (c, _balanced_accuracy(good_scores, bad_scores, c))
            for c in _SWEEP_THRESHOLDS
        ),
        key=lambda tb: tb[1],
    )
    return {
        "threshold_at_matched_fpr": float(t),
        "catch_at_matched_fpr": catch,
        "actual_fpr": fpr,
        "oracle_threshold": float(oracle_t),
        "oracle_balanced_accuracy": oracle_ba,
    }


# ── Sweep + report ───────────────────────────────────────────────────────


def summarise(matrix: Sequence[MatrixEntry]) -> dict[str, Any]:
    """Sweep every (task, variant, aggregation) config over the matrix."""
    report: dict[str, Any] = {}
    for task, variants in TASK_VARIANTS.items():
        entries = [e for e in matrix if e.task == task]
        if not entries:
            continue
        good = [e for e in entries if not e.hallucinated]
        bad = [e for e in entries if e.hallucinated]
        grid: dict[str, dict[str, Any]] = {}
        best: dict[str, Any] | None = None
        for variant in variants:
            grid[variant] = {}
            for agg in AGGREGATIONS:
                pts = operating_points(
                    [aggregate(e.scores[variant], agg) for e in good],
                    [aggregate(e.scores[variant], agg) for e in bad],
                    MATCHED_FPR[task],
                )
                grid[variant][agg] = pts
                if (
                    best is None
                    or pts["catch_at_matched_fpr"] > best["catch_at_matched_fpr"]
                ):
                    best = {"variant": variant, "aggregation": agg, **pts}
        report[task] = {
            "n_good": len(good),
            "n_bad": len(bad),
            "matched_fpr_target": MATCHED_FPR[task],
            "baseline_catch_tracked_200": BASELINE_CATCH[task],
            "grid": grid,
            "best_by_matched_catch": best,
        }
    return report


# ── Heavy collaborators (metered run only) ───────────────────────────────


class MiniCheckPredictor:
    """MiniCheck adapter over the production NLIScorer backend.

    ``NLIScorer._minicheck_score`` returns a DIVERGENCE (1 − support), so
    this adapter inverts it back to P(supported). Loading MiniCheck needs
    the manual install documented in pyproject's ``minicheck`` extra.
    """

    def __init__(self) -> None:  # pragma: no cover - GPU/model path
        from director_ai.core.scoring.nli import NLIScorer

        self._scorer = NLIScorer(backend="minicheck")

    def score(self, premise: str, hypothesis: str) -> float:  # pragma: no cover
        return 1.0 - float(self._scorer._minicheck_score(premise, hypothesis))


def build_predictor(checker: str, *, nli_model: str | None = None) -> Predictor:
    """Construct the requested heavy checker (metered run only)."""
    if checker == "factcg":
        from benchmarks.aggrefact_eval import _BinaryNLIPredictor

        return _BinaryNLIPredictor(model_name=nli_model)
    if checker == "minicheck":
        return MiniCheckPredictor()
    raise ValueError(f"unknown checker {checker!r}")


def load_rows(tasks: Sequence[str], max_samples: int | None) -> list[Row]:
    """Load HaluEval rows for *tasks* via the cached benchmark downloader."""
    from benchmarks._halueval_data import _download_task_data

    rows: list[Row] = []
    for task in tasks:
        samples = _download_task_data(task)
        if max_samples:
            samples = samples[:max_samples]
        rows.extend(rows_from_halueval(task, samples))
    return rows


def load_ragtruth_rows(max_samples: int | None) -> list[Row]:
    """Load RAGTruth Summary test rows (the WCS-1 second long-context set).

    RAGTruth's labels are natural (unpaired): each model response over a
    source document is hallucinated iff any span was annotated. Rows map
    onto the ``summarization`` task, so the evidence variants and matched
    FPR target are shared with the HaluEval sweep; ``max_samples`` caps
    the response count in corpus order.
    """
    from benchmarks._ragtruth_data import load_summary_rows

    raw = load_summary_rows("test")
    if max_samples:
        raw = raw[:max_samples]
    return [
        Row("summarization", r["doc"], "", r["response"], r["hallucinated"])
        for r in raw
    ]


# ── Orchestration ────────────────────────────────────────────────────────


def run_longcontext_bench(
    *,
    dataset: str = "halueval",
    tasks: Sequence[str] = ("summarization", "dialogue"),
    max_samples: int | None = None,
    checker: str = "factcg",
    nli_model: str | None = None,
    rows: Sequence[Row] | None = None,
    predictor: Predictor | None = None,
    splitter: Callable[[str], list[str]] | None = None,
    matrix: Sequence[MatrixEntry] | None = None,
) -> tuple[dict[str, Any], list[MatrixEntry]]:
    """Score (or reuse) the matrix and sweep it; return (report, matrix).

    Injectable ``rows`` / ``predictor`` / ``splitter`` / ``matrix`` keep the
    function testable offline; unset arguments are built from the real
    dataset and models.
    """
    t0 = time.perf_counter()
    if matrix is None:
        if splitter is None:
            from director_ai.core.text_segmentation import split_sentences

            splitter = split_sentences
        if rows is None:
            if dataset == "halueval":
                rows = load_rows(tasks, max_samples)
            elif dataset == "ragtruth":
                rows = load_ragtruth_rows(max_samples)
            else:
                raise ValueError(f"unknown dataset {dataset!r}")
        if predictor is None:
            predictor = build_predictor(checker, nli_model=nli_model)
        matrix = score_matrix(rows, predictor, splitter=splitter)

    report = summarise(matrix)
    meta: dict[str, Any] = {
        "benchmark": "longcontext-evidence-agg-sweep",
        "dataset": dataset,
        "checker": checker,
        "nli_model": nli_model or os.environ.get("DIRECTOR_NLI_MODEL", "default"),
        "rows": len(matrix),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "matched_fpr_source": "benchmarks/results/judge_bench_nli_only_200.json",
    }
    try:
        from benchmarks.host_conditions import host_conditions, isolation_verdict

        conditions = host_conditions()
        meta["host_conditions"] = conditions
        meta["isolation_verdict"] = isolation_verdict(conditions)
    except Exception as exc:  # pragma: no cover - host probe best-effort
        meta["host_conditions_error"] = str(exc)
    return {"meta": meta, "per_task": report}, list(matrix)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--dataset", choices=("halueval", "ragtruth"), default="halueval"
    )
    parser.add_argument("--tasks", default="summarization,dialogue")
    parser.add_argument("--checker", choices=("factcg", "minicheck"), default="factcg")
    parser.add_argument("--nli-model", default=None)
    parser.add_argument("--matrix-out", default=None, help="dump the score matrix")
    parser.add_argument(
        "--sweep-from", default=None, help="re-sweep an existing matrix JSON offline"
    )
    parser.add_argument("--out", default="benchmarks/results/longcontext_sweep.json")
    args = parser.parse_args(argv)

    matrix: list[MatrixEntry] | None = None
    if args.sweep_from:
        payload = json.loads(Path(args.sweep_from).read_text(encoding="utf-8"))
        matrix = matrix_from_json(payload["matrix"])

    report, matrix = run_longcontext_bench(
        dataset=args.dataset,
        tasks=tuple(t.strip() for t in args.tasks.split(",") if t.strip()),
        max_samples=args.samples,
        checker=args.checker,
        nli_model=args.nli_model,
        matrix=matrix,
    )

    if args.matrix_out:
        mpath = Path(args.matrix_out)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(
            json.dumps({"meta": report["meta"], "matrix": matrix_to_json(matrix)})
            + "\n",
            encoding="utf-8",
        )
        logger.info("Saved matrix to %s", mpath)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved report to %s", out)

    for task, block in report["per_task"].items():
        best = block["best_by_matched_catch"]
        print(
            f"{task}: baseline catch {block['baseline_catch_tracked_200']:.3f} -> "
            f"best {best['catch_at_matched_fpr']:.3f} "
            f"({best['variant']} / {best['aggregation']}, "
            f"FPR {best['actual_fpr']:.3f} <= {block['matched_fpr_target']:.3f})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
