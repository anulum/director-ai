# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HalluHard citation-grounding benchmark

"""Evaluate citation grounding on HalluHard (EPFL, arXiv:2602.01031).

HalluHard scores whether a model's multi-turn answers cite sources that actually
support their factual claims. This harness ties the four
``core.citation_grounding`` components together: for each seed question it runs a
model through a short conversation (:class:`MultiTurnRunner`), resolves the
citations in the transcript, fetches the cited sources
(:class:`SourceFetcher`), and judges each assertion's grounding
(:class:`CitationGroundingJudge`). It then aggregates groundedness and
citation-coverage across the dataset.

The dataset is the released ``epfml/halluhard`` JSONL (e.g.
``research_questions/data/research_questions_all.jsonl``), whose records carry a
``research_question`` plus the source paper (``doi`` / ``arxiv_id`` / ``title`` /
``abstract``). Pass a local copy via ``source``; the harness reads only the
``research_question`` field as the seed.

The generator, scorer, and fetcher are all injected, so the orchestration is
testable offline with stubs; a real run wires an ``LLMGenerator``, an
``NLIScorer``, and the default ``SourceFetcher``.

Usage::

    python -m benchmarks.halluhard_eval --source research_questions_all.jsonl \\
        --api-url http://127.0.0.1:8081/v1 --nli --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks._common import save_results
from director_ai.core.citation_grounding import (
    CitationGroundingJudge,
    MultiTurnRunner,
    SourceFetcher,
    resolve_citations,
)

logger = logging.getLogger("DirectorAI.Benchmark.HalluHard")

DEFAULT_FOLLOWUPS = (
    "Can you provide more detail and cite specific sources for each claim?",
    "Are you certain about those claims? Please double-check each one.",
)


def _load_halluhard(
    max_samples: int | None = None, *, source: str
) -> list[dict[str, str]]:
    """Load HalluHard records (``research_question`` field) from a local JSONL."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"HalluHard source not found: {path}")
    records: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        question = str(obj.get("research_question", "")).strip()
        if question:
            records.append({"research_question": question})
    if max_samples is not None:
        records = records[:max_samples]
    return records


@dataclass
class HalluHardSample:
    """Grounding outcome for one seed question's transcript."""

    question: str
    n_claims: int
    n_cited: int
    n_grounded: int

    @property
    def grounded_fraction(self) -> float:
        return self.n_grounded / self.n_claims if self.n_claims else 1.0

    @property
    def citation_coverage(self) -> float:
        return self.n_cited / self.n_claims if self.n_claims else 1.0


@dataclass
class HalluHardMetrics:
    """Aggregated HalluHard groundedness metrics."""

    samples: list[HalluHardSample] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def total_claims(self) -> int:
        return sum(s.n_claims for s in self.samples)

    @property
    def grounded_rate(self) -> float:
        """Fraction of all assertions that are grounded (micro-averaged)."""
        claims = self.total_claims
        return sum(s.n_grounded for s in self.samples) / claims if claims else 1.0

    @property
    def citation_coverage(self) -> float:
        claims = self.total_claims
        return sum(s.n_cited for s in self.samples) / claims if claims else 1.0

    @property
    def hallucination_rate(self) -> float:
        """Fraction of assertions that are not grounded."""
        return 1.0 - self.grounded_rate

    def to_dict(self) -> dict[str, object]:
        return {
            "total_questions": self.total,
            "total_claims": self.total_claims,
            "grounded_rate": round(self.grounded_rate, 4),
            "citation_coverage": round(self.citation_coverage, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
        }


def run_halluhard(
    records: list[dict[str, str]],
    *,
    generator,
    scorer,
    fetcher: SourceFetcher | None = None,
    followups: tuple[str, ...] = DEFAULT_FOLLOWUPS,
    support_threshold: float = 0.6,
) -> HalluHardMetrics:
    """Run the gen → resolve → fetch → judge chain over ``records``.

    ``generator`` is the model under test (any ``Generator``), ``scorer`` the NLI
    backend (any ``Scorer``). The cited sources are fetched per transcript and
    each assertion judged grounded only when its citation's source supports it.
    """
    runner = MultiTurnRunner(generator=generator)
    judge = CitationGroundingJudge(scorer=scorer, support_threshold=support_threshold)
    src_fetcher = fetcher if fetcher is not None else SourceFetcher()

    metrics = HalluHardMetrics()
    for record in records:
        question = record["research_question"]
        transcript = runner.run(question, followups)
        # Fetch every cited source once across the whole conversation, then judge
        # each turn against its own citations. Per-turn assessment keeps each
        # turn's body/reference split intact (a transcript that concatenates
        # several turns would have several reference sections).
        sources = src_fetcher.fetch_all(resolve_citations(transcript.full_text))
        n_claims = n_cited = n_grounded = 0
        for turn in transcript.turns:
            report = judge.assess(turn.response, sources)
            n_claims += report.total
            n_cited += sum(1 for c in report.claims if c.has_citation)
            n_grounded += sum(1 for c in report.claims if c.grounded)
        metrics.samples.append(
            HalluHardSample(
                question=question,
                n_claims=n_claims,
                n_cited=n_cited,
                n_grounded=n_grounded,
            )
        )
    logger.info(
        "HalluHard: %d questions, grounded_rate=%.3f",
        metrics.total,
        metrics.grounded_rate,
    )
    return metrics


def _print_results(metrics: HalluHardMetrics) -> None:
    print("\n" + "=" * 64)
    print("HalluHard Citation-Grounding Benchmark")
    print("=" * 64)
    d = metrics.to_dict()
    print(f"  Questions:          {d['total_questions']}")
    print(f"  Assertions:         {d['total_claims']}")
    print(f"  Grounded rate:      {d['grounded_rate']:.1%}")
    print(f"  Citation coverage:  {d['citation_coverage']:.1%}")
    print(f"  Hallucination rate: {d['hallucination_rate']:.1%}")
    print("=" * 64)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="HalluHard grounding benchmark")
    parser.add_argument("--source", required=True, help="local HalluHard JSONL path")
    parser.add_argument("--api-url", required=True, help="OpenAI-style LLM endpoint")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--nli", action="store_true", help="use model-backed NLI")
    parser.add_argument("--mailto", type=str, default="", help="Crossref polite pool")
    args = parser.parse_args()

    from director_ai.core.actor import LLMGenerator
    from director_ai.core.scoring.nli import NLIScorer

    records = _load_halluhard(args.max_samples, source=args.source)
    metrics = run_halluhard(
        records,
        generator=LLMGenerator(api_url=args.api_url),
        scorer=NLIScorer(use_model=args.nli),
        fetcher=SourceFetcher(mailto=args.mailto),
    )
    _print_results(metrics)
    save_results(metrics.to_dict(), "halluhard_results.json")


if __name__ == "__main__":
    main()
