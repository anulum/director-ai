# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multi-turn transcript runner tests
"""Tests for the multi-turn transcript runner and the full grounding chain.

Covers single- and multi-turn runs, the dialogue threaded back into each prompt,
the system-prompt inclusion/omission, the empty-seed guard, empty-candidate and
whitespace handling, ``Transcript.full_text`` / ``to_dict``, and one end-to-end
test that drives a stub generator's citing answer through the real
resolve_citations → SourceFetcher → CitationGroundingJudge chain with stubs.
"""

from __future__ import annotations

import pytest

from director_ai.core.citation_grounding import (
    CitationGroundingJudge,
    ExchangeTurn,
    MultiTurnRunner,
    SourceFetcher,
    Transcript,
    resolve_citations,
)


class _ScriptedGenerator:
    """Returns a configured response per turn and records the prompts it saw."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate_candidates(self, prompt, n):
        self.prompts.append(prompt)
        idx = min(len(self.prompts) - 1, len(self._responses) - 1)
        return [{"text": self._responses[idx]}]


class TestRun:
    def test_single_turn(self):
        runner = MultiTurnRunner(generator=_ScriptedGenerator(["An answer."]))
        transcript = runner.run("A question?")
        assert len(transcript.turns) == 1
        assert transcript.turns[0] == ExchangeTurn("A question?", "An answer.")

    def test_multi_turn_order(self):
        gen = _ScriptedGenerator(["r1", "r2", "r3"])
        transcript = MultiTurnRunner(generator=gen).run("seed", ["f1", "f2"])
        assert [t.prompt for t in transcript.turns] == ["seed", "f1", "f2"]
        assert [t.response for t in transcript.turns] == ["r1", "r2", "r3"]

    def test_dialogue_threaded_into_prompt(self):
        gen = _ScriptedGenerator(["first answer", "second answer"])
        MultiTurnRunner(generator=gen, system_prompt="").run("Q1", ["Q2"])
        second_prompt = gen.prompts[1]
        assert "User: Q1" in second_prompt
        assert "Assistant: first answer" in second_prompt
        assert second_prompt.rstrip().endswith("User: Q2\nAssistant:")

    def test_system_prompt_included(self):
        gen = _ScriptedGenerator(["a"])
        MultiTurnRunner(generator=gen, system_prompt="SYS RULES").run("Q")
        assert gen.prompts[0].startswith("SYS RULES")

    def test_system_prompt_omitted(self):
        gen = _ScriptedGenerator(["a"])
        MultiTurnRunner(generator=gen, system_prompt="").run("Q")
        assert gen.prompts[0].startswith("User: Q")

    def test_empty_seed_raises(self):
        with pytest.raises(ValueError, match="seed question"):
            MultiTurnRunner(generator=_ScriptedGenerator(["a"])).run("   ")

    def test_empty_candidates_yield_empty_response(self):
        class _Empty:
            def generate_candidates(self, prompt, n):
                return []

        transcript = MultiTurnRunner(generator=_Empty()).run("Q")
        assert transcript.turns[0].response == ""

    def test_response_is_stripped(self):
        gen = _ScriptedGenerator(["  padded answer  "])
        assert MultiTurnRunner(generator=gen).run("Q").turns[0].response == (
            "padded answer"
        )


class TestTranscript:
    def test_full_text_concatenates_nonempty_responses(self):
        transcript = Transcript(
            (ExchangeTurn("q1", "r1"), ExchangeTurn("q2", ""), ExchangeTurn("q3", "r3"))
        )
        assert transcript.full_text == "r1\n\nr3"

    def test_to_dict(self):
        transcript = Transcript((ExchangeTurn("q", "r"),))
        assert transcript.to_dict() == {"turns": [{"prompt": "q", "response": "r"}]}

    def test_exchange_turn_to_dict(self):
        assert ExchangeTurn("q", "r").to_dict() == {"prompt": "q", "response": "r"}


class _StubHttp:
    def __init__(self, routes):
        self.routes = routes

    def get(self, url, *, headers=None):
        for needle, resp in self.routes.items():
            if needle in url:
                return resp
        return 404, b"", ""


class _Scorer:
    """Entails when the evidence shares the claim's key term."""

    def score(self, premise, hypothesis):
        return 0.1 if "equation of state" in premise else 0.95


def test_end_to_end_runner_fetch_judge():
    # A model answers with one grounded claim and one cited-but-false claim.
    answer = (
        "Mass and radius constrain the equation of state [1]. "
        "The sky is green and made of cheese [2].\n\n"
        "References:\n"
        "[1] Bogdanov 10.3847/2041-8213/ab50c5\n"
        "[2] Nonsense arXiv:2411.04368\n"
    )
    runner = MultiTurnRunner(generator=_ScriptedGenerator([answer]))
    transcript = runner.run("How is the neutron-star EOS constrained?")

    crossref = (
        b'{"message":{"title":["PSR J0030"],'
        b'"abstract":"Mass-radius measurements constrain the equation of state."}}'
    )
    arxiv = (
        b"<feed><entry><title>SimpleQA</title>"
        b"<summary>A factuality benchmark, nothing about cheese.</summary></entry></feed>"
    )
    fetcher = SourceFetcher(
        http=_StubHttp(
            {
                "crossref": (200, crossref, "application/json"),
                "arxiv": (200, arxiv, "application/atom+xml"),
            }
        )
    )
    citations = resolve_citations(transcript.full_text)
    sources = fetcher.fetch_all(citations)

    report = CitationGroundingJudge(scorer=_Scorer()).assess(
        transcript.full_text, sources
    )
    assert report.total == 2
    assert report.grounded_fraction == pytest.approx(0.5)  # one grounded, one not
    assert any("cheese" in c.claim for c in report.hallucinated)
