#!/usr/bin/env python3
"""
Use Director-AI as a LangChain output checker.

Requires:
    pip install director-ai langchain langchain-openai

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/langchain_guard.py
"""

from __future__ import annotations

import os

from director_ai.core import CoherenceScorer, GroundTruthStore


class DirectorGuard:
    """LangChain-compatible output guard using Director-AI coherence scoring."""

    def __init__(self, facts: dict[str, str] | None = None, threshold: float = 0.6):
        self.store = GroundTruthStore()
        if facts:
            for k, v in facts.items():
                self.store.facts[k] = v
        self.scorer = CoherenceScorer(
            threshold=threshold, ground_truth_store=self.store
        )

    def check(self, query: str, response: str) -> dict:
        approved, score = self.scorer.review(query, response)
        return {
            "approved": approved,
            "score": score.score,
            "h_logical": score.h_logical,
            "h_factual": score.h_factual,
        }


def with_langchain():
    """Score LangChain LLM outputs through Director-AI."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    guard = DirectorGuard(
        facts={
            "company founding": "Acme Corp was founded in 2015.",
            "employee count": "Acme Corp has 150 employees.",
            "headquarters": "Acme Corp is headquartered in Zurich.",
        },
        threshold=0.6,
    )

    queries = [
        "When was Acme Corp founded?",
        "How many employees does Acme Corp have?",
        "Where is Acme Corp headquartered?",
    ]

    for query in queries:
        response = llm.invoke(query)
        result = guard.check(query, response.content)
        status = "PASS" if result["approved"] else "FAIL"
        print(f"[{status}] {query}")
        print(f"  Answer: {response.content[:120]}")
        print(f"  Score:  {result['score']:.3f}")
        print()


def standalone_demo():
    """Demo the guard without LangChain (no API key needed)."""
    guard = DirectorGuard(
        facts={
            "capital of France": "The capital of France is Paris.",
            "boiling point": "Water boils at 100 degrees Celsius.",
        }
    )

    tests = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of France?", "The capital of France is Berlin."),
        ("At what temperature does water boil?", "Water boils at 100 C."),
    ]

    for query, response in tests:
        result = guard.check(query, response)
        status = "PASS" if result["approved"] else "FAIL"
        print(f"[{status}] Q: {query}")
        print(f"         A: {response}")
        print(f"         Score: {result['score']:.3f}")
        print()


if __name__ == "__main__":
    print("--- DirectorGuard standalone demo ---\n")
    standalone_demo()

    if os.environ.get("OPENAI_API_KEY"):
        print("--- LangChain + Director-AI ---\n")
        with_langchain()
    else:
        print("Set OPENAI_API_KEY to run the LangChain example.")
