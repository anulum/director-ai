# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — recall-correctness feedback wiring tests

"""Verify the opt-in wiring of REMANENTIA recall-correctness feedback.

Covers the CoherenceAgent hook (default None preserves behaviour; a configured
client receives exactly one verdict-derived outcome per processed prompt, and a
client that raises never breaks answering) and the config builder (off by
default, refuses a non-remanentia backend with a warning, and otherwise builds a
client carrying the configured base URL, token, and timeout).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from director_ai.core.agent import CoherenceAgent
from director_ai.core.calibration.recall_correctness import (
    RecallOutcome,
    correctness_from_verdict,
)
from director_ai.core.config import DirectorConfig
from director_ai.core.config_builders import build_correctness_feedback


@dataclass
class _FakeClient:
    """Captures try_record calls; optionally raises to exercise hot-path safety."""

    raises: bool = False
    calls: list[RecallOutcome] = field(default_factory=list)

    def try_record(self, outcome: RecallOutcome) -> str | None:
        self.calls.append(outcome)
        if self.raises:
            raise RuntimeError("must never propagate")
        return "evt"


_PROMPT = "Paris is the capital of France."


# --- agent hook -------------------------------------------------------------


def test_default_agent_does_not_report() -> None:
    """With no client configured, process() runs unchanged and reports nothing."""
    agent = CoherenceAgent()
    result = agent.process(_PROMPT)
    assert result.output  # behaviour preserved: an answer is produced


def test_agent_reports_derived_outcome_once() -> None:
    """A configured client gets exactly one outcome matching the verdict."""
    client = _FakeClient()
    agent = CoherenceAgent(correctness_feedback=client)
    result = agent.process(_PROMPT)

    assert len(client.calls) == 1
    outcome = client.calls[0]
    assert outcome.query == _PROMPT
    assert outcome.was_correct == correctness_from_verdict(result)


def test_agent_reporting_failure_never_propagates() -> None:
    """A raising client cannot break answering (try_record owns the swallow)."""
    client = _FakeClient(raises=True)
    agent = CoherenceAgent(correctness_feedback=client)
    result = agent.process(_PROMPT)
    assert result.output
    assert len(client.calls) == 1


# --- config builder ---------------------------------------------------------


def test_builder_off_by_default() -> None:
    cfg = DirectorConfig(vector_backend="remanentia")
    assert build_correctness_feedback(cfg) is None


def test_builder_requires_remanentia_backend(caplog) -> None:
    cfg = DirectorConfig(vector_backend="chroma", remanentia_correctness_feedback=True)
    with caplog.at_level("WARNING"):
        assert build_correctness_feedback(cfg) is None
    assert "no recall ledger to label" in caplog.text


def test_builder_constructs_client_with_config() -> None:
    cfg = DirectorConfig(
        vector_backend="remanentia",
        remanentia_correctness_feedback=True,
        remanentia_base_url="http://127.0.0.1:8001/mem",
        remanentia_token="secret-tok",
        remanentia_timeout_s=3.0,
    )
    client = build_correctness_feedback(cfg)
    assert client is not None
    # The client carries the configured transport parameters.
    assert client._host == "127.0.0.1"
    assert client._port == 8001
    assert client._base_path == "/mem"
    assert client._token == "secret-tok"
    assert client._timeout_s == 3.0


def test_config_method_delegates_to_builder() -> None:
    """DirectorConfig.build_correctness_feedback delegates to the builder."""
    off = DirectorConfig(vector_backend="remanentia")
    assert off.build_correctness_feedback() is None
    on = DirectorConfig(
        vector_backend="remanentia",
        remanentia_correctness_feedback=True,
        remanentia_token="t",
    )
    assert on.build_correctness_feedback() is not None
