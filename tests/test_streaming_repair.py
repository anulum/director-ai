# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for streaming repair.

Covers lossless clause splitting, the repair engine's keep/rewrite/redact paths
with injected scoring/retrieval/rewriting, repair-event emission, metrics, and
the ProductionGuard integration.
"""

from __future__ import annotations

import pytest

from director_ai.core.metrics import metrics
from director_ai.core.streaming_repair import (
    RepairAction,
    RepairResult,
    StreamingRepairer,
    join_clauses,
    split_clauses,
)


def _reject_robot(clause: str) -> float:
    return 0.1 if "robot" in clause.lower() else 0.9


# ── clause splitting ────────────────────────────────────────────────────


class TestClauses:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "no terminator",
            "One sentence.",
            "One. Two. Three.",
            "Question? Yes! Done.",
            "Line one.\nLine two.",
            "Trailing spaces.   And more.   ",
            "Multiple!!! Bangs??? Here.",
        ],
    )
    def test_split_is_lossless(self, text):
        assert join_clauses(split_clauses(text)) == text

    def test_empty_yields_no_clauses(self):
        assert split_clauses("") == []

    def test_multi_sentence_count(self):
        assert len(split_clauses("A. B. C.")) == 3


# ── RepairAction ────────────────────────────────────────────────────────


class TestRepairAction:
    def test_bad_action_rejected(self):
        with pytest.raises(ValueError, match="unsupported repair action"):
            RepairAction(0, "explode", 0.5)

    def test_repaired_flag(self):
        assert RepairAction(0, "rewrite", 0.1).repaired
        assert RepairAction(0, "redact", 0.1).repaired
        assert not RepairAction(0, "keep", 0.9).repaired

    def test_to_dict(self):
        action = RepairAction(
            2, "rewrite", 0.1, evidence_ids=("vector:d1",), reason="x"
        )
        payload = action.to_dict()
        assert payload["clause_index"] == 2
        assert payload["action"] == "rewrite"
        assert payload["evidence_ids"] == ["vector:d1"]


# ── repairer ────────────────────────────────────────────────────────────


class TestStreamingRepairer:
    def test_threshold_validation(self):
        with pytest.raises(ValueError, match="threshold must be"):
            StreamingRepairer(_reject_robot, threshold=2.0)

    def test_all_supported_unchanged(self):
        text = "Refunds take 30 days. Contact support."
        result = StreamingRepairer(_reject_robot).repair(text)
        assert result.repaired_text == text
        assert not result.repaired
        assert all(a.action == "keep" for a in result.actions)

    def test_unsupported_redacted_without_rewrite(self):
        text = "Refunds take 30 days. The CEO is a robot. Contact support."
        result = StreamingRepairer(_reject_robot, threshold=0.6).repair(text)
        assert result.repaired
        assert result.repaired_count == 1
        assert "robot" not in result.repaired_text
        assert "[removed:" in result.repaired_text

    def test_unsupported_rewritten_with_evidence(self):
        def retrieve(_clause):
            return [{"id": "vector:ceo", "text": "The CEO is Jane Doe."}]

        def rewrite(_clause, evidence):
            return "The CEO is Jane Doe."

        text = "The CEO is a robot. Contact support."
        repairer = StreamingRepairer(
            _reject_robot, threshold=0.6, retrieve_fn=retrieve, rewrite_fn=rewrite
        )
        result = repairer.repair(text)
        assert result.repaired_text.startswith("The CEO is Jane Doe.")
        action = result.actions[0]
        assert action.action == "rewrite"
        assert action.evidence_ids == ("vector:ceo",)

    def test_rewrite_without_evidence_falls_back_to_redact(self):
        def retrieve(_clause):
            return []

        def rewrite(_clause, evidence):  # pragma: no cover - never called
            return "should not be used"

        text = "The CEO is a robot. Done."
        repairer = StreamingRepairer(
            _reject_robot, threshold=0.6, retrieve_fn=retrieve, rewrite_fn=rewrite
        )
        result = repairer.repair(text)
        assert result.actions[0].action == "redact"

    def test_blank_rewrite_falls_back_to_redact(self):
        def retrieve(_clause):
            return [{"id": "e1", "text": "ev"}]

        def rewrite(_clause, evidence):
            return "   "

        text = "The CEO is a robot. Done."
        repairer = StreamingRepairer(
            _reject_robot, threshold=0.6, retrieve_fn=retrieve, rewrite_fn=rewrite
        )
        result = repairer.repair(text)
        assert result.actions[0].action == "redact"

    def test_trailing_whitespace_preserved_on_redact(self):
        text = "The CEO is a robot.   Done."
        result = StreamingRepairer(_reject_robot, threshold=0.6).repair(text)
        # The three spaces between sentences survive the redaction.
        assert "]   Done." in result.repaired_text

    def test_whitespace_only_input_kept_unscored(self):
        # A pure-whitespace segment is preserved verbatim and never scored.
        text = "   "
        result = StreamingRepairer(_reject_robot).repair(text)
        assert result.repaired_text == text
        assert result.actions == ()

    def test_empty_text(self):
        result = StreamingRepairer(_reject_robot).repair("")
        assert result.repaired_text == ""
        assert result.actions == ()

    def test_event_emitted_per_fix(self):
        text = "The CEO is a robot. A second robot too."
        result = StreamingRepairer(_reject_robot, threshold=0.6).repair(
            text, tenant_id="acme", request_id="req-1"
        )
        assert len(result.events) == 2
        event = result.events[0]
        assert event.policy_decision == "warn"
        assert event.hook_id == "streaming_repair"
        assert event.tenant_id == "acme"
        assert event.request_id == "req-1"
        assert event.attributes["action"] == "redact"

    def test_observed_score_clamped_in_event(self):
        # A score outside [0, 1] must not break the SafetyEvent unit-interval rule.
        result = StreamingRepairer(lambda _c: -5.0, threshold=0.6).repair("Bad clause.")
        assert result.events[0].observed_score == 0.0

    def test_retrieve_accepts_objects(self):
        class Row:
            source = "vector:obj"
            text = "grounded text"

        def retrieve(_clause):
            return [Row()]

        def rewrite(_clause, evidence):
            return "grounded rewrite."

        result = StreamingRepairer(
            _reject_robot, threshold=0.6, retrieve_fn=retrieve, rewrite_fn=rewrite
        ).repair("The CEO is a robot.")
        assert result.actions[0].evidence_ids == ("vector:obj",)

    def test_metrics_counted(self):
        metrics.reset()
        StreamingRepairer(_reject_robot, threshold=0.6).repair(
            "Good one. The CEO is a robot."
        )
        snapshot = metrics.get_metrics()
        assert snapshot["counters"]["streaming_repair_clauses_total"]["total"] == 2.0
        actions = snapshot["counters"]["streaming_repair_actions_total"]
        assert actions["multi_labels"].get('action="redact"') == 1.0
        assert actions["multi_labels"].get('action="keep"') == 1.0


# ── RepairResult ────────────────────────────────────────────────────────


class TestRepairResult:
    def test_to_dict_keeps_text_but_actions_tenant_safe(self):
        text = "The CEO is a robot. Fine."
        result = StreamingRepairer(_reject_robot, threshold=0.6).repair(text)
        payload = result.to_dict()
        assert payload["repaired"] is True
        assert payload["repaired_count"] == 1
        assert "robot" not in str(payload["actions"])
        assert len(payload["events"]) == 1

    def test_empty_result_not_repaired(self):
        result = RepairResult(repaired_text="")
        assert not result.repaired
        assert result.repaired_count == 0


# ── guard integration ───────────────────────────────────────────────────


class TestGuardIntegration:
    def test_repair_stream_keeps_supported_clause(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        guard.load_facts({"refund": "Refunds are available within 30 days."})
        # threshold=0.0 isolates the wiring from NLI-availability scoring noise:
        # every clause is "supported" so the text passes through unchanged.
        text = "Refunds are available within 30 days."
        result = guard.repair_stream(
            "What is the refund window?",
            text,
            tenant_id="acme",
            threshold=0.0,
        )
        assert isinstance(result, RepairResult)
        assert result.repaired_text == text
        assert not result.repaired

    def test_repair_stream_redacts_unsupported_clause(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        # threshold=1.0 forces every clause to be treated as unsupported,
        # exercising the redaction wiring deterministically.
        result = guard.repair_stream(
            "q", "Some claim here.", tenant_id="acme", threshold=1.0
        )
        assert result.repaired
        assert "[removed:" in result.repaired_text
