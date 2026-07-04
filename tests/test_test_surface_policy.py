# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Behavioral tests for module-specific test-surface policy enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.check_test_surface_policy as policy
from tools.check_test_surface_policy import (
    SurfaceClassification,
    find_forbidden_test_surfaces,
    find_unclassified_mock_surfaces,
    validate_classifications,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _write_test(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def test_contract():\n    assert True\n", encoding="utf-8")


def test_policy_rejects_bucket_style_test_file_names(tmp_path: Path) -> None:
    _write_test(tmp_path, "tests/test_hardware_runner.py")
    _write_test(tmp_path, "tests/test_final_push.py")
    _write_test(tmp_path, "tests/new_modules/test_crypto.py")

    offenders = find_forbidden_test_surfaces(tmp_path)

    assert offenders == [
        (Path("tests/new_modules/test_crypto.py"), "new_modules"),
        (Path("tests/test_final_push.py"), "final"),
        (Path("tests/test_final_push.py"), "push"),
    ]


def test_policy_uses_tokens_not_substrings(tmp_path: Path) -> None:
    _write_test(tmp_path, "tests/test_physical_grounding_evaluator.py")
    _write_test(tmp_path, "tests/test_pushdown_automaton.py")

    assert find_forbidden_test_surfaces(tmp_path) == []


def test_policy_rejects_unclassified_mock_surface(tmp_path: Path) -> None:
    path = tmp_path / "tests/test_adapter_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "from unittest.mock import MagicMock",
                "",
                "def test_contract():",
                "    assert MagicMock().called is False",
            ]
        ),
        encoding="utf-8",
    )

    offenders = find_unclassified_mock_surfaces(tmp_path, classifications={})

    assert offenders == [(Path("tests/test_adapter_contract.py"), "unittest.mock")]


def test_policy_rejects_unclassified_private_helper_surface(tmp_path: Path) -> None:
    path = tmp_path / "tests/test_private_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "from director_ai.core.scoring.nli import _normalise_text",
                "",
                "def test_contract():",
                "    assert _normalise_text('A') == 'a'",
            ]
        ),
        encoding="utf-8",
    )

    offenders = find_unclassified_mock_surfaces(tmp_path, classifications={})

    assert offenders == [
        (Path("tests/test_private_contract.py"), "private-director-ai-import")
    ]


def test_policy_accepts_classified_known_violation(tmp_path: Path) -> None:
    path = tmp_path / "tests/test_known_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from unittest.mock import patch\n\ndef test_contract():\n    assert patch\n",
        encoding="utf-8",
    )
    classifications = {
        "tests/test_known_contract.py": SurfaceClassification(
            classification="violation",
            category="external SDK adapter fake",
        )
    }

    assert (
        find_unclassified_mock_surfaces(
            tmp_path,
            classifications=classifications,
        )
        == []
    )


def test_policy_accepts_marked_protocol_fake_with_companion(tmp_path: Path) -> None:
    _write_test(tmp_path, "tests/test_adapter_real_surface.py")
    path = tmp_path / "tests/test_adapter_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# test-surface: approved-protocol-fake",
                "# real-surface-companion: tests/test_adapter_real_surface.py",
                "import sys",
                "",
                "def test_contract():",
                '    assert "missing_sdk" not in sys.modules',
            ]
        ),
        encoding="utf-8",
    )

    assert find_unclassified_mock_surfaces(tmp_path, classifications={}) == []


def test_policy_reports_missing_inline_companion_marker(tmp_path: Path) -> None:
    path = tmp_path / "tests/test_adapter_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# test-surface: approved-protocol-fake",
                "import sys",
                "",
                "def test_contract():",
                '    assert "missing_sdk" not in sys.modules',
            ]
        ),
        encoding="utf-8",
    )

    assert find_unclassified_mock_surfaces(tmp_path, classifications={}) == [
        (
            Path("tests/test_adapter_contract.py"),
            "missing real-surface-companion marker",
        )
    ]


def test_policy_reports_missing_inline_companion_file(tmp_path: Path) -> None:
    path = tmp_path / "tests/test_adapter_contract.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# test-surface: unit-guard-with-companion",
                "# real-surface-companion: tests/test_adapter_real_surface.py",
                "import sys",
                "",
                "def test_contract():",
                '    assert "missing_sdk" not in sys.modules',
            ]
        ),
        encoding="utf-8",
    )

    assert find_unclassified_mock_surfaces(tmp_path, classifications={}) == [
        (
            Path("tests/test_adapter_contract.py"),
            "missing companion tests/test_adapter_real_surface.py",
        )
    ]


def test_policy_handles_missing_tests_directory(tmp_path: Path) -> None:
    assert find_forbidden_test_surfaces(tmp_path) == []
    assert find_unclassified_mock_surfaces(tmp_path, classifications={}) == []


def test_policy_falls_back_when_tokenization_fails() -> None:
    assert policy._code_without_literals("'") == "'"


def test_policy_validates_classification_manifest() -> None:
    classifications = {
        "tests/test_bad_kind.py": SurfaceClassification(
            classification="approved",
            category="module/workflow fake requiring review",
        ),
        "tests/test_blank_category.py": SurfaceClassification(
            classification="violation",
            category=" ",
        ),
    }

    assert validate_classifications(classifications) == [
        "tests/test_bad_kind.py: invalid classification 'approved'",
        "tests/test_blank_category.py: category must not be blank",
    ]


def test_knowledge_api_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_knowledge_api.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_knowledge_api_real_surface.py" in category


def test_actor_unit_guard_has_real_surface_companion() -> None:
    """Actor unit guard should be backed by real HTTP generator coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS["tests/test_actor.py"]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_actor_real_surface.py" in category


def test_audit_chain_unit_guard_has_real_surface_companion() -> None:
    """Audit-chain unit guard should use a real server and CLI companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_audit_chain.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_audit_chain_real_surface.py" in category


def test_audit_salt_unit_guard_has_real_surface_companion() -> None:
    """Audit-salt unit guard should use a real server-route companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_audit_salt.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_audit_salt_real_surface.py" in category


def test_evidence_packet_unit_guard_has_real_surface_companion() -> None:
    """Evidence packet unit guard should be backed by public CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_evidence_packet.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_evidence_packet_real_surface.py" in category


def test_backend_registry_unit_guard_has_real_surface_companion() -> None:
    """Backend registry guard should be backed by public registry coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_backends.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_backends_real_surface.py" in category


def test_containment_unit_guard_has_real_surface_companion() -> None:
    """Containment guard should be backed by public workflow coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_containment.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_containment_real_surface.py" in category


def test_zk_attestation_unit_guard_has_real_surface_companion() -> None:
    """ZK attestation guard should be backed by public passport coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_zk_attestation.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_zk_attestation_real_surface.py" in category


def test_dp_rag_unit_guard_has_real_surface_companion() -> None:
    """DP-RAG guard should be backed by public budget workflow coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_dp_rag.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_dp_rag_real_surface.py" in category


def test_dialogue_fpr_unit_guard_has_real_surface_companion() -> None:
    """Dialogue FPR guard should be backed by public scorer coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_dialogue_fpr.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_dialogue_fpr_real_surface.py" in category


def test_agent_provider_unit_guard_has_real_surface_companion() -> None:
    """Agent provider guard should be backed by real provider protocol coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_agent_providers.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_agent_providers_real_surface.py" in category


def test_agent_unit_guard_has_real_surface_companion() -> None:
    """Agent guard should be backed by real completion endpoint coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS["tests/test_agent.py"]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_agent_real_surface.py" in category


def test_middleware_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_middleware.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_middleware_real_surface.py" in category


def test_metrics_unit_guard_has_real_surface_companion() -> None:
    """Metrics guard should be backed by real server metrics coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_metrics.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_metrics_real_surface.py" in category


def test_otel_unit_guard_has_real_surface_companion() -> None:
    """OTel guard should be backed by real SDK exporter coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS["tests/test_otel.py"]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_otel_real_surface.py" in category


def test_output_integrity_unit_guard_has_real_surface_companion() -> None:
    """Output-integrity guard should be backed by public facade coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_output_integrity.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_output_integrity_real_surface.py" in category


def test_moderation_unit_guard_has_real_surface_companion() -> None:
    """Moderation detector guards should be backed by production wrapper coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_moderation.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_moderation_real_surface.py" in category


def test_device_selection_unit_guard_has_real_surface_companion() -> None:
    """Device selection guard should be backed by subprocess selector coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_device_selection.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_device_selection_real_surface.py" in category


def test_feedback_store_unit_guard_has_real_surface_companion() -> None:
    """Feedback-store guard should be backed by HTTP and SQLite route coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_feedback_store.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_feedback_store_real_surface.py" in category


def test_lazy_enterprise_import_unit_guard_has_real_surface_companion() -> None:
    """Lazy enterprise import guard should be backed by public import coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lazy_enterprise_import.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lazy_enterprise_import_real_surface.py" in category


def test_sdk_guard_unit_guard_has_real_surface_companion() -> None:
    """SDK guard unit tests should be backed by real OpenAI SDK coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_sdk_guard.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_sdk_guard_real_surface.py" in category


def test_contradiction_unit_guard_has_real_surface_companion() -> None:
    """Contradiction scorer unit tests should be backed by real scorer coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_contradiction.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_contradiction_real_surface.py" in category


def test_cost_integration_unit_guard_has_real_surface_companion() -> None:
    """Cost integration unit tests should be backed by real CLI/config coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_cost_integration.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_cost_integration_real_surface.py" in category


def test_cost_and_attribution_unit_guard_has_real_surface_companion() -> None:
    """Cost/attribution unit guard should use a real scorer companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_cost_and_attribution.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_cost_and_attribution_real_surface.py" in category


def test_review_queue_unit_guard_has_real_surface_companion() -> None:
    """ReviewQueue unit guard should use a real server companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_review_queue.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_review_queue_real_surface.py" in category


def test_server_auth_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_server_auth.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_server_auth_real_surface.py" in category


def test_server_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_server.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_server_real_surface.py" in category


def test_frontierfail_packet_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_frontierfail_packet.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_frontierfail_packet_real_surface.py" in category


def test_hf_space_demo_unit_guard_has_real_surface_companion() -> None:
    """HF Space package guard should be backed by real CLI validation."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_hf_space_demo.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_hf_space_demo_real_surface.py" in category


def test_hf_space_app_safety_unit_guard_has_real_surface_companion() -> None:
    """HF Space app-safety guard should be backed by real app callback coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_hf_space_app_safety.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_hf_space_app_safety_real_surface.py" in category


def test_hf_space_deployment_smoke_unit_guard_has_real_surface_companion() -> None:
    """HF Space deployment-smoke guard should be backed by real CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_hf_space_deployment_smoke.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_hf_space_deployment_smoke_real_surface.py" in category


def test_handoff_scorer_unit_guard_has_real_surface_companion() -> None:
    """Handoff scorer guard should be backed by real adapter wiring coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_handoff_scorer.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_handoff_scorer_real_surface.py" in category


def test_pint_replication_packet_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_pint_replication_packet.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_pint_replication_packet_real_surface.py" in category


def test_pint_seed_smoke_runner_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_pint_seed_smoke_runner.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_pint_seed_smoke_runner_real_surface.py" in category


def test_pint_official_export_runner_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_pint_official_export_runner.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_pint_official_export_runner_real_surface.py" in category


def test_lite_scorer_v2_plan_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_plan.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_plan_real_surface.py" in category


def test_lite_scorer_v2_run_plan_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_run_plan.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_run_plan_real_surface.py" in category


def test_lite_scorer_v2_eval_unit_guard_has_real_surface_companion() -> None:
    """Lite Scorer v2 evaluator should be backed by real CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_eval.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_eval_real_surface.py" in category


def test_lite_scorer_v2_eval_runner_unit_guard_has_real_surface_companion() -> None:
    """Lite Scorer v2 eval runner should be backed by real CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_eval_runner.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_eval_runner_real_surface.py" in category


def test_lite_scorer_v2_launcher_unit_guard_has_real_surface_companion() -> None:
    """Lite Scorer v2 launcher should be backed by real CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_launcher.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_launcher_real_surface.py" in category


def test_lite_scorer_v2_export_runner_unit_guard_has_real_surface_companion() -> None:
    """Lite Scorer v2 export runner should be backed by real CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_export_runner.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_export_runner_real_surface.py" in category


def test_lite_scorer_v2_training_status_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_training_status.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_training_status_real_surface.py" in category


def test_lite_scorer_v2_record_evidence_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_record_evidence.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_record_evidence_real_surface.py" in category


def test_lite_scorer_v2_heldout_builder_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_heldout_builder.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_heldout_builder_real_surface.py" in category


def test_ci_gate_unit_guard_has_real_surface_companion() -> None:
    """CI-gate helper coverage should be backed by public CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_ci_gate.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_ci_gate_real_surface.py" in category


def test_config_wizard_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_config_wizard.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_config_wizard_real_surface.py" in category


def test_config_guard_has_real_surface_companion() -> None:
    """Config guard should be backed by real env/file/runtime wiring coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_config.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_config_real_surface.py" in category


def test_finetune_benchmark_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_finetune_benchmark.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_finetune_benchmark_real_surface.py" in category


def test_multilingual_corpus_unit_guard_has_real_surface_companion() -> None:
    """Multilingual corpus guard should be backed by real CLI validation."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_multilingual_corpus.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_multilingual_corpus_real_surface.py" in category


def test_streaming_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_streaming.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_streaming_runtime_real_surface.py" in category


def test_task_scoring_paths_unit_guard_has_real_surface_companion() -> None:
    """Task scoring path guards should be backed by public scorer coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_task_scoring_paths.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_task_scoring_paths_real_surface.py" in category


def test_hyde_backend_unit_guard_has_real_surface_companion() -> None:
    """HyDE unit guards should be backed by real retrieval wiring coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_hyde_backend.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_hyde_backend_real_surface.py" in category


def test_providers_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_providers.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_providers_real_surface.py" in category


def test_semantic_kernel_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_integrations_semantic_kernel.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_integrations_semantic_kernel_real_surface.py" in category


def test_guardrails_ai_unit_guard_has_real_surface_companion() -> None:
    """Guardrails AI unit guard should be backed by protocol parse coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_guardrails_ai_integration.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_guardrails_ai_real_surface.py" in category


def test_public_endpoint_policy_unit_guard_has_real_surface_companion() -> None:
    """Public endpoint guard should be backed by real server auth coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_public_endpoint_exposure_policy.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_public_endpoint_exposure_real_surface.py" in category


def test_python_only_contributor_unit_guard_has_real_surface_companion() -> None:
    """Python-only contributor guard should be backed by real Make/CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_python_only_contributor_path.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_python_only_contributor_path_real_surface.py" in category


def test_prompt_guard_unit_guard_has_real_server_companion() -> None:
    """Prompt guard unit tests should be backed by real server request coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_prompt_guard.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_server_prompt_guard.py" in category


def test_proxy_facts_path_unit_guard_has_real_surface_companion() -> None:
    """Proxy facts-path guard should be backed by the public proxy app."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_proxy_facts_path_security.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_proxy_facts_path_real_surface.py" in category


def test_proxy_unit_guard_has_real_surface_companion() -> None:
    """Proxy unit guard should be backed by real ASGI route coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS["tests/test_proxy.py"]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_proxy_real_surface.py" in category


def test_voice_adapter_sdk_unit_guard_has_real_surface_companion() -> None:
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_voice_adapters_sdk.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_voice_adapters_real_surface.py" in category


def test_voice_adapter_unit_guard_has_real_surface_companion() -> None:
    """Voice adapter unit guard should be backed by real SDK construction."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_voice_adapters.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_voice_adapters_real_surface.py" in category


def test_capability_manifest_tool_guard_has_real_surface_companion() -> None:
    """Capability manifest guard should exercise the production generator."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_tools/test_capability_manifest.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tools/capability_manifest.py" in category


def test_studio_manifest_tool_guard_has_real_surface_companion() -> None:
    """STUDIO manifest guard should exercise the production CLI emitter."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_tools/test_emit_studio_manifest.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_tools/test_emit_studio_manifest_real_surface.py" in category


def test_report_templates_guard_has_real_surface_companion() -> None:
    """Report template guard should be backed by production renderer coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_report_templates.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "director_ai.compliance.report_templates" in category


def test_polar_deployment_smoke_guard_has_real_surface_companion() -> None:
    """Polar smoke guard should be backed by production packet validation."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_polar_deployment_smoke.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tools/validate_polar_deployment_smoke.py" in category


def test_prepare_threshold_data_guard_has_real_surface_companion() -> None:
    """Threshold feeder guard should be backed by real CLI conversion coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_prepare_threshold_data.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_prepare_threshold_data_real_surface.py" in category


def test_secrets_guard_has_real_surface_companion() -> None:
    """Secrets guard should be backed by real env backend hydration coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_secrets.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_secrets_real_surface.py" in category


def test_production_guard_has_real_surface_companion() -> None:
    """ProductionGuard unit guard should have a real public-facade companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_production_guard.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_production_guard_real_surface.py" in category


def test_doc_modules_guard_has_real_surface_companion() -> None:
    """Document module guard should be backed by real parser/chunker wiring."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_doc_modules.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_doc_modules_real_surface.py" in category


def test_doc_chunker_model_cache_unit_guard_has_real_surface_companion() -> None:
    """Doc chunker cache guard should be backed by public semantic splitting."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_doc_chunker_model_cache.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_doc_chunker_model_cache_real_surface.py" in category


def test_query_decomposition_unit_guard_has_real_surface_companion() -> None:
    """Query-decomposition guard should be backed by real store wiring coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_query_decomposition.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_query_decomposition_real_surface.py" in category


def test_contextual_compression_unit_guard_has_real_surface_companion() -> None:
    """Contextual-compression guard should be backed by real store wiring."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_contextual_compression.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_contextual_compression_real_surface.py" in category


def test_vector_store_http_embedding_unit_guard_has_real_surface_companion() -> None:
    """HTTP embedding guard should be backed by public adapter coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_vector_store_http_embedding.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_vector_store_http_embedding_real_surface.py" in category


def test_policy_main_returns_success_for_clean_tree(tmp_path: Path) -> None:
    _write_test(tmp_path, "tests/test_real_surface.py")

    assert policy.main(["--root", str(tmp_path)]) == 0


def test_policy_main_reports_forbidden_and_unclassified_surfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_test(tmp_path, "tests/test_final_push.py")
    path = tmp_path / "tests/test_adapter_contract.py"
    path.write_text(
        "from unittest.mock import MagicMock\n\ndef test_contract():\n    assert MagicMock\n",
        encoding="utf-8",
    )

    assert policy.main(["--root", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "Forbidden bucket-style test file names detected" in captured.err
    assert "Unclassified mock/sys.modules test surfaces detected" in captured.err
    assert "tests/test_final_push.py: token 'final'" in captured.err
    assert "tests/test_adapter_contract.py: unittest.mock" in captured.err


def test_policy_main_reports_invalid_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        policy,
        "_default_classifications",
        lambda: {
            "tests/test_bad.py": SurfaceClassification(
                classification="bad",
                category="module/workflow fake requiring review",
            )
        },
    )

    assert policy.main(["--root", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "Invalid test-surface classification manifest" in captured.err
    assert "tests/test_bad.py: invalid classification 'bad'" in captured.err
