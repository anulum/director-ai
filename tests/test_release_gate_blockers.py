# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Release-gate blocker policy module contracts

"""Contract tests for the release-gate blocker policy module.

``director_ai.core.customer_model_factory._gate_blockers`` owns the
twelve blocker collectors of the Customer Model Factory release gate;
``release_gate`` imports them for its ``_collect_blockers`` assembly.
These tests pin where the collectors live, the facade wiring, and the
blocker record shape; the full per-domain blocker behaviour matrix
stays in ``tests/test_customer_model_factory_release_gate.py``.
"""

from __future__ import annotations

from director_ai.core.customer_model_factory import _gate_blockers, release_gate

_COLLECTOR_NAMES = (
    "_extend_readiness_blockers",
    "_extend_observability_operations_blockers",
    "_extend_provenance_lineage_blockers",
    "_extend_conformal_routing_blockers",
    "_extend_trajectory_rollback_blockers",
    "_extend_multimodal_temporal_blockers",
    "_extend_federated_privacy_blockers",
    "_extend_edge_mobile_blockers",
    "_extend_auto_redteam_defence_blockers",
    "_extend_formal_symbolic_blockers",
    "_extend_deployment_hardening_blockers",
    "_extend_boundary_blockers",
)


class TestModulePlacement:
    def test_collectors_are_defined_in_the_blockers_module(self):
        for name in _COLLECTOR_NAMES:
            func = getattr(_gate_blockers, name)
            assert func.__module__ == _gate_blockers.__name__

    def test_module_exports_collectors_and_helpers(self):
        assert set(_gate_blockers.__all__) == {
            *_COLLECTOR_NAMES,
            "_blocker",
            "_is_sha256",
        }

    def test_release_gate_serves_the_same_collector_objects(self):
        for name in _COLLECTOR_NAMES:
            assert getattr(release_gate, name) is getattr(_gate_blockers, name)


class TestBlockerContracts:
    def test_blocker_record_shape_and_blank_extra_dropped(self):
        record = _gate_blockers._blocker("code_x", "message x", debt_ids="")
        assert record == {
            "code": "code_x",
            "severity": "error",
            "message": "message x",
        }
        with_extra = _gate_blockers._blocker("code_x", "message x", debt_ids="a,b")
        assert with_extra["debt_ids"] == "a,b"

    def test_sha256_guard_accepts_only_lowercase_hex_digests(self):
        assert _gate_blockers._is_sha256("a" * 64)
        assert not _gate_blockers._is_sha256("A" * 64)
        assert not _gate_blockers._is_sha256("a" * 63)

    def test_environment_policy_flags_non_staging_production(self):
        evidence = release_gate.DeploymentHardeningEvidence(
            ready=True,
            environment="laptop",
            observation_window="72h",
            telemetry_uri="s3://telemetry",
            sustained_load_packet_uri="s3://load",
            operator_signoff_uri="s3://signoff",
            async_ordering_passed=True,
            tenant_poisoning_passed=True,
            evidence_hash="a" * 64,
        )
        blockers: list[dict[str, str]] = []
        _gate_blockers._extend_deployment_hardening_blockers(evidence, blockers)
        assert [b["code"] for b in blockers] == [
            "deployment_hardening_environment_invalid"
        ]
