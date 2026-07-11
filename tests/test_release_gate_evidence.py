# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Release-gate evidence module contracts

"""Contract tests for the release-gate evidence records module.

``director_ai.core.customer_model_factory._gate_evidence`` owns the ten
per-domain evidence dataclasses of the Customer Model Factory release
gate; ``release_gate`` re-exports them unchanged. These tests pin where
the classes live, the re-export identity, and the ``to_dict``/
``from_dict`` round-trip shape; the full blocker behaviour matrix stays
in ``tests/test_customer_model_factory_release_gate.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from director_ai.core.customer_model_factory import _gate_evidence, release_gate

_EVIDENCE_CLASS_NAMES = (
    "AutoRedteamDefenceEvidence",
    "ConformalRoutingEvidence",
    "DeploymentHardeningEvidence",
    "EdgeMobileEvidence",
    "FederatedPrivacyEvidence",
    "FormalSymbolicEvidence",
    "MultimodalTemporalEvidence",
    "ObservabilityOperationsEvidence",
    "ProvenanceLineageEvidence",
    "TrajectoryRollbackEvidence",
)


class TestModulePlacement:
    def test_module_exports_exactly_the_ten_evidence_classes(self):
        assert _gate_evidence.__all__ == list(_EVIDENCE_CLASS_NAMES)

    def test_classes_are_defined_in_the_evidence_module(self):
        for name in _EVIDENCE_CLASS_NAMES:
            cls = getattr(_gate_evidence, name)
            assert cls.__module__ == _gate_evidence.__name__

    def test_release_gate_re_exports_the_same_objects(self):
        for name in _EVIDENCE_CLASS_NAMES:
            assert getattr(release_gate, name) is getattr(_gate_evidence, name)

    def test_package_surface_still_serves_the_evidence_classes(self):
        from director_ai.core import customer_model_factory as package

        for name in _EVIDENCE_CLASS_NAMES:
            assert getattr(package, name) is getattr(_gate_evidence, name)


class TestRecordContracts:
    def test_records_are_frozen_dataclasses(self):
        for name in _EVIDENCE_CLASS_NAMES:
            cls = getattr(_gate_evidence, name)
            assert dataclasses.is_dataclass(cls)
            assert cls.__dataclass_params__.frozen

    def test_round_trip_preserves_every_field(self):
        evidence = _gate_evidence.DeploymentHardeningEvidence(
            ready=True,
            environment="staging",
            observation_window="72h",
            telemetry_uri="s3://telemetry",
            sustained_load_packet_uri="s3://load",
            operator_signoff_uri="s3://signoff",
            async_ordering_passed=True,
            tenant_poisoning_passed=True,
            evidence_hash="a" * 64,
        )
        payload = evidence.to_dict()
        rebuilt = _gate_evidence.DeploymentHardeningEvidence.from_dict(payload)
        assert rebuilt == evidence
        assert set(payload) == {field.name for field in dataclasses.fields(evidence)}

    def test_from_dict_requires_every_field(self):
        with pytest.raises(KeyError):
            _gate_evidence.ObservabilityOperationsEvidence.from_dict({"ready": True})
