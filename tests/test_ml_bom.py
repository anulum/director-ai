# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ML bill-of-materials tests

"""Multi-angle tests for the supply-chain ML-BOM (OWASP ASVS).

Covers digest computation and validation, component matching (poisoning
detection), BOM assembly with duplicate rejection, the tamper-evident BOM digest,
and the verify() classification (intact / tampered / unverified / unknown), plus
ProductionGuard wiring.
"""

from __future__ import annotations

import pytest

from director_ai.core.ml_bom import (
    ComponentType,
    MachineLearningBOM,
    MLBOMComponent,
    VerificationReport,
    compute_sha256,
)

_DIGEST = compute_sha256(b"x")


class TestComputeSha256:
    def test_bytes_and_bytearray(self):
        assert compute_sha256(b"abc") == compute_sha256(bytearray(b"abc"))
        assert len(compute_sha256(b"abc")) == 64

    def test_non_bytes_rejected(self):
        with pytest.raises(TypeError, match="must be bytes"):
            compute_sha256("a string")


class TestComponent:
    def test_valid_component(self):
        c = MLBOMComponent("m", "1.0", ComponentType.MODEL, _DIGEST, supplier="anulum")
        assert c.component_type is ComponentType.MODEL
        assert c.to_dict()["supplier"] == "anulum"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": "  "}, "name is required"),
            ({"version": ""}, "version is required"),
            ({"sha256": "nothex"}, "64 lower-case hex"),
            ({"sha256": "AB" * 32}, "64 lower-case hex"),
        ],
    )
    def test_validation(self, kwargs, match):
        base = {
            "name": "m",
            "version": "1",
            "component_type": ComponentType.MODEL,
            "sha256": _DIGEST,
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            MLBOMComponent(**base)

    def test_matches_true_and_false(self):
        c = MLBOMComponent("m", "1", ComponentType.MODEL, compute_sha256(b"weights"))
        assert c.matches(b"weights") is True
        assert c.matches(b"poisoned") is False

    def test_to_dict_keys(self):
        c = MLBOMComponent("m", "1", ComponentType.DATASET, _DIGEST)
        assert set(c.to_dict()) == {
            "name",
            "version",
            "component_type",
            "sha256",
            "supplier",
            "source",
            "license",
        }


class TestBom:
    def test_add_and_components_sorted(self):
        bom = MachineLearningBOM()
        bom.add(MLBOMComponent("zeta", "1", ComponentType.MODEL, _DIGEST))
        bom.add(MLBOMComponent("alpha", "1", ComponentType.DATASET, _DIGEST))
        assert [c.name for c in bom.components] == ["alpha", "zeta"]

    def test_duplicate_rejected(self):
        bom = MachineLearningBOM()
        bom.add(MLBOMComponent("m", "1", ComponentType.MODEL, _DIGEST))
        with pytest.raises(ValueError, match="already recorded"):
            bom.add(MLBOMComponent("m", "2", ComponentType.MODEL, _DIGEST))

    def test_add_artifact_pins_digest(self):
        bom = MachineLearningBOM()
        c = bom.add_artifact(
            "model", "1.0", ComponentType.MODEL, b"weights", supplier="anulum"
        )
        assert c.sha256 == compute_sha256(b"weights")
        assert c.supplier == "anulum"

    def test_bom_digest_stable_and_sensitive(self):
        bom = MachineLearningBOM()
        bom.add_artifact("m", "1", ComponentType.MODEL, b"w")
        d1 = bom.bom_digest
        assert d1 == bom.bom_digest  # stable
        bom.add_artifact("d", "1", ComponentType.DATASET, b"data")
        assert bom.bom_digest != d1  # changes when inventory changes

    def test_to_dict_carries_digest_and_components(self):
        bom = MachineLearningBOM()
        bom.add_artifact("m", "1", ComponentType.MODEL, b"w")
        d = bom.to_dict()
        assert len(d["bom_digest"]) == 64
        assert d["components"][0]["name"] == "m"


class TestVerify:
    def _bom(self) -> MachineLearningBOM:
        bom = MachineLearningBOM()
        bom.add_artifact("model", "1", ComponentType.MODEL, b"model-bytes")
        bom.add_artifact("data", "1", ComponentType.DATASET, b"data-bytes")
        bom.add_artifact("dep", "1", ComponentType.DEPENDENCY, b"dep-bytes")
        return bom

    def test_intact_component(self):
        report = self._bom().verify({"model": b"model-bytes"})
        assert report.intact == ("model",)
        assert report.ok is True

    def test_tampered_component_detected(self):
        report = self._bom().verify({"model": b"POISONED"})
        assert report.tampered == ("model",)
        assert report.ok is False

    def test_unverified_when_not_supplied(self):
        report = self._bom().verify({"model": b"model-bytes"})
        assert set(report.unverified) == {"data", "dep"}

    def test_unknown_name_is_tampered(self):
        report = self._bom().verify({"rogue": b"x"})
        assert "rogue" in report.tampered
        assert report.ok is False

    def test_mixed_classification(self):
        report = self._bom().verify(
            {"model": b"model-bytes", "data": b"changed", "rogue": b"y"}
        )
        assert report.intact == ("model",)
        assert report.tampered == ("data", "rogue")
        assert report.unverified == ("dep",)


class TestVerificationReport:
    def test_ok_true_when_no_tamper(self):
        assert VerificationReport(intact=("a",)).ok is True

    def test_ok_false_when_tampered(self):
        assert VerificationReport(tampered=("a",)).ok is False

    def test_to_dict(self):
        d = VerificationReport(
            intact=("a",), tampered=("b",), unverified=("c",)
        ).to_dict()
        assert d == {
            "ok": False,
            "intact": ["a"],
            "tampered": ["b"],
            "unverified": ["c"],
        }

    def test_defaults_empty(self):
        report = VerificationReport()
        assert report.intact == () and report.tampered == () and report.unverified == ()


class TestGuardWiring:
    def test_production_guard_exposes_ml_bom(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        bom = guard.ml_bom
        assert isinstance(bom, MachineLearningBOM)
        assert guard.ml_bom is bom  # cached
        bom.add_artifact("m", "1", ComponentType.MODEL, b"w")
        assert bom.verify({"m": b"w"}).ok is True
