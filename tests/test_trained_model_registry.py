# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Trained Model Registry Tests

"""Multi-angle tests for the trained-model registry and its promotion gate."""

from __future__ import annotations

import json

import pytest

from director_ai.core.training.dataset_fingerprint import fingerprint_dataset
from director_ai.core.training.trained_model_registry import (
    STAGE_CANDIDATE,
    STAGE_PRODUCTION,
    STAGE_RETIRED,
    TrainedModelRecord,
    TrainedModelRegistry,
)

_EVIDENCE = {
    "metric": "balanced_accuracy",
    "candidate": 0.79,
    "baseline": 0.758,
    "source": "benchmarks/results/model_refresh.json",
}


def _fingerprint(tmp_path):
    dataset = tmp_path / "train.jsonl"
    if not dataset.exists():
        dataset.write_text('{"label": 1}\n', encoding="utf-8")
    return fingerprint_dataset(str(dataset))


def _register(registry, tmp_path, name="domain-nli", **overrides):
    values = {
        "name": name,
        "artifact_uri": "file:///artifacts/domain-nli",
        "base_model_id": "yaxili96/FactCG-DeBERTa-v3-Large",
        "dataset_fingerprint": _fingerprint(tmp_path),
        "metrics": {"balanced_accuracy": 0.79},
        "run_id": "local-abc123",
        "base_model_revision": "0430e3509dbd28d2dff7a117c0eae25359ff3e80",
        "config_hash": "cafe0123cafe0123",
    }
    values.update(overrides)
    return registry.register(**values)


class TestRegister:
    def test_first_registration_is_candidate_v1_with_lineage(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        fingerprint = _fingerprint(tmp_path)

        record = _register(registry, tmp_path)

        assert record.version == 1
        assert record.stage == STAGE_CANDIDATE
        assert record.dataset_digest == fingerprint.digest
        assert record.dataset_hash_source == "content"
        assert record.run_id == "local-abc123"
        assert record.config_hash == "cafe0123cafe0123"
        assert record.base_model_revision == (
            "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
        )
        assert record.metrics == {"balanced_accuracy": 0.79}
        assert record.registered_at > 0.0
        assert record.promoted_at == 0.0
        assert record.benchmark_evidence is None

    def test_versions_increment_per_name(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)
        second = _register(registry, tmp_path)
        other = _register(registry, tmp_path, name="other-model")

        assert second.version == 2
        assert other.version == 1

    def test_optional_lineage_fields_default_to_empty(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        record = registry.register(
            name="bare-model",
            artifact_uri="file:///artifacts/bare",
            base_model_id="distilroberta-base",
            dataset_fingerprint=_fingerprint(tmp_path),
        )

        assert record.run_id == ""
        assert record.base_model_revision == ""
        assert record.config_hash == ""
        assert record.metrics == {}

    @pytest.mark.parametrize(
        "name", ["", "UPPER", "spaces here", "-leading", "x" * 129]
    )
    def test_invalid_names_are_rejected(self, tmp_path, name):
        registry = TrainedModelRegistry(tmp_path / "registry")
        with pytest.raises(ValueError, match="model name must be a lowercase slug"):
            _register(registry, tmp_path, name=name)

    def test_missing_artifact_uri_is_rejected(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        with pytest.raises(ValueError, match="artifact_uri is required"):
            _register(registry, tmp_path, artifact_uri="")

    def test_missing_base_model_is_rejected(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        with pytest.raises(ValueError, match="base_model_id is required"):
            _register(registry, tmp_path, base_model_id="")


class TestQueries:
    def test_get_unknown_model_raises_key_error(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        with pytest.raises(KeyError, match="unknown trained model"):
            registry.get("absent", 1)

    def test_list_models_and_versions(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path, name="bravo")
        _register(registry, tmp_path, name="alpha")
        _register(registry, tmp_path, name="alpha")

        assert registry.list_models() == ["alpha", "bravo"]
        assert [record.version for record in registry.list_versions("alpha")] == [
            1,
            2,
        ]
        assert registry.list_versions("absent") == []

    def test_production_is_none_before_any_promotion(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)

        assert registry.production("domain-nli") is None

    def test_root_property_reports_storage_directory(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "nested" / "registry")

        assert registry.root == tmp_path / "nested" / "registry"
        assert registry.root.is_dir()


class TestPromotionGate:
    def test_promotion_requires_complete_evidence(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)

        with pytest.raises(ValueError, match="missing required keys"):
            registry.promote("domain-nli", 1, benchmark_evidence={"metric": "acc"})

    @pytest.mark.parametrize("metric", ["", 7])
    def test_promotion_requires_named_metric(self, tmp_path, metric):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)
        evidence = dict(_EVIDENCE, metric=metric)

        with pytest.raises(ValueError, match="metric must be a non-empty string"):
            registry.promote("domain-nli", 1, benchmark_evidence=evidence)

    def test_promotion_requires_numeric_scores(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)
        evidence = dict(_EVIDENCE, candidate="fast", baseline=None)

        with pytest.raises(ValueError, match="must be numeric"):
            registry.promote("domain-nli", 1, benchmark_evidence=evidence)

    def test_regressing_candidate_is_refused(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)
        evidence = dict(_EVIDENCE, candidate=0.70)

        with pytest.raises(ValueError, match="anti-regression gate"):
            registry.promote("domain-nli", 1, benchmark_evidence=evidence)
        assert registry.production("domain-nli") is None

    def test_equal_candidate_passes_the_gate(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)
        evidence = dict(_EVIDENCE, candidate=0.758)

        record = registry.promote("domain-nli", 1, benchmark_evidence=evidence)

        assert record.stage == STAGE_PRODUCTION

    def test_promotion_stamps_evidence_and_time(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)

        record = registry.promote("domain-nli", 1, benchmark_evidence=_EVIDENCE)

        assert record.stage == STAGE_PRODUCTION
        assert record.promoted_at > 0.0
        assert record.benchmark_evidence == _EVIDENCE
        assert registry.production("domain-nli") == record

    def test_promotion_retires_previous_production(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)
        _register(registry, tmp_path)
        registry.promote("domain-nli", 1, benchmark_evidence=_EVIDENCE)

        promoted = registry.promote("domain-nli", 2, benchmark_evidence=_EVIDENCE)

        assert promoted.stage == STAGE_PRODUCTION
        assert registry.get("domain-nli", 1).stage == STAGE_RETIRED
        production = registry.production("domain-nli")
        assert production is not None and production.version == 2

    def test_re_promoting_production_refreshes_evidence(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)
        registry.promote("domain-nli", 1, benchmark_evidence=_EVIDENCE)
        refreshed = dict(_EVIDENCE, candidate=0.80)

        record = registry.promote("domain-nli", 1, benchmark_evidence=refreshed)

        assert record.stage == STAGE_PRODUCTION
        assert record.benchmark_evidence == refreshed

    def test_retired_version_cannot_be_promoted(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)
        registry.retire("domain-nli", 1)

        with pytest.raises(ValueError, match="cannot promote retired version"):
            registry.promote("domain-nli", 1, benchmark_evidence=_EVIDENCE)


class TestRetireAndSerialisation:
    def test_retire_moves_version_to_retired(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)

        record = registry.retire("domain-nli", 1)

        assert record.stage == STAGE_RETIRED
        assert registry.get("domain-nli", 1).stage == STAGE_RETIRED

    def test_round_trip_preserves_every_field(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        _register(registry, tmp_path)
        promoted = registry.promote("domain-nli", 1, benchmark_evidence=_EVIDENCE)

        rebuilt = TrainedModelRecord.from_dict(
            json.loads(json.dumps(promoted.to_dict())),
        )

        assert rebuilt == promoted

    def test_from_dict_defaults_optional_fields(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        record = _register(registry, tmp_path)
        payload = record.to_dict()
        for key in ("run_id", "base_model_revision", "config_hash", "promoted_at"):
            payload.pop(key)

        rebuilt = TrainedModelRecord.from_dict(payload)

        assert rebuilt.run_id == ""
        assert rebuilt.base_model_revision == ""
        assert rebuilt.config_hash == ""
        assert rebuilt.promoted_at == 0.0
        assert rebuilt.benchmark_evidence is None

    def test_record_rejects_unknown_stage(self, tmp_path):
        registry = TrainedModelRegistry(tmp_path / "registry")
        record = _register(registry, tmp_path)
        payload = dict(record.to_dict(), stage="bogus")

        with pytest.raises(ValueError, match="stage must be one of"):
            TrainedModelRecord.from_dict(payload)

    def test_registry_surface_exports_from_training_package(self):
        import director_ai.core.training as training_api

        assert training_api.TrainedModelRegistry is TrainedModelRegistry
        assert training_api.STAGE_CANDIDATE == STAGE_CANDIDATE
