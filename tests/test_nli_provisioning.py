# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI model provisioning contracts

"""Contract tests for the NLI model-provisioning module.

``director_ai.core.scoring._nli_provisioning`` owns revision resolution,
GCS artefact download, the cached model loader, cache eviction, and the
availability probe. These tests pin the compatibility re-exports on
``nli.py`` (callers and test patches keep resolving through the nli
namespace) and the pure helpers' contracts; the loader and GCS behaviour
matrix stays in ``tests/test_nli_backend_contracts.py``.
"""

from __future__ import annotations

import pytest

import director_ai.core.scoring._nli_provisioning as nli_provisioning
import director_ai.core.scoring.nli as nli_mod


class TestCompatibilitySurface:
    def test_nli_module_re_exports_the_provisioning_names(self):
        for name in (
            "MODEL_REGISTRY",
            "_DEFAULT_MODEL",
            "_DEFAULT_MODEL_REVISION",
            "_RECOMMENDED_MODEL",
            "_load_nli_model",
            "_resolve_revision",
            "clear_model_cache",
            "nli_available",
        ):
            assert getattr(nli_mod, name) is getattr(nli_provisioning, name)

    def test_registry_pins_the_default_model(self):
        assert nli_provisioning._DEFAULT_MODEL in nli_provisioning.MODEL_REGISTRY
        assert (
            nli_provisioning.MODEL_REGISTRY[nli_provisioning._DEFAULT_MODEL]
            == nli_provisioning._DEFAULT_MODEL_REVISION
        )


class TestArtefactHelpers:
    def test_split_gs_uri_requires_bucket_and_prefix(self):
        assert nli_provisioning._split_gs_uri("gs://bucket/some/prefix") == (
            "bucket",
            "some/prefix",
        )
        for bad in ("local/model", "gs://", "gs://bucket", "gs://bucket/"):
            with pytest.raises(ValueError):
                nli_provisioning._split_gs_uri(bad)

    def test_safe_cache_name_is_deterministic_and_filesystem_safe(self):
        first = nli_provisioning._safe_cache_name("gs://bucket/model v1")
        second = nli_provisioning._safe_cache_name("gs://bucket/model v1")
        other = nli_provisioning._safe_cache_name("gs://bucket/model v2")
        assert first == second
        assert first != other
        assert "/" not in first
        assert " " not in first

    def test_should_skip_artifact_ignores_training_state(self):
        assert nli_provisioning._should_skip_artifact("optimizer.pt") is True
        assert (
            nli_provisioning._should_skip_artifact("checkpoint-500/model.bin") is True
        )
        assert nli_provisioning._should_skip_artifact("model.safetensors") is False

    def test_resolve_model_source_passes_local_names_through(self):
        assert nli_provisioning._resolve_model_source("org/model") == "org/model"
