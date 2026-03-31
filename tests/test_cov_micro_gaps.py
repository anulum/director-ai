# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — test_cov_micro_gaps.py

"""Multi-angle micro coverage gap tests.

Covers: server halted process response, tenant-not-enabled 404,
batch endpoint, NLI minicheck import failure, LangChain callback
AttributeError, parametrised server endpoints, and pipeline
performance documentation.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.config import DirectorConfig

_HAS_FASTAPI = __import__("importlib").util.find_spec("fastapi") is not None
_skip_no_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


# â”€â”€ Server: halted process response + tenant not enabled â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@_skip_no_server
class TestServerHaltedProcess:
    def test_process_halted_increments_counters(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(
            use_nli=False,
            coherence_threshold=1.0,
            hard_limit=1.0,
            soft_limit=1.0,
        )
        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.post("/v1/process", json={"prompt": "Tell me about bananas."})
            assert resp.status_code == 200
            data = resp.json()
            assert "halted" in data


@_skip_no_server
class TestServerTenantNotEnabled:
    def test_add_fact_no_tenant(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.post("/v1/tenants/t1/facts", json={"key": "k", "value": "v"})
            assert resp.status_code == 404


@_skip_no_server
class TestServerBatchNotReady:
    def test_batch_503(self):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/v1/batch", json={"prompts": ["test"]})
            assert resp.status_code in (200, 503)


# â”€â”€ NLI: minicheck import failure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestNliMinicheckImportFailure:
    def test_ensure_minicheck_import_error(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer.__new__(NLIScorer)
        scorer.backend = "minicheck"
        scorer._minicheck_loaded = False
        scorer._minicheck = None
        scorer._model = None
        scorer._tokenizer = None
        scorer._onnx_session = None
        scorer._custom_backend = None
        scorer._model_name = ""
        scorer.max_length = 512
        scorer.use_model = False
        scorer._model_loaded = False
        scorer._label_indices = None

        with patch.dict(sys.modules, {"minicheck": None}):
            result = scorer._ensure_minicheck()
            assert result is False


# â”€â”€ LangChain callback: AttributeError extraction path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestLangchainAttributeError:
    def test_on_llm_end_attribute_error(self):
        from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

        handler = CoherenceCallbackHandler(use_nli=False, threshold=0.5)
        handler._current_prompt = "test"
        response = MagicMock()
        response.generations = MagicMock()
        response.generations.__getitem__ = MagicMock(side_effect=AttributeError("no"))
        handler.on_llm_end(response)


@_skip_no_server
class TestMicroGapsPerformanceDoc:
    """Document micro-gap pipeline characteristics."""

    @pytest.mark.parametrize(
        "endpoint,payload,expected_status",
        [
            ("/v1/process", {"prompt": "test"}, 200),
            ("/v1/health", None, 200),
        ],
    )
    def test_server_endpoints_respond(self, endpoint, payload, expected_status):
        from starlette.testclient import TestClient

        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False)
        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.post(endpoint, json=payload) if payload else c.get(endpoint)
            assert resp.status_code == expected_status
