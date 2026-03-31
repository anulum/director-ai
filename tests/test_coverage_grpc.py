# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for grpc_server.py."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


class TestCreateGrpcServer:
    def test_grpc_import_error(self):
        with (
            patch.dict(sys.modules, {"grpc": None}),
            pytest.raises(ImportError, match="grpcio"),
        ):
            from director_ai.core.config import DirectorConfig
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(DirectorConfig())
