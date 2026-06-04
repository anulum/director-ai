# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — gRPC API documentation contract tests

from __future__ import annotations

from pathlib import Path


def test_grpc_api_docs_match_shipped_service_names() -> None:
    docs = Path("docs-site/api/grpc.md").read_text(encoding="utf-8")
    legacy_proto = Path("proto/director.proto").read_text(encoding="utf-8")
    scoring_proto = Path("schemas/proto/director/v1/director.proto").read_text(
        encoding="utf-8"
    )

    assert "proto/director.proto" in docs
    assert "schemas/proto/director/v1/director.proto" in docs
    assert "DirectorService" in docs
    assert "CoherenceScoring" in docs

    for rpc in ("Review", "Process", "ReviewBatch", "StreamTokens"):
        assert f"rpc {rpc}" in legacy_proto
        assert f"`{rpc}`" in docs

    for rpc in ("ScoreClaim", "ScoreStream"):
        assert f"rpc {rpc}" in scoring_proto
        assert f"`{rpc}`" in docs

    assert "StreamReview" not in docs
    assert "HealthCheck" not in docs
