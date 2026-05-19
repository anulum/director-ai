# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vertex container CUDA policy tests

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vertex_benchmark_container_overrides_to_vertex_compatible_cuda() -> None:
    text = (ROOT / "training" / "Dockerfile.benchmarks").read_text()

    assert "torch==2.5.1+cu121" in text
    assert "https://download.pytorch.org/whl/cu121" in text
    assert "DIRECTOR_REQUIRE_CUDA=1" in text


def test_vertex_lite_scorer_container_overrides_to_vertex_compatible_cuda() -> None:
    text = (ROOT / "training" / "Dockerfile.lite_scorer_v2").read_text()

    assert "torch==2.5.1+cu121" in text
    assert "https://download.pytorch.org/whl/cu121" in text
    assert "DIRECTOR_REQUIRE_CUDA=1" in text


def test_vertex_benchmark_entrypoint_fails_fast_without_cuda() -> None:
    text = (ROOT / "benchmarks" / "run_in_container.sh").read_text()

    assert "DIRECTOR_REQUIRE_CUDA" in text
    assert "torch.cuda.is_available()" in text
    assert 'torch.ones(1, device="cuda")' in text
