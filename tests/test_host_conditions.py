# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for benchmark host-condition metadata (WCE-2).

Covers: snapshot shape, load-per-CPU arithmetic, pinning detection,
all three isolation verdicts including the quiet-threshold boundary,
governor read fallbacks, and the wiring into both benchmark
``_environment()`` recorders.
"""

from __future__ import annotations

import benchmarks.host_conditions as hc


def _snapshot(monkeypatch, *, load=(1.0, 1.0, 1.0), cpus=8, affinity=None):
    monkeypatch.setattr(hc.os, "getloadavg", lambda: load)
    monkeypatch.setattr(hc.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(
        hc.os,
        "sched_getaffinity",
        lambda pid: set(affinity if affinity is not None else range(cpus)),
    )
    return hc.host_conditions()


class TestHostConditionsSnapshot:
    def test_records_load_affinity_and_cpu_count(self, monkeypatch):
        conditions = _snapshot(
            monkeypatch,
            load=(3.14159, 2.5, 1.25),
            cpus=12,
            affinity=[4, 5],
        )

        assert conditions["load_avg"] == [3.14, 2.5, 1.25]
        assert conditions["load_per_cpu_1m"] == round(3.14159 / 12, 3)
        assert conditions["cpu_count"] == 12
        assert conditions["cpu_affinity"] == [4, 5]
        assert conditions["pinned"] is True

    def test_full_affinity_is_not_pinned(self, monkeypatch):
        conditions = _snapshot(monkeypatch, cpus=4, affinity=[0, 1, 2, 3])

        assert conditions["pinned"] is False

    def test_timestamp_is_utc_isoformat(self, monkeypatch):
        conditions = _snapshot(monkeypatch)

        assert conditions["captured_utc"].endswith("+00:00")

    def test_cpu_count_none_falls_back_to_one(self, monkeypatch):
        monkeypatch.setattr(hc.os, "getloadavg", lambda: (0.5, 0.5, 0.5))
        monkeypatch.setattr(hc.os, "cpu_count", lambda: None)
        monkeypatch.setattr(hc.os, "sched_getaffinity", lambda pid: {0})

        conditions = hc.host_conditions()

        assert conditions["cpu_count"] == 1
        assert conditions["load_per_cpu_1m"] == 0.5


class TestIsolationVerdict:
    def test_unpinned_is_shared_even_when_quiet(self, monkeypatch):
        conditions = _snapshot(monkeypatch, load=(0.0, 0.0, 0.0), cpus=8)

        assert conditions["isolation_verdict"] == "shared"

    def test_pinned_quiet_host(self, monkeypatch):
        conditions = _snapshot(
            monkeypatch,
            load=(1.0, 1.0, 1.0),
            cpus=8,
            affinity=[6, 7],
        )

        assert conditions["load_per_cpu_1m"] == 0.125
        assert conditions["isolation_verdict"] == "isolated-quiet"

    def test_pinned_loaded_host(self, monkeypatch):
        conditions = _snapshot(
            monkeypatch,
            load=(27.0, 25.0, 20.0),
            cpus=12,
            affinity=[10, 11],
        )

        assert conditions["isolation_verdict"] == "pinned-loaded-host"

    def test_quiet_threshold_boundary_is_inclusive(self, monkeypatch):
        cpus = 8
        boundary_load = hc.QUIET_LOAD_PER_CPU * cpus

        conditions = _snapshot(
            monkeypatch,
            load=(boundary_load, 0.0, 0.0),
            cpus=cpus,
            affinity=[0],
        )

        assert conditions["isolation_verdict"] == "isolated-quiet"


class TestGovernorRead:
    def test_reads_governor_when_present(self, monkeypatch, tmp_path):
        governor = tmp_path / "scaling_governor"
        governor.write_text("performance\n", encoding="ascii")
        monkeypatch.setattr(hc, "_GOVERNOR_PATH", governor)

        assert hc._read_governor() == "performance"

    def test_missing_governor_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hc, "_GOVERNOR_PATH", tmp_path / "absent")

        assert hc._read_governor() is None


class TestEnvironmentWiring:
    """Wiring checks need the optional extras the recorders import —
    the core CI dependency set has no faiss/torch, so these are
    extras-gated (they run locally and in the extras matrix jobs)."""

    def test_grounded_ann_bench_environment_embeds_conditions(self):
        import pytest

        pytest.importorskip("faiss")
        from benchmarks.grounded_ann_bench import _environment

        env = _environment()

        conditions = env["host_conditions"]
        assert conditions["isolation_verdict"] in {
            "shared",
            "isolated-quiet",
            "pinned-loaded-host",
        }
        assert len(conditions["load_avg"]) == 3

    def test_retrieval_refresh_environment_embeds_conditions(self):
        import pytest

        pytest.importorskip("faiss")
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")
        from benchmarks.retrieval_model_refresh_ab import _environment

        env = _environment()

        assert "host_conditions" in env
        assert env["host_conditions"]["cpu_count"] >= 1
