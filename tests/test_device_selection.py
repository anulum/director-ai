# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — torch device selector tests

"""Multi-angle coverage of
:func:`director_ai.core._device.select_torch_device`. Monkeypatches
the torch probe so the test suite does not depend on a specific
PyTorch build and runs identically on CPU-only boxes."""

from __future__ import annotations

import logging

import pytest

from director_ai.core import _device
from director_ai.core._device import reset_warn_cache, select_torch_device


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_warn_cache()
    monkeypatch.delenv("DIRECTOR_FORCE_CPU", raising=False)
    yield


def _patch_cuda(
    monkeypatch,
    *,
    count: int,
    capabilities: list[tuple[int, int]] | None,
    arches: list[str] | None = None,
):
    monkeypatch.setattr(_device, "_visible_device_count", lambda: count)

    if capabilities is None:
        def cap(_idx: int):
            return None
    else:
        def cap(idx: int):
            return capabilities[idx] if 0 <= idx < len(capabilities) else None

    monkeypatch.setattr(_device, "_capability", cap)

    if arches is None:
        arches = ["sm_70", "sm_80", "sm_86", "sm_90"]

    def fake_min() -> tuple[int, int]:
        ints = [int(a[3:-1]) * 10 + int(a[-1]) for a in arches if a.startswith("sm_")]
        if not ints:
            return (7, 0)
        lowest = min(ints)
        return (lowest // 10, lowest % 10)

    monkeypatch.setattr(_device, "_minimum_capability", fake_min)


class TestSelectDevice:
    def test_no_cuda_returns_cpu(self, monkeypatch):
        _patch_cuda(monkeypatch, count=0, capabilities=[])
        assert select_torch_device() == "cpu"

    def test_capable_gpu_selected(self, monkeypatch):
        _patch_cuda(monkeypatch, count=1, capabilities=[(8, 6)])
        assert select_torch_device() == "cuda:0"

    def test_sm_61_falls_back_to_cpu(self, monkeypatch):
        _patch_cuda(monkeypatch, count=1, capabilities=[(6, 1)])
        assert select_torch_device() == "cpu"

    def test_picks_first_capable_when_mixed(self, monkeypatch):
        _patch_cuda(monkeypatch, count=3, capabilities=[(6, 1), (8, 6), (9, 0)])
        assert select_torch_device() == "cuda:1"

    def test_force_cpu_env_overrides_gpu(self, monkeypatch):
        _patch_cuda(monkeypatch, count=1, capabilities=[(9, 0)])
        monkeypatch.setenv("DIRECTOR_FORCE_CPU", "1")
        assert select_torch_device() == "cpu"

    def test_preferred_cpu_honoured(self, monkeypatch):
        _patch_cuda(monkeypatch, count=1, capabilities=[(9, 0)])
        assert select_torch_device("cpu") == "cpu"

    def test_preferred_cuda_passes_when_capable(self, monkeypatch):
        _patch_cuda(monkeypatch, count=2, capabilities=[(8, 0), (9, 0)])
        monkeypatch.setattr(_device, "_cuda_usable_for", lambda dev: True)
        assert select_torch_device("cuda:1") == "cuda:1"

    def test_preferred_cuda_falls_through_when_incompatible(self, monkeypatch):
        # Preferred points at an unsupported device; the selector
        # re-runs the capability walk rather than crashing.
        _patch_cuda(monkeypatch, count=2, capabilities=[(6, 1), (8, 6)])
        monkeypatch.setattr(_device, "_cuda_usable_for", lambda dev: False)
        assert select_torch_device("cuda:0") == "cuda:1"

    def test_warning_emitted_once(self, monkeypatch, caplog):
        _patch_cuda(monkeypatch, count=1, capabilities=[(6, 1)])
        caplog.set_level(logging.WARNING, logger="DirectorAI.Device")
        select_torch_device()
        select_torch_device()
        warnings = [r for r in caplog.records if "no CUDA device" in r.message]
        assert len(warnings) == 1

    def test_minimum_capability_falls_back_to_7_0(self, monkeypatch):
        _patch_cuda(monkeypatch, count=0, capabilities=None, arches=[])
        assert _device._minimum_capability() == (7, 0)

    def test_force_cpu_various_values(self, monkeypatch):
        _patch_cuda(monkeypatch, count=1, capabilities=[(9, 0)])
        for raw in ("1", "true", "True", "yes", "YES"):
            monkeypatch.setenv("DIRECTOR_FORCE_CPU", raw)
            assert select_torch_device() == "cpu"
