# SPDX-License-Identifier: Apache-2.0
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
import sys
import types
import warnings

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

    def test_minimum_capability_parses_installed_torch_arches(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            get_arch_list=lambda: ["compute_90", "sm_86", "sm_70", "sm_bad"]
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert _device._minimum_capability() == (7, 0)

    def test_minimum_capability_falls_back_when_torch_arch_list_missing(
        self, monkeypatch
    ):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert _device._minimum_capability() == (7, 0)

    def test_minimum_capability_keeps_lowest_supported_arch(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            get_arch_list=lambda: ["sm_90", "sm_80", "sm_86"]
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert _device._minimum_capability() == (8, 0)

    def test_raw_capability_probe_normalises_valid_tuples(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            get_device_capability=lambda idx: ("8", 6)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert _device._capability(0) == (8, 6)

    def test_raw_capability_probe_rejects_unusable_shapes(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            get_device_capability=lambda idx: ("bad", 6)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert _device._capability(0) is None
        fake_torch.cuda.get_device_capability = lambda idx: (8, 6, 0)
        assert _device._capability(0) is None

    def test_visible_device_count_uses_cuda_availability_and_count(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: "2",
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert _device._visible_device_count() == 2
        fake_torch.cuda.is_available = lambda: False
        assert _device._visible_device_count() == 0

    def test_visible_device_count_suppresses_cuda_initialisation_warning(
        self,
        monkeypatch,
        recwarn,
    ):
        def _warn_then_false():
            warnings.warn(
                "CUDA initialization: Unexpected driver mismatch",
                UserWarning,
                stacklevel=1,
            )
            return False

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=_warn_then_false,
            device_count=lambda: 4,
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert _device._visible_device_count() == 0
        assert not [w for w in recwarn if "CUDA initialization:" in str(w.message)]

    def test_cuda_usable_for_default_and_indexed_devices(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_capability=lambda idx: [(8, 0), (6, 1)][idx],
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(_device, "_minimum_capability", lambda: (7, 0))

        assert _device._cuda_usable_for("cuda")
        assert not _device._cuda_usable_for("cuda:1")
        assert not _device._cuda_usable_for("cuda:9")

    def test_cuda_usable_for_rejects_unavailable_cuda(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 1,
            get_device_capability=lambda idx: (8, 0),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert not _device._cuda_usable_for("cuda:0")

    def test_cuda_usable_for_rejects_bad_capability_and_parse_errors(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_capability=lambda idx: ("bad", 0),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert not _device._cuda_usable_for("cuda:0")
        fake_torch.cuda.get_device_capability = lambda idx: (8, 0, 0)
        assert not _device._cuda_usable_for("cuda:0")
        assert not _device._cuda_usable_for("cuda:bad")

    def test_release_torch_cuda_collects_and_empties_available_cache(self, monkeypatch):
        calls: list[str] = []
        fake_gc = types.ModuleType("gc")
        fake_gc.collect = lambda: calls.append("collect")
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append("empty_cache"),
        )
        monkeypatch.setitem(sys.modules, "gc", fake_gc)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        _device.release_torch_cuda()

        assert calls == ["collect", "empty_cache"]

    def test_release_torch_cuda_suppresses_cuda_initialisation_warning(
        self,
        monkeypatch,
        recwarn,
    ):
        calls: list[str] = []
        fake_gc = types.ModuleType("gc")
        fake_gc.collect = lambda: calls.append("collect")

        def _warn_then_false():
            warnings.warn(
                "CUDA initialization: Unsupported HW",
                UserWarning,
                stacklevel=1,
            )
            return False

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=_warn_then_false,
            empty_cache=lambda: calls.append("empty_cache"),
        )
        monkeypatch.setitem(sys.modules, "gc", fake_gc)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        _device.release_torch_cuda()

        assert calls == ["collect"]
        assert not [w for w in recwarn if "CUDA initialization:" in str(w.message)]
