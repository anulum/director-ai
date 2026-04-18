# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Torch device selection helper

"""Pick a PyTorch device that the installed PyTorch binary can
actually run kernels on.

``torch.cuda.is_available()`` returns ``True`` for any CUDA GPU the
driver can see, including devices whose compute capability is below
the minimum supported by the installed PyTorch wheel. Running a
kernel on such a device yields
``torch.AcceleratorError: no kernel image is available for execution
on the device`` at the first forward pass, deep inside whatever
caller triggered the load. Mining rigs typically expose a GTX
10-series card (sm_61) alongside the main workhorse; current
PyTorch wheels drop sm_6x support and the scorer crashes.

:func:`select_torch_device` walks every visible CUDA device and
returns the first one whose ``(major, minor)`` capability matches
PyTorch's supported list. If no CUDA device qualifies, it falls
back to ``"cpu"`` with a one-shot warning so operators know their
GPU is sitting idle.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import cast

logger = logging.getLogger("DirectorAI.Device")

_warn_lock = threading.Lock()
_warned_unsupported: set[int] = set()


def _minimum_capability() -> tuple[int, int]:
    """Query PyTorch for its lowest supported compute capability.

    Older PyTorch wheels expose ``torch.cuda.get_arch_list`` which
    returns e.g. ``["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]``.
    Parse the lowest entry to get the minimum capability. On
    fallback paths (custom builds that omit the list) return
    ``(7, 0)`` as the conservative PyTorch 2.x default.
    """
    try:
        import torch

        arches = torch.cuda.get_arch_list()
    except (ImportError, AttributeError):
        return (7, 0)
    mins = (10, 0)
    for arch in arches:
        if not arch.startswith("sm_"):
            continue
        try:
            num = arch[3:]
            major = int(num[:-1])
            minor = int(num[-1])
        except (TypeError, ValueError):
            continue
        if (major, minor) < mins:
            mins = (major, minor)
    return mins if mins != (10, 0) else (7, 0)


def _capability(device_index: int) -> tuple[int, int] | None:
    try:
        import torch

        # ``get_device_capability`` returns a tuple at runtime but
        # torch does not ship strict type stubs; cast documents the
        # narrowing at the FFI-ish boundary.
        return cast("tuple[int, int]", torch.cuda.get_device_capability(device_index))
    except Exception:  # pragma: no cover — defensive
        return None


def _visible_device_count() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:  # pragma: no cover — defensive
        return 0


def select_torch_device(preferred: str | None = None) -> str:
    """Return a PyTorch device string that the installed binary can
    run kernels on.

    Honour ``DIRECTOR_FORCE_CPU=1`` regardless of what GPUs are
    visible — operators on noisy test environments appreciate a
    blunt override. Honour an explicit ``preferred`` argument
    (``"cpu"``, ``"cuda"``, ``"cuda:N"``) when possible; fall
    through to the capability walk otherwise.
    """
    if os.environ.get("DIRECTOR_FORCE_CPU", "").lower() in {"1", "true", "yes"}:
        return "cpu"
    if preferred is not None:
        lowered = preferred.lower()
        if lowered == "cpu":
            return "cpu"
        if lowered.startswith("cuda") and _cuda_usable_for(lowered):
            return preferred
        # fall through — preferred was cuda-shaped but the device is
        # unusable; pick the best available instead of crashing.
    count = _visible_device_count()
    if count == 0:
        return "cpu"
    min_cap = _minimum_capability()
    for idx in range(count):
        cap = _capability(idx)
        if cap is not None and cap >= min_cap:
            return f"cuda:{idx}"
    _warn_once_unsupported(count, min_cap)
    return "cpu"


def _cuda_usable_for(device: str) -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        idx = int(device.split(":", 1)[1]) if ":" in device else 0
        if idx >= torch.cuda.device_count():
            return False
        cap = cast("tuple[int, int]", torch.cuda.get_device_capability(idx))
        return cap >= _minimum_capability()
    except Exception:  # pragma: no cover — defensive
        return False


def _warn_once_unsupported(count: int, min_cap: tuple[int, int]) -> None:
    """Emit one log line per unique (count, min_cap) pair so
    stream output does not balloon when many callers fall back."""
    key = hash((count, min_cap))
    with _warn_lock:
        if key in _warned_unsupported:
            return
        _warned_unsupported.add(key)
    logger.warning(
        "no CUDA device with capability >= sm_%d%d (%d visible) — "
        "running on CPU. Set DIRECTOR_FORCE_CPU=1 to silence this "
        "at startup.",
        min_cap[0],
        min_cap[1],
        count,
    )


def reset_warn_cache() -> None:
    """Test hook — clear the one-shot warning cache."""
    with _warn_lock:
        _warned_unsupported.clear()
