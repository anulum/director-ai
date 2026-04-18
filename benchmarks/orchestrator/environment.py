# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — orchestrator environment capture

"""Captures the full ``(commit, hardware, python, package)`` tuple
needed to replicate a benchmark run.

The principle is **no fabrication**: every field is read from the
system at capture time. If a value cannot be determined (e.g. no
GPU present, not a git checkout), the corresponding field is the
empty string or zero. It is never invented.

Two capture helpers:

* :func:`capture_environment` — full fingerprint with git +
  hardware probing. Call once at the start of a run.
* :class:`EnvironmentFingerprint` — dataclass alias for the
  :class:`EnvironmentRecord` in :mod:`.schema`; exposed here so
  callers importing from :mod:`.environment` do not have to
  reach into :mod:`.schema`.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass

from .schema import EnvironmentRecord

EnvironmentFingerprint = EnvironmentRecord


@dataclass
class _GPUInfo:
    model: str
    count: int
    memory_gb: float


def capture_environment(runner: str = "local") -> EnvironmentRecord:
    """Probe the current process and host for the fingerprint that
    goes into every :class:`RunReport`.

    ``runner`` should be one of ``"local"``, ``"vertex"``,
    ``"ci"``, ``"remote"`` — the ``DIRECTOR_RUN_ENV`` environment
    variable overrides the explicit argument so cloud orchestrators
    can set it through Vertex AI custom-job env vars.
    """
    resolved_runner = os.environ.get("DIRECTOR_RUN_ENV", runner)
    commit, dirty, branch = _git_state()
    gpu = _probe_gpu()
    return EnvironmentRecord(
        git_commit=commit,
        git_dirty=dirty,
        git_branch=branch,
        package_version=_package_version(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count() or 0,
        ram_gb=_ram_gb(),
        gpu_model=gpu.model,
        gpu_count=gpu.count,
        gpu_memory_gb=gpu.memory_gb,
        runner=resolved_runner,
    )


def _git_state() -> tuple[str, bool, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return commit, bool(status.strip()), branch
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        # Not a git checkout, or git binary absent on the Vertex
        # container. Report the tuple honestly; a downstream check
        # still enforces ``git_commit != ""`` on the *local* runner.
        return "", False, ""


def _package_version() -> str:
    # Late-import: environment capture runs before the orchestrator
    # has a director_ai import on the path; keeping this out of the
    # module-level import list means environment.py works even
    # under a partial checkout.
    try:
        from director_ai import __version__

        return str(__version__)
    except ImportError:
        return ""


def _cpu_model() -> str:
    """Read the CPU model string from ``/proc/cpuinfo`` on Linux,
    ``sysctl`` on macOS, or :mod:`platform` as a last resort."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                timeout=5,
            ).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    return platform.processor() or ""


def _ram_gb() -> float:
    """Total physical RAM in GiB. Returns 0 when the value cannot
    be determined rather than fabricating a plausible default."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 * 1024), 2)
        except (OSError, ValueError, IndexError):
            return 0.0
    elif sys.platform == "darwin":
        try:
            bytes_total = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                    timeout=5,
                ).strip()
            )
            return round(bytes_total / (1024**3), 2)
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            return 0.0
    return 0.0


def _probe_gpu() -> _GPUInfo:
    """Probe ``nvidia-smi`` for GPU info. Vertex AI workers ship
    nvidia-smi in the standard container; absent a CUDA device the
    helper returns zeros."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return _GPUInfo(model="", count=0, memory_gb=0.0)
    lines = [ln for ln in output.splitlines() if ln.strip()]
    if not lines:
        return _GPUInfo(model="", count=0, memory_gb=0.0)
    # Sum memory across devices; the "model" is the unique name or
    # the first one when heterogeneous. Multi-GPU hosts with mixed
    # models surface that fact in the string.
    models: list[str] = []
    total_mem_mib = 0.0
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        models.append(parts[0])
        try:
            total_mem_mib += float(parts[1])
        except ValueError:
            continue
    unique_models = sorted(set(models))
    model = unique_models[0] if len(unique_models) == 1 else " + ".join(unique_models)
    return _GPUInfo(
        model=model,
        count=len(models),
        memory_gb=round(total_mem_mib / 1024, 2),
    )
