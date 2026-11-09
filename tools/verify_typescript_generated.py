# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native TypeScript generated artefact verification
"""Verify same-source SDK reproducibility, actual npm members and documentation.

Supplied archive and documentation roots are explicit artefact inputs. Successful
local comparison does not attest a remote publication or another builder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from tools import (
    ci_workflow_inventory,
    quality_ratchet_contracts,
    quality_ratchet_inventory,
    typescript_generated_recipe,
    typescript_package_artifacts,
)
from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import decode_quality_json
from tools.quality_ratchet_inventory import source_path
from tools.typescript_generated_recipe import (
    _digest,
    _directory,
    _identity,
    _remaining,
    capture_toolchain,
    run_native,
)
from tools.typescript_package_artifacts import inspect_npm_package

SDK = "packages/vercel-ai"
OUTPUT_ROOTS = ("dist", "build/api-docs")
NATIVE_VERSIONS = {"node": "26.13.0", "typescript": "6.0.3", "typedoc": "0.28.20"}
COMPILER_CONFIGURATION = {
    "compilerOptions": {
        "declaration": True,
        "module": "NodeNext",
        "moduleResolution": "NodeNext",
        "noUncheckedIndexedAccess": True,
        "outDir": "dist",
        "rootDir": "src",
        "strict": True,
        "target": "ES2022",
    },
    "include": ["src/**/*.ts"],
}
DOCUMENTATION_CONFIGURATION = {
    "$schema": "https://typedoc.org/schema.json",
    "entryPoints": ["src/index.ts"],
    "out": "build/api-docs",
    "readme": "README.md",
    "name": "Director-AI Vercel AI SDK Middleware",
    "includeVersion": True,
    "treatWarningsAsErrors": True,
    "treatValidationWarningsAsErrors": True,
    "validation": {"invalidLink": True, "notDocumented": True, "notExported": True},
}


@dataclass(frozen=True)
class GeneratedVerification:
    """Bind source, tool, package and separate documentation artefact identities.

    Attributes
    ----------
    source_commit : str
        Actual source HEAD observed before and after generation.
    git_remote : str | None
        Safe effective origin URL, or absence. Authenticated URLs always refuse.
    package_version : str
        Actual SDK package version used by native TypeDoc includeVersion.
    inputs : dict[str, str]
        All real Git-visible SDK authored/config/test input byte digests.
    input_modes : dict[str, int]
        Actual input permission bits, without modifying local source permissions.
    dependencies : dict[str, str]
        Every actual installed dependency file and internal link identity.
    node_identity : str
        Selected real executable byte digest.
    node_mode : int
        Selected executable permission bits.
    engine_identity : str
        Digest of the actual verifier, archive and native recipe implementations.
    outputs : dict[str, str]
        All dist and separate documentation files relative to the artefact root.
    output_modes : dict[str, int]
        Actual regenerated permission bits for those complete output trees.
    package_archive_sha256 : str
        Original compressed supplied npm archive, without rewriting it.
    package_members : dict[str, tuple[str, int]]
        Exact npm member names, byte digests and permission modes.
    identity : str
        Complete effective source/context/tool/artefact identity.
    """

    source_commit: str
    git_remote: str | None
    package_version: str
    inputs: dict[str, str]
    input_modes: dict[str, int]
    dependencies: dict[str, str]
    node_identity: str
    node_mode: int
    engine_identity: str
    outputs: dict[str, str]
    output_modes: dict[str, int]
    package_archive_sha256: str
    package_members: dict[str, tuple[str, int]]
    identity: str


def _outputs(root: Path, deadline: float) -> dict[str, str]:
    """Inventory both complete generated trees without following symbolic entries."""
    files: dict[str, str] = {}
    for relative in OUTPUT_ROOTS:
        parent = _directory(root / relative)
        for directory, directories, names in os.walk(parent, followlinks=False):
            base = Path(directory)
            _remaining(deadline)
            if not directories and not names:
                raise ValueError(
                    "Generated TypeScript output contains an empty directory"
                )
            for name in directories:
                if (base / name).is_symlink():
                    raise ValueError(
                        "Generated TypeScript output contains a symbolic directory"
                    )
            for name in names:
                path = base / name
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
                    raise ValueError(
                        "Generated TypeScript output is empty or nonregular"
                    )
                files[path.relative_to(root).as_posix()] = _digest(path, deadline)
    if (
        not {"dist/index.js", "dist/index.d.ts", "build/api-docs/index.html"}
        <= files.keys()
    ):
        raise ValueError(
            "Generated TypeScript output lacks required public entry points"
        )
    return dict(sorted(files.items()))


def _remote(
    root: Path,
    workspace: Path,
    environment: dict[str, str],
    deadline: float,
    label: str,
) -> str | None:
    """Validate effective source-link context before it can enter any receipt."""
    value = run_native(
        ["git", "remote", "get-url", "origin"],
        root,
        deadline,
        environment,
        workspace / (label + "-remote-command.json"),
        accepted_returncodes=(0, 2),
        record_output=False,
    ).removesuffix("\n")
    if not value:
        return None
    parsed = urlsplit(value)
    safe_url = (
        parsed.scheme in {"https", "http", "git", "ssh"}
        and bool(parsed.hostname)
        and parsed.username in {None, "git"}
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )
    safe_scp = re.fullmatch(r"git@[a-zA-Z0-9._-]+:[a-zA-Z0-9._/-]+", value) is not None
    if any(ord(c) <= 32 for c in value) or not (safe_url or safe_scp):
        raise ValueError(
            "Native TypeScript Git source-link context is not a safe origin"
        )
    with (workspace / (label + "-remote-context.json")).open("x") as stream:
        json.dump({"origin": value}, stream)
        stream.write("\n")
    return value


def verify_typescript_generated(
    root: Path,
    artefacts: Path,
    workspace: Path,
    package_archive: Path,
    node: str = "node",
    *,
    timeout: float = 120.0,
) -> GeneratedVerification:
    """Check actual SDK package and documentation against fresh native outputs.

    Parameters
    ----------
    root : Path
        Exact source Git root containing the maintained SDK sources/configuration.
    artefacts : Path
        Real supplied/staging root containing dist and separate build/api-docs trees.
    workspace : Path
        New caller-owned path outside source and supplied artefacts. Receipts and
        native output remain for caller custody; existing paths refuse.
    package_archive : Path
        Actual gzip npm archive to compare with source and fresh compiler bytes.
        The caller owns the authentic npm pack invocation and its original receipt.
    node : str
        Real pinned Node executable or explicit native launcher.
    timeout : float
        Positive finite whole deadline; process termination has a five-second grace.

    Returns
    -------
    GeneratedVerification
        Exact reproducibility/member/byte/mode agreement and original identities.
        This does not attest remote publication, whole Studio or baseline authority.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        If source, context, native recipe, archive, output or custody cannot qualify.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Generated TypeScript deadline must be positive and finite")
    deadline = time.monotonic() + timeout
    root, artefacts = _directory(root), _directory(artefacts)
    expected = _outputs(artefacts, deadline)
    expected_modes = {
        name: stat.S_IMODE(source_path(artefacts, name).stat().st_mode)
        for name in expected
    }
    package_archive = source_path(
        _directory(package_archive.absolute().parent), package_archive.name
    )
    original_archive = _digest(package_archive, deadline)
    if workspace.name in {"", ".", ".."} or any(
        workspace.absolute().is_relative_to(p) for p in (root, artefacts)
    ):
        raise ValueError("Generated TypeScript workspace overlaps protected inputs")
    workspace = _directory(workspace.parent) / workspace.name
    workspace.mkdir(mode=0o700)
    temporary, cache = workspace / "tmp", workspace / "cache"
    temporary.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    environment = os.environ.copy()
    environment.update(
        TMPDIR=str(temporary),
        XDG_CACHE_HOME=str(cache),
        NPM_CONFIG_CACHE=str(cache / "npm"),
        NODE_COMPILE_CACHE=str(cache / "node"),
    )
    sdk = _directory(root / SDK)
    if (
        decode_quality_json(source_path(sdk, "tsconfig.json").read_bytes())
        != COMPILER_CONFIGURATION
        or decode_quality_json(source_path(sdk, "typedoc.json").read_bytes())
        != DOCUMENTATION_CONFIGURATION
    ):
        raise ValueError(
            "Native TypeScript configuration differs from the qualified recipe"
        )
    package = object_mapping(
        decode_quality_json(source_path(sdk, "package.json").read_bytes())
    )
    version = package.get("version")
    if (
        not isinstance(version, str)
        or not version.strip()
        or package.get("files") != ["dist", "README.md", "LICENSE", "LICENSE.license"]
        or package.get("main") != "./dist/index.js"
        or package.get("types") != "./dist/index.d.ts"
        or "bin" in package
    ):
        raise ValueError(
            "Native SDK public package contract differs from the qualified recipe"
        )

    def git(arguments: list[str], name: str) -> str:
        """Execute actual Git within the same shared caller deadline."""
        return run_native(
            ["git", *arguments], root, deadline, environment, workspace / name
        )

    if Path(git(["rev-parse", "--show-toplevel"], "git-root.json").strip()) != root:
        raise ValueError(
            "Generated TypeScript verification requires the actual Git root"
        )
    commit = git(["rev-parse", "HEAD"], "git-head.json").strip()
    remote = _remote(root, workspace, environment, deadline, "before")

    def inputs(name: str) -> dict[str, str]:
        """Capture all actual Git-visible SDK authored, test and recipe inputs."""
        paths = git(
            ["ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", SDK],
            name,
        )
        return {
            p: _digest(source_path(root, p), deadline)
            for p in sorted(set(paths.split("\0")))
            if p
        }

    original = inputs("inputs-before.json")
    if (
        not {
            f"{SDK}/{name}"
            for name in (
                "src/index.ts",
                "package.json",
                "package-lock.json",
                "tsconfig.json",
                "tsconfig.test.json",
                "typedoc.json",
                "README.md",
                "LICENSE",
                "LICENSE.license",
            )
        }
        <= original.keys()
    ):
        raise ValueError(
            "Actual Git-visible SDK generator input inventory is incomplete"
        )
    input_modes = {
        name: stat.S_IMODE(source_path(root, name).stat().st_mode) for name in original
    }
    tool_sdk = _directory(
        Path(quality_ratchet_contracts.__file__).resolve().parents[1] / SDK
    )
    if (
        source_path(tool_sdk, "package-lock.json").read_bytes()
        != source_path(sdk, "package-lock.json").read_bytes()
    ):
        raise ValueError(
            "Selected SDK source lock differs from the installed generator graph"
        )
    tooling = capture_toolchain(tool_sdk, node, deadline=deadline)
    engine_paths = [
        Path(__file__),
        Path(ci_workflow_inventory.__file__),
        Path(quality_ratchet_inventory.__file__),
        Path(quality_ratchet_contracts.__file__),
        Path(typescript_generated_recipe.__file__),
        Path(typescript_package_artifacts.__file__),
    ]
    engine = _identity([_digest(p, deadline) for p in engine_paths])
    metadata_code = "const fs=require('node:fs');console.log(JSON.stringify({node:process.versions.node,typescript:require('typescript').version,typedoc:JSON.parse(fs.readFileSync('node_modules/typedoc/package.json','utf8')).version}));"
    metadata = run_native(
        [str(tooling.executable), "-e", metadata_code],
        tool_sdk,
        deadline,
        environment,
        workspace / "native-versions.json",
    )
    if decode_quality_json(metadata.encode()) != NATIVE_VERSIONS:
        raise ValueError("Actual native TypeScript recipe versions differ")
    candidate = workspace / "candidate"
    commands = [
        [
            str(tooling.executable),
            str(tool_sdk / "node_modules/typescript/bin/tsc"),
            "--project",
            str(sdk / "tsconfig.json"),
            "--outDir",
            str(candidate / "dist"),
        ],
        [
            str(tooling.executable),
            str(tool_sdk / "node_modules/typedoc/bin/typedoc"),
            "--options",
            str(sdk / "typedoc.json"),
            "--out",
            str(candidate / "build/api-docs"),
        ],
    ]
    for index, command in enumerate(commands):
        run_native(
            command, sdk, deadline, environment, workspace / f"generator-{index}.json"
        )
    actual = _outputs(candidate, deadline)
    modes = {
        name: stat.S_IMODE(source_path(candidate, name).stat().st_mode)
        for name in actual
    }
    with (workspace / "output-comparison.json").open("x") as stream:
        json.dump(
            {
                "expected": expected,
                "actual": actual,
                "expected_modes": expected_modes,
                "actual_modes": modes,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    if actual != expected or modes != expected_modes:
        raise ValueError(
            "Supplied TypeScript artefacts differ from the actual native recipe"
        )
    members = {
        "package/" + name: (digest, modes[name] & 0o755)
        for name, digest in actual.items()
        if name.startswith("dist/")
    }
    for name in ("package.json", "README.md", "LICENSE", "LICENSE.license"):
        relative = f"{SDK}/{name}"
        members["package/" + name] = (original[relative], input_modes[relative] & 0o755)
    packed = inspect_npm_package(package_archive, members, deadline=deadline)
    if packed.archive_sha256 != original_archive:
        raise ValueError("Original npm package changed during native generation")
    if (
        inputs("inputs-after.json") != original
        or {
            name: stat.S_IMODE(source_path(root, name).stat().st_mode)
            for name in original
        }
        != input_modes
        or git(["rev-parse", "HEAD"], "git-head-after.json").strip() != commit
        or _remote(root, workspace, environment, deadline, "after") != remote
        or capture_toolchain(tool_sdk, node, deadline=deadline) != tooling
        or _identity([_digest(p, deadline) for p in engine_paths]) != engine
        or _outputs(artefacts, deadline) != expected
        or {
            name: stat.S_IMODE(source_path(artefacts, name).stat().st_mode)
            for name in expected
        }
        != expected_modes
        or _digest(package_archive, deadline) != original_archive
    ):
        raise ValueError(
            "Generated TypeScript source, context, tool or artefact custody changed"
        )
    payload = {
        "profile": "typescript-sdk-generated",
        "source_commit": commit,
        "git_remote": remote,
        "package_version": version,
        "inputs": original,
        "input_modes": input_modes,
        "dependencies": tooling.dependencies,
        "node_identity": tooling.executable_sha256,
        "node_mode": tooling.executable_mode,
        "engine_identity": engine,
        "outputs": actual,
        "output_modes": modes,
        "package_archive_sha256": packed.archive_sha256,
        "package_members": packed.members,
        "versions": NATIVE_VERSIONS,
        "output_roots": OUTPUT_ROOTS,
    }
    identity = _identity(payload)
    _remaining(deadline)
    return GeneratedVerification(
        commit,
        remote,
        version,
        original,
        input_modes,
        tooling.dependencies,
        tooling.executable_sha256,
        tooling.executable_mode,
        engine,
        actual,
        modes,
        packed.archive_sha256,
        packed.members,
        identity,
    )


def main(argv: list[str] | None = None) -> int:
    """Compare real source, package and documentation through the public CLI.

    Parameters
    ----------
    argv : list[str] | None
        Explicit arguments or the process command line.

    Returns
    -------
    int
        Zero for exact reproducibility/member agreement, two for input/native refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artefacts", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--timeout", type=float, default=120.0)
    arguments = parser.parse_args(argv)
    try:
        result = verify_typescript_generated(
            arguments.root,
            arguments.artefacts,
            arguments.workspace,
            arguments.package,
            arguments.node,
            timeout=arguments.timeout,
        )
        print(json.dumps(asdict(result), sort_keys=True))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError):
        print("Native TypeScript generated artefact verification refused")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
