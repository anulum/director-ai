# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual native quality ratchet command
"""Compare actual native diagnostics with immutable original quality allowances."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.quality_ratchet_baseline import (
    BASELINE_PATH,
    Baseline,
    baseline_payload,
    parse_baseline,
    portable_baseline,
    portable_snapshot,
    read_committed_baseline,
    resolve_commit,
)
from tools.quality_ratchet_contracts import Snapshot, snapshot_payload
from tools.quality_ratchet_gate import (
    POLICY_PATH,
    check_native_baseline,
    compare_required_baselines,
    policy_origin,
    qualify_required_record,
    required_authority,
    required_baseline_scope,
    required_input_state,
    verify_required_captures,
)
from tools.quality_ratchet_inventory import source_path
from tools.quality_ratchet_profiles import (
    REQUIRED_PROFILES,
    capture_profile,
    capture_required_profiles,
)
from tools.quality_ratchet_succession import Transition
from tools.quality_ratchet_succession_load import canary_refusals, load_transitions
from tools.quality_ratchet_walk_tip import entries_since, range_canary_refusals


def write_original_json(path: Path, payload: dict[str, object]) -> None:
    """Publish complete JSON bytes atomically without replacing an existing file.

    Parameters
    ----------
    path : Path
        Explicit selected output path in an existing directory.
    payload : dict[str, object]
        Complete validated carrier to serialize.

    Raises
    ------
    OSError, ValueError
        If publication, encoding or an existing output prevents an original write.
    """
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def _profile_summary(actual: Snapshot) -> dict[str, object]:
    """Summarise one native snapshot without discarding its diagnostic status."""
    return {
        "profile": actual.profile,
        "complete": actual.complete,
        "sources": len(actual.sources),
        "findings": len(actual.findings),
        "advisories": list(actual.advisories),
        "source_inventory_sha256": hashlib.sha256(
            json.dumps(actual.sources, sort_keys=True).encode()
        ).hexdigest(),
    }


def _run_required(root: Path, args: argparse.Namespace) -> int:
    """Run the complete required profile set against trusted baseline authority."""
    previous = None
    candidate = None
    transitions: tuple[Transition, ...] = ()
    policy = source_path(root, POLICY_PATH).read_bytes()
    if args.command == "capture":
        origin = policy_origin(policy)
    else:
        base = resolve_commit(root, args.base_ref)
        origin, previous = required_authority(
            root, base, recording=args.command == "record"
        )
        transitions = load_transitions(root, base)
        if args.command == "check":
            candidate = parse_baseline(source_path(root, BASELINE_PATH).read_bytes())
            required_baseline_scope(candidate, origin)
    before = required_input_state(root)
    actual = portable_baseline(
        Baseline(origin, capture_required_profiles(root, args)), root
    )
    required_baseline_scope(actual, origin)
    verify_required_captures(root, actual, before)
    if source_path(root, POLICY_PATH).read_bytes() != policy:
        raise ValueError("Required quality policy changed during analysis")
    if (
        candidate is not None
        and parse_baseline(source_path(root, BASELINE_PATH).read_bytes()) != candidate
    ):
        raise ValueError("Required quality candidate baseline changed during analysis")
    refusals: tuple[str, ...] = ()
    if args.command == "capture":
        payload = baseline_payload(actual)
        payload["schema"] = "quality-ratchet.capture.v1"
        write_original_json(args.output, payload)
    elif previous is None:
        qualify_required_record(root, base, actual)
        write_original_json(args.output, baseline_payload(actual))
    else:
        proposed = candidate if candidate is not None else actual
        refusals = compare_required_baselines(
            previous, proposed, actual, origin, transitions
        ) + canary_refusals(root, args, previous.profiles, actual.profiles, transitions)
        if args.canaries_since is not None:
            since = resolve_commit(root, args.canaries_since)
            refusals = tuple(
                sorted(
                    set(refusals)
                    | set(
                        range_canary_refusals(
                            root,
                            args,
                            entries_since(root, since, transitions),
                            actual.profiles,
                        )
                    )
                )
            )
        if not refusals and args.command == "record":
            write_original_json(args.output, baseline_payload(actual))
    print(
        json.dumps(
            {
                "complete": True,
                "profiles": {
                    name: _profile_summary(snapshot)
                    for name, snapshot in actual.profiles.items()
                },
                "refusals": refusals,
            }
        )
    )
    return 1 if refusals else 0


def main(argv: list[str] | None = None) -> int:
    """Run the real native capture, adoption or immutable-base comparison CLI.

    Parameters
    ----------
    argv : list[str] | None
        Explicit arguments or the native process command line.

    Returns
    -------
    int
        Zero for this qualified profile; one for regression and two for refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "record", "check"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--go", default="go")
    parser.add_argument("--revive", default="revive")
    parser.add_argument("--node", default="node")
    parser.add_argument("--protoc", default="protoc")
    parser.add_argument("--lean", default="lean")
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--rustc", default="rustc")
    parser.add_argument("--timeout", type=float, default=120.0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--profile", choices=REQUIRED_PROFILES, default="python-docs"
    )
    selection.add_argument(
        "--all", action="store_true", help="Run every required native profile"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--canaries-since",
        help="Also run the canaries of every entry added after this commit",
    )
    args = parser.parse_args(argv)
    if args.command != "check" and args.output is None:
        parser.error("capture and record require an explicit --output")
    try:
        root = args.root.resolve(strict=True)
        if args.all:
            return _run_required(root, args)
        actual = portable_snapshot(capture_profile(root, args, args.profile), root)
        if args.command == "capture":
            write_original_json(args.output, snapshot_payload(actual))
            refusals: tuple[str, ...] = ()
        else:
            base = resolve_commit(root, args.base_ref)
            transitions = load_transitions(root, base)
            original = read_committed_baseline(root, base)
            if args.command == "record":
                origin = base if original is None else original.origin_commit
                candidate = Baseline(origin, {actual.profile: actual})
            else:
                candidate = parse_baseline(
                    source_path(root, BASELINE_PATH).read_bytes()
                )
            refusals = check_native_baseline(root, base, candidate, actual, transitions)
            if original is not None:
                refusals = tuple(
                    sorted(
                        set(refusals)
                        | set(
                            canary_refusals(
                                root,
                                args,
                                original.profiles,
                                {actual.profile: actual},
                                transitions,
                            )
                        )
                    )
                )
            if not refusals and args.command == "record":
                write_original_json(args.output, baseline_payload(candidate))
        print(
            json.dumps(
                {
                    **_profile_summary(actual),
                    "refusals": refusals,
                }
            )
        )
        return 1 if refusals else 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Native quality refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
