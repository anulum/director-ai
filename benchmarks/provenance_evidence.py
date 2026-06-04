# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Provenance evidence packet

"""Generate local evidence for KB lineage and provenance hardening.

The packet checks the production-relevant R9 primitives without model downloads:

* tenant-scoped KB snapshot Merkle roots change across fact evolution;
* protected signed-fact conflicts are reported without raw fact payloads;
* citation facts produce verifiable Merkle proofs;
* the HMAC provenance chain verifies across healthy and tampered fact bundles.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess  # nosec B404
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._common import save_results
from director_ai.core.provenance import (
    CitationFact,
    MerkleTree,
    ProvenanceChain,
    ProvenanceVerifier,
    SourceCredibility,
)
from director_ai.core.vector_store import InMemoryBackend, VectorGroundTruthStore


def _git_commit() -> str:
    git = shutil.which("git")
    if not git:
        return "unknown"
    try:
        completed = subprocess.run(  # nosec B603
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def run_kb_lineage_probe() -> dict[str, Any]:
    """Return evidence that KB evolution changes tenant-scoped roots."""

    store = VectorGroundTruthStore(backend=InMemoryBackend())
    tenant = "tenant-alpha"
    store.add_fact(
        "refund-window",
        "Refunds are available for thirty days.",
        tenant_id=tenant,
        metadata={
            "claim_id": "refund-window",
            "signed_fact_id": "signed-refund-v1",
            "claim_source": "signed_fact",
            "source_timestamp": "1700000000",
            "citation_status": "active",
        },
    )
    initial_root = store.kb_snapshot_root(tenant)
    initial_record = store.fact_version_record("refund-window", tenant)

    replacement = store.replace_fact(
        "refund-window",
        "Refunds are available for forty-five days.",
        tenant_id=tenant,
        reason="policy update",
        metadata={
            "claim_id": "refund-window",
            "signed_fact_id": "signed-refund-v2",
            "claim_source": "signed_fact",
            "updated_timestamp": "1710000000",
            "citation_status": "superseded",
        },
    )
    replaced_root = store.kb_snapshot_root(tenant)

    retraction = store.retract_fact(
        "refund-window",
        tenant_id=tenant,
        reason="source withdrawn",
    )
    retracted_root = store.kb_snapshot_root(tenant)

    store.add_fact(
        "dose-signed",
        "Dose is 5 mg.",
        tenant_id=tenant,
        metadata={
            "claim_id": "dose-claim",
            "signed_fact_id": "signed-dose-v1",
            "claim_source": "signed_fact",
        },
    )
    store.add_fact(
        "dose-incoming",
        "Dose is 10 mg.",
        tenant_id=tenant,
        metadata={"claim_id": "dose-claim"},
    )
    conflict = store.conflict_reports(tenant, key="dose-incoming")[0]
    audit_record = store.kb_snapshot_audit_record(tenant)
    root_sequence = [initial_root, replaced_root, retracted_root]
    roots_changed = len(set(root_sequence)) == len(root_sequence)
    tenant_root = store.kb_snapshot_root(tenant)
    other_store = VectorGroundTruthStore(backend=InMemoryBackend())
    other_store.add_fact("refund-window", "Other tenant fact.", tenant_id="other")
    tenant_scoped = tenant_root != other_store.kb_snapshot_root("other")

    return {
        "name": "kb_lineage",
        "tenant_id": tenant,
        "initial_root": initial_root,
        "replaced_root": replaced_root,
        "retracted_root": retracted_root,
        "roots_changed": roots_changed,
        "tenant_scoped": tenant_scoped,
        "initial_version": initial_record["version"] if initial_record else "",
        "replacement": {
            "from_version": replacement["from_version"],
            "to_version": replacement["to_version"],
            "reason": replacement["reason"],
        },
        "retraction": {
            "version": retraction["version"],
            "reason": retraction["reason"],
        },
        "conflict": {
            "conflict_type": conflict["conflict_type"],
            "claim_id": conflict["claim_id"],
            "signed_fact_id": conflict["signed_fact_id"],
            "reason": conflict["reason"],
        },
        "audit_record": audit_record,
        "raw_fact_payload_included": False,
        "passed": bool(
            roots_changed
            and tenant_scoped
            and conflict["conflict_type"] == "signed_fact"
            and audit_record["merkle_root"] == store.kb_snapshot_root(tenant)
        ),
    }


def run_provenance_chain_probe(*, fact_count: int = 4) -> dict[str, Any]:
    """Return Merkle proof and HMAC-chain evidence for citation facts."""

    if fact_count < 1:
        raise ValueError("fact_count must be >= 1")

    chain = ProvenanceChain(secret=b"p" * 32)
    credibility = SourceCredibility(prior=0.5)
    facts = [
        CitationFact(
            source_id=f"kb://source-{idx}",
            content=f"supported claim {idx}",
            timestamp=float(idx),
        )
        for idx in range(fact_count)
    ]
    for fact in facts:
        credibility.observe(fact.source_id, 1.0)

    tree = MerkleTree(facts)
    proofs = [tree.proof_for(fact) for fact in facts]
    verifier = ProvenanceVerifier(
        chain=chain,
        credibility=credibility,
        min_source_score=0.2,
    )
    healthy = verifier.verify(facts)

    tampered = CitationFact(source_id="kb://tampered", content="original", timestamp=0.0)
    object.__setattr__(tampered, "content", "changed after signing")
    tampered_verdict = verifier.verify([tampered])
    chain_ok, first_bad_index = chain.verify()

    return {
        "name": "provenance_chain",
        "fact_count": fact_count,
        "merkle_root": tree.root,
        "proofs_verified": all(proof.verify() for proof in proofs),
        "healthy_all_ok": healthy.all_ok,
        "healthy_chain_index": healthy.chain_entry.index,
        "healthy_trust_score": round(healthy.trust_score, 4),
        "tamper_detected": not tampered_verdict.all_ok,
        "tamper_failure_reason_present": bool(tampered_verdict.failures[0].reason),
        "chain_ok": chain_ok,
        "first_bad_index": first_bad_index,
        "chain_entries": len(chain),
        "raw_fact_payload_included": False,
        "passed": bool(
            all(proof.verify() for proof in proofs)
            and healthy.all_ok
            and not tampered_verdict.all_ok
            and chain_ok
            and first_bad_index is None
        ),
    }


def run_provenance_evidence(*, fact_count: int = 4) -> dict[str, Any]:
    """Return the complete local R9 provenance evidence packet."""

    kb = run_kb_lineage_probe()
    chain = run_provenance_chain_probe(fact_count=fact_count)
    passed = kb["passed"] and chain["passed"]
    return {
        "schema_version": "director-ai.provenance-evidence.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "acceptance": {
            "passed": passed,
            "checks": {
                "kb_lineage": kb["passed"],
                "provenance_chain": chain["passed"],
            },
            "limits": {
                "local_only": True,
                "external_operator_signoff_included": False,
            },
        },
        "probes": {
            "kb_lineage": kb,
            "provenance_chain": chain,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Director-AI provenance evidence packet.",
    )
    parser.add_argument(
        "--fact-count",
        type=int,
        default=4,
        help="Number of citation facts in the Merkle proof probe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to benchmarks/results/.",
    )
    args = parser.parse_args(argv)

    payload = run_provenance_evidence(fact_count=args.fact_count)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        save_results(payload, f"provenance_evidence_{stamp}.json")
    print(json.dumps(payload["acceptance"], indent=2))
    return 0 if payload["acceptance"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
