# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - FrontierFail seed packet validation tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_frontierfail_packet.py"
SPEC = importlib.util.spec_from_file_location("validate_frontierfail_packet", VALIDATOR)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

validate_frontierfail_packet = MODULE.validate_frontierfail_packet


def _write_packet(root: Path, public_benchmark_eligible: bool = False) -> None:
    (root / "benchmarks").mkdir(parents=True, exist_ok=True)
    (root / "benchmarks" / "frontierfail_seed_packet.toml").write_text(
        f"""
schema_version = "1.0.0"
packet_id = "frontierfail-seed"
cases = "benchmarks/frontierfail_cases.jsonl"
public_benchmark_eligible = {str(public_benchmark_eligible).lower()}
claim_boundary = "Seed regression fixture only; not an externally validated benchmark."
minimum_cases = 2
minimum_public_incident_cases = 0
minimum_public_incident_categories = 0
minimum_public_incident_domains = 0
minimum_public_incident_publishers = 0
minimum_public_incident_evidence_refs = 0
required_categories = ["numeric_contradiction", "fabricated_policy"]
""".strip(),
        encoding="utf-8",
    )
    rows = [
        {
            "id": "ff-001",
            "source_type": "synthetic_regression",
            "category": "numeric_contradiction",
            "domain": "finance",
            "prompt": "Summarise the quarterly revenue.",
            "source": "The quarter closed with revenue of 4.2 million euros.",
            "bad_response": "The quarter closed with revenue of 7.8 million euros.",
            "expected_failure": "numeric contradiction",
            "expected_decision": "halt",
            "evidence_ref": "seed-taxonomy:numeric_contradiction",
            "redaction": "none",
            "benchmark_eligible": public_benchmark_eligible,
        },
        {
            "id": "ff-002",
            "source_type": "synthetic_regression",
            "category": "fabricated_policy",
            "domain": "support",
            "prompt": "State the refund policy.",
            "source": "Refunds are available within 30 days.",
            "bad_response": "Refunds are never available.",
            "expected_failure": "fabricated policy",
            "expected_decision": "halt",
            "evidence_ref": "seed-taxonomy:fabricated_policy",
            "redaction": "none",
            "benchmark_eligible": public_benchmark_eligible,
        },
    ]
    (root / "benchmarks" / "frontierfail_cases.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_frontierfail_seed_packet_validates_current_fixture() -> None:
    assert validate_frontierfail_packet(ROOT) == []


def test_frontierfail_packet_rejects_public_benchmark_seed_cases(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path, public_benchmark_eligible=True)

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_seed_packet.toml: seed packet must not set public_benchmark_eligible=true"
        in errors
    )
    assert (
        "benchmarks/frontierfail_cases.jsonl:1: synthetic_regression rows must not be benchmark_eligible"
        in errors
    )


def test_frontierfail_packet_rejects_missing_category_coverage(tmp_path: Path) -> None:
    _write_packet(tmp_path)
    path = tmp_path / "benchmarks" / "frontierfail_cases.jsonl"
    rows = [json.loads(path.read_text(encoding="utf-8").splitlines()[0])]
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_seed_packet.toml: category fabricated_policy has 0 cases"
        in errors
    )


def test_frontierfail_packet_enforces_minimum_public_incident_coverage(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path)
    packet_path = tmp_path / "benchmarks" / "frontierfail_seed_packet.toml"
    packet_path.write_text(
        packet_path.read_text(encoding="utf-8").replace(
            "minimum_public_incident_cases = 0",
            "minimum_public_incident_cases = 1",
        ),
        encoding="utf-8",
    )

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_seed_packet.toml: expected at least 1 public_incident benchmark-eligible cases, found 0"
        in errors
    )


def test_frontierfail_packet_requires_public_incident_evidence_metadata(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path)
    path = tmp_path / "benchmarks" / "frontierfail_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.append(
        {
            "id": "ff-public-001",
            "source_type": "public_incident",
            "category": "fabricated_policy",
            "domain": "travel",
            "prompt": "Answer using only the airline policy.",
            "source": "The controlling policy did not allow the requested refund.",
            "bad_response": "The requested refund is allowed.",
            "expected_failure": "fabricated public refund policy",
            "expected_decision": "halt",
            "evidence_ref": "https://example.test/public-incident-1",
            "redaction": "public incident summary",
            "benchmark_eligible": True,
        }
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_cases.jsonl:3: public_incident benchmark-eligible rows require evidence_publisher"
        in errors
    )
    assert (
        "benchmarks/frontierfail_cases.jsonl:3: public_incident benchmark-eligible rows require evidence_title"
        in errors
    )


def test_frontierfail_packet_requires_public_incident_access_date(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path)
    path = tmp_path / "benchmarks" / "frontierfail_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.append(
        {
            "id": "ff-public-001",
            "source_type": "public_incident",
            "category": "fabricated_policy",
            "domain": "travel",
            "prompt": "Answer using only the airline policy.",
            "source": "The controlling policy did not allow the requested refund.",
            "bad_response": "The requested refund is allowed.",
            "expected_failure": "fabricated public refund policy",
            "expected_decision": "halt",
            "evidence_ref": "https://example.test/public-incident-1",
            "evidence_publisher": "Example Publisher",
            "evidence_title": "Public incident one",
            "redaction": "public incident summary",
            "benchmark_eligible": True,
        }
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_cases.jsonl:3: public_incident benchmark-eligible rows require evidence_accessed_date"
        in errors
    )


def test_frontierfail_packet_rejects_invalid_public_incident_access_date(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path)
    path = tmp_path / "benchmarks" / "frontierfail_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.append(
        {
            "id": "ff-public-001",
            "source_type": "public_incident",
            "category": "fabricated_policy",
            "domain": "travel",
            "prompt": "Answer using only the airline policy.",
            "source": "The controlling policy did not allow the requested refund.",
            "bad_response": "The requested refund is allowed.",
            "expected_failure": "fabricated public refund policy",
            "expected_decision": "halt",
            "evidence_ref": "https://example.test/public-incident-1",
            "evidence_publisher": "Example Publisher",
            "evidence_title": "Public incident one",
            "evidence_accessed_date": "2026-02-31",
            "redaction": "public incident summary",
            "benchmark_eligible": True,
        }
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_cases.jsonl:3: public_incident benchmark-eligible rows require evidence_accessed_date"
        in errors
    )


def test_frontierfail_packet_rejects_future_public_incident_access_date(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path)
    path = tmp_path / "benchmarks" / "frontierfail_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.append(
        {
            "id": "ff-public-001",
            "source_type": "public_incident",
            "category": "fabricated_policy",
            "domain": "travel",
            "prompt": "Answer using only the airline policy.",
            "source": "The controlling policy did not allow the requested refund.",
            "bad_response": "The requested refund is allowed.",
            "expected_failure": "fabricated public refund policy",
            "expected_decision": "halt",
            "evidence_ref": "https://example.test/public-incident-1",
            "evidence_publisher": "Example Publisher",
            "evidence_title": "Public incident one",
            "evidence_accessed_date": "2999-01-01",
            "redaction": "public incident summary",
            "benchmark_eligible": True,
        }
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_cases.jsonl:3: public_incident benchmark-eligible rows require evidence_accessed_date"
        in errors
    )


def test_frontierfail_packet_rejects_duplicate_public_incident_evidence_refs(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path)
    path = tmp_path / "benchmarks" / "frontierfail_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.extend(
        [
            {
                "id": "ff-public-001",
                "source_type": "public_incident",
                "category": "fabricated_policy",
                "domain": "travel",
                "prompt": "Answer using only the airline policy.",
                "source": "The controlling policy did not allow the requested refund.",
                "bad_response": "The requested refund is allowed.",
                "expected_failure": "fabricated public refund policy",
                "expected_decision": "halt",
                "evidence_ref": "https://example.test/public-incident-1",
                "evidence_publisher": "Example Publisher",
                "evidence_title": "Public incident one",
                "evidence_accessed_date": "2026-05-18",
                "redaction": "public incident summary",
                "benchmark_eligible": True,
            },
            {
                "id": "ff-public-002",
                "source_type": "public_incident",
                "category": "unsupported_citation",
                "domain": "legal",
                "prompt": "Cite only verified authorities.",
                "source": "The cited authorities do not exist.",
                "bad_response": "The cited authorities support the claim.",
                "expected_failure": "fabricated public citation",
                "expected_decision": "halt",
                "evidence_ref": "https://example.test/public-incident-1",
                "evidence_publisher": "Example Publisher",
                "evidence_title": "Public incident duplicate",
                "evidence_accessed_date": "2026-05-18",
                "redaction": "public incident summary",
                "benchmark_eligible": True,
            },
        ]
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_cases.jsonl:4: duplicate public_incident evidence_ref https://example.test/public-incident-1"
        in errors
    )


def test_frontierfail_packet_enforces_public_incident_category_diversity(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path)
    packet_path = tmp_path / "benchmarks" / "frontierfail_seed_packet.toml"
    packet_path.write_text(
        packet_path.read_text(encoding="utf-8")
        .replace(
            "minimum_public_incident_cases = 0",
            "minimum_public_incident_cases = 2",
        )
        .replace(
            "minimum_public_incident_categories = 0",
            "minimum_public_incident_categories = 2",
        )
        .replace("minimum_cases = 2", "minimum_cases = 4"),
        encoding="utf-8",
    )
    path = tmp_path / "benchmarks" / "frontierfail_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.extend(
        [
            {
                "id": "ff-public-001",
                "source_type": "public_incident",
                "category": "fabricated_policy",
                "domain": "travel",
                "prompt": "Answer using only the airline policy.",
                "source": "The controlling policy did not allow the requested refund.",
                "bad_response": "The requested refund is allowed.",
                "expected_failure": "fabricated public refund policy",
                "expected_decision": "halt",
                "evidence_ref": "https://example.test/public-incident-1",
                "evidence_publisher": "Example Publisher",
                "evidence_title": "Public incident one",
                "evidence_accessed_date": "2026-05-18",
                "redaction": "public incident summary",
                "benchmark_eligible": True,
            },
            {
                "id": "ff-public-002",
                "source_type": "public_incident",
                "category": "fabricated_policy",
                "domain": "public_services",
                "prompt": "Answer using only official guidance.",
                "source": "The official guidance disallowed the exception.",
                "bad_response": "The exception is allowed.",
                "expected_failure": "fabricated public compliance policy",
                "expected_decision": "halt",
                "evidence_ref": "https://example.test/public-incident-2",
                "evidence_publisher": "Example Publisher",
                "evidence_title": "Public incident two",
                "evidence_accessed_date": "2026-05-18",
                "redaction": "public incident summary",
                "benchmark_eligible": True,
            },
        ]
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_seed_packet.toml: expected at least 2 public_incident benchmark-eligible categories, found 1"
        in errors
    )


def test_frontierfail_packet_enforces_public_incident_domain_diversity(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path)
    packet_path = tmp_path / "benchmarks" / "frontierfail_seed_packet.toml"
    packet_path.write_text(
        packet_path.read_text(encoding="utf-8")
        .replace(
            "minimum_public_incident_cases = 0",
            "minimum_public_incident_cases = 2",
        )
        .replace(
            "minimum_public_incident_categories = 0",
            "minimum_public_incident_categories = 2",
        )
        .replace(
            "minimum_public_incident_domains = 0",
            "minimum_public_incident_domains = 2",
        )
        .replace("minimum_cases = 2", "minimum_cases = 4"),
        encoding="utf-8",
    )
    path = tmp_path / "benchmarks" / "frontierfail_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.extend(
        [
            {
                "id": "ff-public-001",
                "source_type": "public_incident",
                "category": "fabricated_policy",
                "domain": "customer_support",
                "prompt": "Answer using only the airline policy.",
                "source": "The controlling policy did not allow the requested refund.",
                "bad_response": "The requested refund is allowed.",
                "expected_failure": "fabricated public refund policy",
                "expected_decision": "halt",
                "evidence_ref": "https://example.test/public-incident-1",
                "evidence_publisher": "Example Publisher",
                "evidence_title": "Public incident one",
                "evidence_accessed_date": "2026-05-18",
                "redaction": "public incident summary",
                "benchmark_eligible": True,
            },
            {
                "id": "ff-public-002",
                "source_type": "public_incident",
                "category": "unsupported_citation",
                "domain": "customer_support",
                "prompt": "Cite only verified authorities.",
                "source": "The cited authorities do not exist.",
                "bad_response": "The cited authorities support the claim.",
                "expected_failure": "fabricated public citation",
                "expected_decision": "halt",
                "evidence_ref": "https://example.test/public-incident-2",
                "evidence_publisher": "Example Publisher",
                "evidence_title": "Public incident two",
                "evidence_accessed_date": "2026-05-18",
                "redaction": "public incident summary",
                "benchmark_eligible": True,
            },
        ]
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_seed_packet.toml: expected at least 2 public_incident benchmark-eligible domains, found 1"
        in errors
    )


def test_frontierfail_packet_enforces_public_incident_publisher_diversity(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path)
    packet_path = tmp_path / "benchmarks" / "frontierfail_seed_packet.toml"
    packet_path.write_text(
        packet_path.read_text(encoding="utf-8")
        .replace(
            "minimum_public_incident_cases = 0",
            "minimum_public_incident_cases = 2",
        )
        .replace(
            "minimum_public_incident_publishers = 0",
            "minimum_public_incident_publishers = 2",
        )
        .replace("minimum_cases = 2", "minimum_cases = 4"),
        encoding="utf-8",
    )
    path = tmp_path / "benchmarks" / "frontierfail_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.extend(
        [
            {
                "id": "ff-public-001",
                "source_type": "public_incident",
                "category": "fabricated_policy",
                "domain": "travel",
                "prompt": "Answer using only the airline policy.",
                "source": "The controlling policy did not allow the requested refund.",
                "bad_response": "The requested refund is allowed.",
                "expected_failure": "fabricated public refund policy",
                "expected_decision": "halt",
                "evidence_ref": "https://example.test/public-incident-1",
                "evidence_publisher": "Example Publisher",
                "evidence_title": "Public incident one",
                "evidence_accessed_date": "2026-05-18",
                "redaction": "public incident summary",
                "benchmark_eligible": True,
            },
            {
                "id": "ff-public-002",
                "source_type": "public_incident",
                "category": "unsupported_citation",
                "domain": "legal",
                "prompt": "Cite only verified authorities.",
                "source": "The cited authorities do not exist.",
                "bad_response": "The cited authorities support the claim.",
                "expected_failure": "fabricated public citation",
                "expected_decision": "halt",
                "evidence_ref": "https://example.test/public-incident-2",
                "evidence_publisher": "Example Publisher",
                "evidence_title": "Public incident two",
                "evidence_accessed_date": "2026-05-18",
                "redaction": "public incident summary",
                "benchmark_eligible": True,
            },
        ]
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_seed_packet.toml: expected at least 2 public_incident publishers, found 1"
        in errors
    )


def test_frontierfail_packet_enforces_public_incident_evidence_ref_diversity(
    tmp_path: Path,
) -> None:
    _write_packet(tmp_path)
    packet_path = tmp_path / "benchmarks" / "frontierfail_seed_packet.toml"
    packet_path.write_text(
        packet_path.read_text(encoding="utf-8")
        .replace(
            "minimum_public_incident_cases = 0",
            "minimum_public_incident_cases = 2",
        )
        .replace(
            "minimum_public_incident_evidence_refs = 0",
            "minimum_public_incident_evidence_refs = 3",
        )
        .replace("minimum_cases = 2", "minimum_cases = 4"),
        encoding="utf-8",
    )
    path = tmp_path / "benchmarks" / "frontierfail_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.extend(
        [
            {
                "id": "ff-public-001",
                "source_type": "public_incident",
                "category": "fabricated_policy",
                "domain": "travel",
                "prompt": "Answer using only the airline policy.",
                "source": "The controlling policy did not allow the requested refund.",
                "bad_response": "The requested refund is allowed.",
                "expected_failure": "fabricated public refund policy",
                "expected_decision": "halt",
                "evidence_ref": "https://example.test/public-incident-1",
                "evidence_publisher": "Example Publisher One",
                "evidence_title": "Public incident one",
                "evidence_accessed_date": "2026-05-18",
                "redaction": "public incident summary",
                "benchmark_eligible": True,
            },
            {
                "id": "ff-public-002",
                "source_type": "public_incident",
                "category": "unsupported_citation",
                "domain": "legal",
                "prompt": "Cite only verified authorities.",
                "source": "The cited authorities do not exist.",
                "bad_response": "The cited authorities support the claim.",
                "expected_failure": "fabricated public citation",
                "expected_decision": "halt",
                "evidence_ref": "https://example.test/public-incident-2",
                "evidence_publisher": "Example Publisher Two",
                "evidence_title": "Public incident two",
                "evidence_accessed_date": "2026-05-18",
                "redaction": "public incident summary",
                "benchmark_eligible": True,
            },
        ]
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    errors = validate_frontierfail_packet(tmp_path)

    assert (
        "benchmarks/frontierfail_seed_packet.toml: expected at least 3 public_incident evidence refs, found 2"
        in errors
    )
