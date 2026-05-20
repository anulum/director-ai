# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory evidence pack exporter

"""Export a Customer Model Factory evidence-pack manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from director_ai.core.customer_model_factory.deployment_manifest import (
    CustomerDeploymentManifest,
)
from director_ai.core.customer_model_factory.evidence_pack import (
    build_customer_evidence_pack,
)
from director_ai.core.customer_model_factory.sector_extension import (
    SectorEvidenceMapping,
)


def main(argv: list[str] | None = None) -> int:
    """Run the evidence-pack exporter."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument(
        "--sector-evidence-mapping",
        "--regulation-mapping",
        dest="sector_evidence_mapping",
        type=Path,
        required=True,
    )
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--export-uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-external-callbacks", action="store_true")
    parser.add_argument("--callback-endpoint", action="append", default=[])
    args = parser.parse_args(argv)

    deployment = CustomerDeploymentManifest.from_dict(
        json.loads(args.deployment_manifest.read_text(encoding="utf-8"))
    )
    mapping = SectorEvidenceMapping.from_dict(
        json.loads(args.sector_evidence_mapping.read_text(encoding="utf-8"))
    )
    manifest = build_customer_evidence_pack(
        package_id=args.package_id,
        deployment_manifest=deployment,
        regulation_mapping=mapping,
        classification=args.classification,
        export_uri=args.export_uri,
        external_callbacks_allowed=args.allow_external_callbacks,
        callback_endpoints=tuple(args.callback_endpoint),
    )
    manifest.write_json(args.output)
    return 0 if manifest.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
