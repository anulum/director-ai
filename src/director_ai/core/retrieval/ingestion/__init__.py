# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ingestion plugins package

"""Auto-ingestion plugins for external knowledge sources.

Each plugin pulls documents from a source (S3 bucket, Notion
workspace, Google Drive folder), produces a stream of
:class:`IngestedDocument` records, and hands them to
:class:`director_ai.core.retrieval.knowledge.GroundTruthStore`.

Design notes:

* **Clients are injected** — the adapters accept an already-built
  ``boto3`` / ``notion-client`` / ``googleapiclient`` instance. Tests
  substitute stubs with the same surface so no real cloud call is
  needed in the test suite.
* **Creds never touch this package** — operators read from the vault
  / their own secret manager and pass the configured client.
* **No side effects beyond ``store.add``** — the adapters do not
  mutate the source, do not delete documents, do not change ACLs.
* **Tenant-aware** — ``tenant_id`` flows through to
  ``GroundTruthStore.add``.

Optional extras:

* ``[ingestion-s3]`` — ``boto3``
* ``[ingestion-notion]`` — ``notion-client``
* ``[ingestion-gdrive]`` — ``google-api-python-client``
* ``[ingestion]`` — all three
"""

from .base import IngestedDocument, IngestionPlugin
from .gdrive import GoogleDrivePlugin
from .notion import NotionPlugin
from .s3 import S3Plugin

__all__ = [
    "GoogleDrivePlugin",
    "IngestedDocument",
    "IngestionPlugin",
    "NotionPlugin",
    "S3Plugin",
]
