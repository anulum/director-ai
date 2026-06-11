<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Document Ingestion Pipeline

`DocumentIngestionPipeline` is the reusable Python API behind document
ingestion workflows. It parses bytes, chunks text, writes chunks into a
`VectorGroundTruthStore`, and keeps a `DocRegistry` in sync for update and
delete operations.

```python
from director_ai.core.ingestion import DocumentIngestionPipeline
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

store = VectorGroundTruthStore()
pipeline = DocumentIngestionPipeline(store=store)

result = pipeline.ingest_bytes(
    b"Refund policy: 30 days.",
    filename="policy.txt",
    doc_id="refund-policy",
    source="policy.txt",
    tenant_id="acme",
)

print(result.chunk_ids)
```

Use `update_text()` for replacement sync. If the content hash is unchanged, the
pipeline returns `unchanged=True` and does not re-embed. If content changes, new
chunks are staged before old chunks are removed, so a failed replacement does
not silently orphan the document.

Use `delete()` to remove both registry metadata and vector-store chunks.

::: director_ai.core.ingestion.IngestionConfig

::: director_ai.core.ingestion.IngestionResult

::: director_ai.core.ingestion.DeletedDocument

::: director_ai.core.ingestion.DocumentIngestionPipeline
