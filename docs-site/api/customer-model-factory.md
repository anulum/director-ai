# Customer Model Factory API

The Customer Model Factory package is organised as deterministic manifest
builders and validation gates. Each module owns one responsibility and emits
JSON-safe dataclasses with stable hashes for enterprise evidence review.

This public API page documents the open-core manifest and evidence interfaces.
Customer-specific sector packs, database-class mappings, private retrieval
schemas, tuning recipes, and customer benchmark packages are commercial
extensions and are not published as public API documentation.

::: director_ai.core.customer_model_factory.dataset_contract

::: director_ai.core.customer_model_factory.training_manifest

::: director_ai.core.customer_model_factory.benchmark_selection

::: director_ai.core.customer_model_factory.deployment_manifest

::: director_ai.core.customer_model_factory.sector_extension

::: director_ai.core.customer_model_factory.evidence_pack

::: director_ai.core.customer_model_factory.runtime_package

::: director_ai.core.customer_model_factory.monitoring_manifest

::: director_ai.core.customer_model_factory.risk_register

::: director_ai.core.customer_model_factory.release_gate
