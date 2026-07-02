#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Known mock/sys.modules test-surface classifications.

This manifest is intentionally conservative: entries stay classified as
``violation`` until the file is backed by a real production boundary test or is
proved to be an approved protocol-preserving fake.
"""

from __future__ import annotations

KNOWN_TEST_SURFACE_CLASSIFICATIONS: dict[str, tuple[str, str]] = {
    "tests/test_actor.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_actor_real_surface.py",
    ),
    "tests/test_accuracy_improvements.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_agent.py": ("violation", "module/workflow fake requiring review"),
    "tests/test_agent_providers.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_aggrefact_save_scores.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_api_reference_index.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_api_reference_cli_real_surface.py",
    ),
    "tests/test_audit_chain.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_audit_salt.py": ("violation", "private-helper bypass requiring review"),
    "tests/test_autogen_swarm.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_autopoietic.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_backends.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_backends_real_surface.py",
    ),
    "tests/test_build_judge_dataset.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_ci_gate.py": ("violation", "private-helper bypass requiring review"),
    "tests/test_cli.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_cli_core_real_surface.py",
    ),
    "tests/test_cli_bench_paths.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_cli_bench_real_surface.py",
    ),
    "tests/test_cli_bench_branch_guards.py": (
        "unit-guard-with-companion",
        "CLI/server/API typed branch guard with companion tests/test_cli_bench_real_surface.py",
    ),
    "tests/test_cli_ingest_formats.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_cli_ingest_real_surface.py",
    ),
    "tests/test_cli_new_commands.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_cli_tools_real_surface.py",
    ),
    "tests/test_cli_serve_paths.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_cli_serve_real_surface.py",
    ),
    "tests/test_cli_verify_deep.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companions tests/test_cli_verify_real_surface.py and tests/test_cli_verify_reporting_real_surface.py",
    ),
    "tests/test_competitor_aggrefact.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_config.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tests/test_config_real_surface.py",
    ),
    "tests/test_config_wizard.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tests/test_config_wizard_real_surface.py",
    ),
    "tests/test_containment.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_consumer_api.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_consumer_api_real_surface.py",
    ),
    "tests/test_contextual_compression.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_contextual_compression_real_surface.py",
    ),
    "tests/test_contradiction.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_cost_and_attribution.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_cost_integration.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_cross_language_contracts.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_cyber_physical.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_data_pipeline.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_demo_streaming_halt_live.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_device_selection.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_dialogue_fpr.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_distilled_scorer.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_doc_chunker_model_cache.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_doc_modules.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tests/test_doc_modules_real_surface.py",
    ),
    "tests/test_dp_rag.py": ("violation", "private-helper bypass requiring review"),
    "tests/test_embed_scorer.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_embed_scorer_real_surface.py",
    ),
    "tests/test_embed_scorer_real_surface.py": (
        "approved-protocol-fake",
        "real scorer/config surface with local sentence-transformers protocol fake",
    ),
    "tests/test_embedding_tuner.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_embedding_tuner_real_surface.py",
    ),
    "tests/test_evidence_packet.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_fastapi_guard.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_fastapi_guard_real_surface.py",
    ),
    "tests/test_feedback_store.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_finetune.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_finetune_api.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_finetune_api_real_surface.py",
    ),
    "tests/test_finetune_benchmark.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_finetune_benchmark_real_surface.py",
    ),
    "tests/test_finetune_gpu.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_finetune_metrics_real_surface.py": (
        "unit-guard-with-companion",
        "ML/export/eval callback guard with companion tests/test_finetune.py",
    ),
    "tests/test_formal_verification.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_frontierfail_packet.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_frontierfail_packet_real_surface.py",
    ),
    "tests/test_gemma_aggrefact_cot.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_gemma_aggrefact_eval.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_gemma_aggrefact_hiss.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_gemma_aggrefact_hiss_routed.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_gemma_aggrefact_logprob.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_gemma_aggrefact_logprob_routed.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_gemma_aggrefact_routed.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_gemma_aggrefact_self_consistency.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_grpc_server.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_grpc_server_real_surface.py",
    ),
    "tests/test_guardrails_ai_integration.py": (
        "unit-guard-with-companion",
        "external SDK adapter unit guard with companion tests/test_guardrails_ai_real_surface.py",
    ),
    "tests/test_handoff_scorer.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_hf_space_app_safety.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_hf_space_demo.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_hf_space_deployment_smoke.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_hybrid_hardening.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_hyde_backend.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_ingestion_plugins.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_injection_detector.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_injection_integration.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_injection_phase3.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_integrations_dspy.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_integrations_semantic_kernel.py": (
        "unit-guard-with-companion",
        "external SDK adapter unit guard with companion tests/test_integrations_semantic_kernel_real_surface.py",
    ),
    "tests/test_knowledge_api.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_knowledge_api_real_surface.py",
    ),
    "tests/test_knowledge_write_security.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_knowledge_write_security_real_surface.py",
    ),
    "tests/test_langgraph_integration.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_lazy_enterprise_import.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_lite_scorer_v2_eval.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_lite_scorer_v2_eval_runner.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_lite_scorer_v2_export_runner.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_lite_scorer_v2_heldout_builder.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_lite_scorer_v2_heldout_builder_real_surface.py",
    ),
    "tests/test_lite_scorer_v2_launcher.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_lite_scorer_v2_plan.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_lite_scorer_v2_plan_real_surface.py",
    ),
    "tests/test_lite_scorer_v2_record_evidence.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_lite_scorer_v2_record_evidence_real_surface.py",
    ),
    "tests/test_lite_scorer_v2_run_plan.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_lite_scorer_v2_run_plan_real_surface.py",
    ),
    "tests/test_lite_scorer_v2_training_status.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_lite_scorer_v2_training_status_real_surface.py",
    ),
    "tests/test_live_red_team.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_llm_judge.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_local_judge.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_managed_training_jobs.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_metrics.py": ("violation", "private-helper bypass requiring review"),
    "tests/test_middleware.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_middleware_real_surface.py",
    ),
    "tests/test_moderation.py": ("violation", "module/workflow fake requiring review"),
    "tests/test_multi_vector.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_multi_vector_real_surface.py",
    ),
    "tests/test_multilingual_corpus.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_multimodal_factory.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_multimodal_guard.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_nli_backend_contracts.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_nli_export.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_nli_minicheck.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_nli_scorer.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_notebook_gallery.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_onnx_backend.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_onnx_dynamic_scheduler.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_otel.py": ("violation", "module/workflow fake requiring review"),
    "tests/test_output_integrity.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_paladin_mini.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_phase3_hardening.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_phase4_hardening.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_pint_official_export_runner.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_pint_official_export_runner_real_surface.py",
    ),
    "tests/test_pint_replication_packet.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_pint_replication_packet_real_surface.py",
    ),
    "tests/test_pint_seed_smoke_runner.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_pint_seed_smoke_runner_real_surface.py",
    ),
    "tests/test_polar_deployment_smoke.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tools/validate_polar_deployment_smoke.py",
    ),
    "tests/test_prepare_threshold_data.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_production_guard.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tests/test_production_guard_real_surface.py",
    ),
    "tests/test_prompt_guard.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_proxy.py": ("violation", "private-helper bypass requiring review"),
    "tests/test_proxy_facts_path_security.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_proxy_facts_path_real_surface.py",
    ),
    "tests/test_providers.py": (
        "unit-guard-with-companion",
        "external SDK adapter unit guard with companion tests/test_providers_real_surface.py",
    ),
    "tests/test_public_endpoint_exposure_policy.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_public_endpoint_exposure_real_surface.py",
    ),
    "tests/test_python_only_contributor_path.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_python_only_contributor_path_real_surface.py",
    ),
    "tests/test_query_decomposition.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_query_decomposition_real_surface.py",
    ),
    "tests/test_recall_correctness_client.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_recall_correctness_client_real_surface.py",
    ),
    "tests/test_redis_enterprise.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_redis_enterprise_real_surface.py",
    ),
    "tests/test_report_templates.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion director_ai.compliance.report_templates",
    ),
    "tests/test_reranker.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_vector_store_reranker_real_surface.py",
    ),
    "tests/test_review_queue.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_rust_pipeline_integration.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_rust_signals.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_run_judge_benchmark.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_safety_dashboard.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_scorer_backend.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_scorer_edge_cases.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_sdk_guard.py": ("violation", "module/workflow fake requiring review"),
    "tests/test_secrets.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tests/test_secrets_real_surface.py",
    ),
    "tests/test_self_evolving.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_sentinel_judge_analyser.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_server.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_server_real_surface.py",
    ),
    "tests/test_server_auth.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_server_auth_real_surface.py",
    ),
    "tests/test_span_detector.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_streaming.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_streaming_runtime_real_surface.py",
    ),
    "tests/test_task_scoring_paths.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_tensorrt_export.py": ("violation", "ML/export/eval boundary fake"),
    "tests/test_tools/test_capability_manifest.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tools/capability_manifest.py",
    ),
    "tests/test_tools/test_emit_studio_manifest.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_tools/test_emit_studio_manifest_real_surface.py",
    ),
    "tests/test_train_distillation_reproducibility.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_trajectory_simulator.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_v320_hardening.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_v330_hardening.py": (
        "violation",
        "module/workflow fake requiring review",
    ),
    "tests/test_vector_store.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_vector_store_real_surface.py",
    ),
    "tests/test_vector_store_backends.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_vector_store_backends_real_surface.py",
    ),
    "tests/test_vector_store_embedding.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_vector_store_embedding_real_surface.py",
    ),
    "tests/test_vector_store_http_embedding.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
    "tests/test_vector_store_reranker.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_vector_store_reranker_real_surface.py",
    ),
    "tests/test_voice_adapters.py": (
        "unit-guard-with-companion",
        "external SDK adapter unit guard with companion tests/test_voice_adapters_real_surface.py",
    ),
    "tests/test_voice_adapters_sdk.py": (
        "unit-guard-with-companion",
        "external SDK adapter unit guard with companion tests/test_voice_adapters_real_surface.py",
    ),
    "tests/test_zk_attestation.py": (
        "violation",
        "private-helper bypass requiring review",
    ),
}
