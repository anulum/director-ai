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
        "unit-guard-with-companion",
        "accuracy-routing unit guard with companion tests/test_accuracy_improvements_real_surface.py",
    ),
    "tests/test_agent.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_agent_real_surface.py",
    ),
    "tests/test_agent_providers.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_agent_providers_real_surface.py",
    ),
    "tests/test_aggrefact_save_scores.py": (
        "unit-guard-with-companion",
        "AggreFact score-cache unit guard with companion tests/test_aggrefact_save_scores_real_surface.py",
    ),
    "tests/test_api_reference_index.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_api_reference_cli_real_surface.py",
    ),
    "tests/test_audit_chain.py": (
        "unit-guard-with-companion",
        "private-helper unit guard with companion tests/test_audit_chain_real_surface.py",
    ),
    "tests/test_audit_salt.py": (
        "unit-guard-with-companion",
        "private-helper unit guard with companion tests/test_audit_salt_real_surface.py",
    ),
    "tests/test_autogen_swarm.py": (
        "unit-guard-with-companion",
        "AutoGen swarm hook unit guard with companion tests/test_autogen_swarm_real_surface.py",
    ),
    "tests/test_autopoietic.py": (
        "unit-guard-with-companion",
        "autopoietic unit guard with companion tests/test_autopoietic_real_surface.py",
    ),
    "tests/test_backends.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_backends_real_surface.py",
    ),
    "tests/test_build_judge_dataset.py": (
        "unit-guard-with-companion",
        "judge-dataset builder unit guard with companion tests/test_build_judge_dataset_real_surface.py",
    ),
    "tests/test_ci_gate.py": (
        "unit-guard-with-companion",
        "private-helper unit guard with companion tests/test_ci_gate_real_surface.py",
    ),
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
        "unit-guard-with-companion",
        "competitor AggreFact unit guard with companion tests/test_competitor_aggrefact_real_surface.py",
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
        "unit-guard-with-companion",
        "safety/containment unit guard with companion tests/test_containment_real_surface.py",
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
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_contradiction_real_surface.py",
    ),
    "tests/test_cost_and_attribution.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_cost_and_attribution_real_surface.py",
    ),
    "tests/test_cost_integration.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_cost_integration_real_surface.py",
    ),
    "tests/test_cross_language_contracts.py": (
        "unit-guard-with-companion",
        "cross-language property guard with companion tests/test_cross_language_contracts_real_surface.py",
    ),
    "tests/test_cyber_physical.py": (
        "unit-guard-with-companion",
        "cyber-physical unit guard with companion tests/test_cyber_physical_real_surface.py",
    ),
    "tests/test_cyber_physical_real_surface.py": (
        "approved-protocol-fake",
        "real cyber-physical facade surface with local ROS2 runtime protocol module",
    ),
    "tests/test_data_pipeline.py": (
        "unit-guard-with-companion",
        "training data pipeline unit guard with companion tests/test_data_pipeline_real_surface.py",
    ),
    "tests/test_demo_streaming_halt_live.py": (
        "unit-guard-with-companion",
        "live streaming halt demo unit guard with companion tests/test_demo_streaming_halt_live_real_surface.py",
    ),
    "tests/test_device_selection.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_device_selection_real_surface.py",
    ),
    "tests/test_device_selection_real_surface.py": (
        "approved-protocol-fake",
        "real subprocess device-selection surface with local torch protocol package",
    ),
    "tests/test_dialogue_fpr.py": (
        "unit-guard-with-companion",
        "scoring/dialogue unit guard with companion tests/test_dialogue_fpr_real_surface.py",
    ),
    "tests/test_distilled_scorer.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_distilled_scorer_real_surface.py",
    ),
    "tests/test_doc_chunker_model_cache.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_doc_chunker_model_cache_real_surface.py",
    ),
    "tests/test_doc_chunker_model_cache_real_surface.py": (
        "approved-protocol-fake",
        "real semantic chunking surface with local sentence-transformers protocol fake",
    ),
    "tests/test_doc_modules.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tests/test_doc_modules_real_surface.py",
    ),
    "tests/test_dp_rag.py": (
        "unit-guard-with-companion",
        "privacy/dp-rag unit guard with companion tests/test_dp_rag_real_surface.py",
    ),
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
        "unit-guard-with-companion",
        "private-helper unit guard with companion tests/test_evidence_packet_real_surface.py",
    ),
    "tests/test_fastapi_guard.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_fastapi_guard_real_surface.py",
    ),
    "tests/test_feedback_store.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_feedback_store_real_surface.py",
    ),
    "tests/test_finetune.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_finetune_real_surface.py",
    ),
    "tests/test_finetune_api.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_finetune_api_real_surface.py",
    ),
    "tests/test_finetune_benchmark.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_finetune_benchmark_real_surface.py",
    ),
    "tests/test_finetune_gpu.py": (
        "unit-guard-with-companion",
        "GPU fine-tune unit guard with companions tests/test_finetune_real_surface.py, tests/test_finetune_api_real_surface.py, and tests/test_finetune_benchmark_real_surface.py",
    ),
    "tests/test_finetune_metrics_real_surface.py": (
        "unit-guard-with-companion",
        "ML/export/eval callback guard with companion tests/test_finetune.py",
    ),
    "tests/test_formal_verification.py": (
        "unit-guard-with-companion",
        "formal verification unit guard with companion tests/test_formal_verification_real_surface.py",
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
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_handoff_scorer_real_surface.py",
    ),
    "tests/test_hf_space_app_safety.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_hf_space_app_safety_real_surface.py",
    ),
    "tests/test_hf_space_demo.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_hf_space_demo_real_surface.py",
    ),
    "tests/test_hf_space_deployment_smoke.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_hf_space_deployment_smoke_real_surface.py",
    ),
    "tests/test_hybrid_hardening.py": (
        "unit-guard-with-companion",
        "hybrid LLM-judge unit guard with companion tests/test_hybrid_hardening_real_surface.py",
    ),
    "tests/test_hybrid_hardening_real_surface.py": (
        "approved-protocol-fake",
        "real LLM-judge scorer surface with local OpenAI protocol fake",
    ),
    "tests/test_hyde_backend.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_hyde_backend_real_surface.py",
    ),
    "tests/test_ingestion_plugins.py": (
        "unit-guard-with-companion",
        "ingestion plugin unit guard with companion tests/test_ingestion_plugins_real_surface.py",
    ),
    "tests/test_injection_detector.py": (
        "unit-guard-with-companion",
        "safety/injection unit guard with companion tests/test_injection_detector_real_surface.py",
    ),
    "tests/test_injection_integration.py": (
        "unit-guard-with-companion",
        "injection integration unit guard with companion tests/test_injection_integration_real_surface.py",
    ),
    "tests/test_injection_phase3.py": (
        "unit-guard-with-companion",
        "injection Phase 3 unit guard with companions tests/test_injection_integration_real_surface.py, tests/test_fastapi_guard_real_surface.py, and tests/test_sdk_guard_real_surface.py",
    ),
    "tests/test_integrations_dspy.py": (
        "unit-guard-with-companion",
        "external SDK adapter unit guard with companion tests/test_integrations_dspy_real_surface.py",
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
        "unit-guard-with-companion",
        "external SDK adapter unit guard with companion tests/test_langgraph_integration_real_surface.py",
    ),
    "tests/test_lazy_enterprise_import.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_lazy_enterprise_import_real_surface.py",
    ),
    "tests/test_lite_scorer_v2_eval.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_lite_scorer_v2_eval_real_surface.py",
    ),
    "tests/test_lite_scorer_v2_eval_real_surface.py": (
        "approved-protocol-fake",
        "real evaluator CLI surface with local onnxruntime, transformers, and backfire-kernel protocol fakes",
    ),
    "tests/test_lite_scorer_v2_eval_runner.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_lite_scorer_v2_eval_runner_real_surface.py",
    ),
    "tests/test_lite_scorer_v2_export_runner.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_lite_scorer_v2_export_runner_real_surface.py",
    ),
    "tests/test_lite_scorer_v2_heldout_builder.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_lite_scorer_v2_heldout_builder_real_surface.py",
    ),
    "tests/test_lite_scorer_v2_launcher.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_lite_scorer_v2_launcher_real_surface.py",
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
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_live_red_team_real_surface.py",
    ),
    "tests/test_llm_judge.py": (
        "unit-guard-with-companion",
        "ML/export/eval LLM judge guard with companion tests/test_hybrid_hardening_real_surface.py",
    ),
    "tests/test_local_judge.py": (
        "unit-guard-with-companion",
        "ML/export/eval local judge guard with companions tests/test_hybrid_hardening_real_surface.py and tests/test_config_real_surface.py",
    ),
    "tests/test_managed_training_jobs.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_metrics.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_metrics_real_surface.py",
    ),
    "tests/test_middleware.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_middleware_real_surface.py",
    ),
    "tests/test_moderation.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_moderation_real_surface.py",
    ),
    "tests/test_multi_vector.py": (
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_multi_vector_real_surface.py",
    ),
    "tests/test_multilingual_corpus.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion tests/test_multilingual_corpus_real_surface.py",
    ),
    "tests/test_multimodal_factory.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_multimodal_factory_real_surface.py",
    ),
    "tests/test_multimodal_guard.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_multimodal_guard_real_surface.py",
    ),
    "tests/test_nli_backend_contracts.py": (
        "violation",
        "ML/export/eval boundary fake",
    ),
    "tests/test_nli_export.py": (
        "unit-guard-with-companion",
        "ML/export/eval NLI export guard with companion "
        "tests/test_nli_export_real_surface.py",
    ),
    "tests/test_nli_export_real_surface.py": (
        "approved-protocol-fake",
        "real public NLI export surface with local torch/transformers protocol modules",
    ),
    "tests/test_nli_minicheck.py": (
        "unit-guard-with-companion",
        "ML/export/eval MiniCheck NLI guard with companion "
        "tests/test_nli_scorer_real_surface.py",
    ),
    "tests/test_nli_scorer.py": (
        "unit-guard-with-companion",
        "ML/export/eval NLI scorer guard with companion tests/test_nli_scorer_real_surface.py",
    ),
    "tests/test_notebook_gallery.py": (
        "unit-guard-with-companion",
        "ML/export/eval notebook gallery guard with companion "
        "tests/test_notebook_gallery_real_surface.py",
    ),
    "tests/test_onnx_backend.py": (
        "unit-guard-with-companion",
        "ML/export/eval ONNX backend guard with companion tests/test_onnx_backend_real_surface.py",
    ),
    "tests/test_onnx_dynamic_scheduler.py": (
        "unit-guard-with-companion",
        "ML/export/eval ONNX scheduler guard with companion "
        "tests/test_onnx_dynamic_scheduler_real_surface.py",
    ),
    "tests/test_otel.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_otel_real_surface.py",
    ),
    "tests/test_output_integrity.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_output_integrity_real_surface.py",
    ),
    "tests/test_paladin_mini.py": (
        "unit-guard-with-companion",
        "ML/export/eval Paladin-mini guard with companion "
        "tests/test_paladin_mini_real_surface.py",
    ),
    "tests/test_phase3_hardening.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companions tests/test_agent_real_surface.py, tests/test_actor_real_surface.py, tests/test_config_real_surface.py, and tests/test_cli_serve_real_surface.py",
    ),
    "tests/test_phase4_hardening.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companions tests/test_actor_real_surface.py, tests/test_cli_core_real_surface.py, tests/test_config_real_surface.py, and tests/test_server_real_surface.py",
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
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_prepare_threshold_data_real_surface.py",
    ),
    "tests/test_production_guard.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tests/test_production_guard_real_surface.py",
    ),
    "tests/test_prompt_guard.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_server_prompt_guard.py",
    ),
    "tests/test_proxy.py": (
        "unit-guard-with-companion",
        "CLI/server/API unit guard with companion tests/test_proxy_real_surface.py",
    ),
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
        "unit-guard-with-companion",
        "private-helper unit guard with companion tests/test_review_queue_real_surface.py",
    ),
    "tests/test_rust_pipeline_integration.py": (
        "unit-guard-with-companion",
        "private-helper unit guard with companions "
        "tests/test_backends_real_surface.py, "
        "tests/test_production_guard_real_surface.py, and "
        "tests/test_streaming_runtime_real_surface.py",
    ),
    "tests/test_rust_signals.py": (
        "unit-guard-with-companion",
        "private-helper unit guard with companions "
        "tests/test_production_guard_real_surface.py, "
        "tests/test_streaming_runtime_real_surface.py, and "
        "tests/test_vector_store_real_surface.py",
    ),
    "tests/test_run_judge_benchmark.py": (
        "unit-guard-with-companion",
        "ML/export/eval benchmark guard with companion "
        "tests/test_run_judge_benchmark_real_surface.py",
    ),
    "tests/test_safety_dashboard.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_safety_dashboard_real_surface.py",
    ),
    "tests/test_scorer_backend.py": (
        "unit-guard-with-companion",
        "scorer backend dispatch unit guard with companion tests/test_backends_real_surface.py",
    ),
    "tests/test_scorer_edge_cases.py": (
        "unit-guard-with-companion",
        "scorer edge-case unit guard with companion tests/test_config_real_surface.py",
    ),
    "tests/test_sdk_guard.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_sdk_guard_real_surface.py",
    ),
    "tests/test_secrets.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tests/test_secrets_real_surface.py",
    ),
    "tests/test_self_evolving.py": (
        "unit-guard-with-companion",
        "self-evolving unit guard with companion tests/test_self_evolving_real_surface.py",
    ),
    "tests/test_sentinel_judge_analyser.py": (
        "unit-guard-with-companion",
        "ML/export/eval Sentinel-Judge analyser guard with companion "
        "tests/test_sentinel_judge_analyser_real_surface.py",
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
        "unit-guard-with-companion",
        "span detector unit guard with companion tests/test_span_detector_real_surface.py",
    ),
    "tests/test_streaming.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_streaming_runtime_real_surface.py",
    ),
    "tests/test_task_scoring_paths.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_task_scoring_paths_real_surface.py",
    ),
    "tests/test_tensorrt_export.py": (
        "unit-guard-with-companion",
        "ML/export/eval TensorRT export guard with companion "
        "tests/test_tensorrt_export_real_surface.py",
    ),
    "tests/test_tools/test_capability_manifest.py": (
        "unit-guard-with-companion",
        "generated/docs/config guard with companion tools/capability_manifest.py",
    ),
    "tests/test_tools/test_emit_studio_manifest.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companion tests/test_tools/test_emit_studio_manifest_real_surface.py",
    ),
    "tests/test_train_distillation_reproducibility.py": (
        "unit-guard-with-companion",
        "ML/export/eval unit guard with companion "
        "tests/test_train_distillation_reproducibility_real_surface.py",
    ),
    "tests/test_trajectory_simulator.py": (
        "unit-guard-with-companion",
        "trajectory/preflight unit guard with companion tests/test_trajectory_simulator_real_surface.py",
    ),
    "tests/test_v320_hardening.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companions "
        "tests/test_agent_real_surface.py, tests/test_cli_core_real_surface.py, "
        "tests/test_config_real_surface.py, and "
        "tests/test_vector_store_real_surface.py",
    ),
    "tests/test_v330_hardening.py": (
        "unit-guard-with-companion",
        "module/workflow unit guard with companions "
        "tests/test_agent_real_surface.py, tests/test_cli_ingest_real_surface.py, "
        "tests/test_config_real_surface.py, tests/test_consumer_api_real_surface.py, "
        "tests/test_grpc_server_real_surface.py, and "
        "tests/test_server_real_surface.py",
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
        "unit-guard-with-companion",
        "storage/retrieval unit guard with companion tests/test_vector_store_http_embedding_real_surface.py",
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
        "unit-guard-with-companion",
        "attestation/passport unit guard with companion tests/test_zk_attestation_real_surface.py",
    ),
}
