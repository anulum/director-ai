# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Benchmark Package

"""Expose evaluator modules and the batch runner for module-based invocation.

Importing this package loads no model or dataset. Run an individual evaluator
with ``python -m benchmarks.<module>`` or use ``benchmarks.run_all`` for the
batch entry point. Evaluation split and provenance must be established by the
individual runner and its recorded inputs before interpreting the results.
"""

# Director-AI — Public Benchmarks
#
# Held-out eval suite:
#   python -m benchmarks.mnli_eval [N]           -- NLI regression
#   python -m benchmarks.anli_eval [N]           -- Adversarial NLI (R1/R2/R3)
#   python -m benchmarks.fever_eval [N]          -- Fact verification
#   python -m benchmarks.vitaminc_eval [N]       -- Contrastive fact-check
#   python -m benchmarks.paws_eval [N]           -- Adversarial paraphrase
#   python -m benchmarks.falsepositive_eval [N]  -- Clean RAG FP rate
#   python -m benchmarks.aggrefact_eval [N]      -- LLM-AggreFact (requires HF_TOKEN)
#   python -m benchmarks.hallubench_eval         -- HalluBench gated VLM eval
#   python -m benchmarks.run_all --max-samples N -- Full suite + comparison table
#
# In-training eval (data leakage — report separately):
#   python -m benchmarks.halueval_eval [N]
#   python -m benchmarks.truthfulqa_eval [N]
