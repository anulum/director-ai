# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — __init__
# Director-Class AI — Public Benchmarks
#
# Held-out eval suite:
#   python -m benchmarks.mnli_eval [N]           -- NLI regression
#   python -m benchmarks.anli_eval [N]           -- Adversarial NLI (R1/R2/R3)
#   python -m benchmarks.fever_eval [N]          -- Fact verification
#   python -m benchmarks.vitaminc_eval [N]       -- Contrastive fact-check
#   python -m benchmarks.paws_eval [N]           -- Adversarial paraphrase
#   python -m benchmarks.falsepositive_eval [N]  -- Clean RAG FP rate
#   python -m benchmarks.aggrefact_eval [N]      -- LLM-AggreFact (requires HF_TOKEN)
#   python -m benchmarks.run_all --max-samples N -- Full suite + comparison table
#
# In-training eval (data leakage — report separately):
#   python -m benchmarks.halueval_eval [N]
#   python -m benchmarks.truthfulqa_eval [N]
