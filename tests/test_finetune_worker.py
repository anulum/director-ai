# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — fine-tuning worker contract tests

"""Contract tests for the local fine-tuning worker module.

The worker's behaviour (success, failure, cleanup) is exercised through
the historical ``finetune_api`` surface in ``test_finetune_api.py``; this
file pins the module boundary introduced by the WCB-7 decomposition.
"""

from __future__ import annotations

import director_ai._finetune_worker as worker_module
import director_ai.finetune_api as finetune_api_module


def test_facade_reexports_the_worker():
    assert (
        finetune_api_module._run_training_worker is worker_module._run_training_worker
    )


def test_worker_module_owns_only_the_worker():
    assert worker_module.__all__ == ["_run_training_worker"]


def test_worker_logs_to_the_finetune_api_logger():
    assert worker_module.logger.name == "DirectorAI.FinetuneAPI"
