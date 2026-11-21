# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Validator Facade

"""Re-export of every validator, digest function and check of the kernel.

Consumers import from :mod:`director_ai.model_lifecycle`; this facade only
groups the functions for the package itself.
"""

from __future__ import annotations

from .base_checkpoints import validate_base_checkpoint_record
from .cost_and_promotion_contracts import (
    check_promotion_pre_state,
    validate_promotion_decision,
)
from .custody_records import is_sealed_kind, validate_custody_ingest_record
from .digests import (
    canonical_json_bytes,
    decode_record,
    record_digest,
    strict_json_loads,
)
from .envelopes import require_active_family, validate_envelope, validate_lineage_edge
from .evaluation_contracts import validate_preregistration, validate_sealed_split_pool
from .evaluator_outputs import (
    validate_evaluator_output,
    validate_look_result,
    validate_slice_estimate,
)
from .float_scan import any_lane_matches, scan_float_chunk
from .graders import validate_grader
from .licences import (
    require_output_training_permitted,
    validate_compute_budget,
    validate_licence_record,
)
from .lineage import derive_proposer_lineage, exempt_base_digests
from .look_admission import admit_sealed_look
from .look_budget import check_look_budget
from .power import required_paired_sample_size
from .preregistration_approvals import validate_preregistration_approval
from .record_codec import from_canonical_json_bytes
from .safetensors_parser import (
    compute_tensor_set_digest,
    compute_tensor_set_identity,
    read_safetensors_descriptors,
)
from .sealed_looks import validate_journal_entry, validate_look_request
from .tensors import tensor_set_identity
from .weight_archives import compute_archive_tensor_set_identity, list_archive_members

__all__ = [
    "any_lane_matches",
    "from_canonical_json_bytes",
    "scan_float_chunk",
    "admit_sealed_look",
    "canonical_json_bytes",
    "check_look_budget",
    "check_promotion_pre_state",
    "compute_archive_tensor_set_identity",
    "compute_tensor_set_digest",
    "compute_tensor_set_identity",
    "decode_record",
    "derive_proposer_lineage",
    "exempt_base_digests",
    "is_sealed_kind",
    "list_archive_members",
    "read_safetensors_descriptors",
    "record_digest",
    "require_active_family",
    "require_output_training_permitted",
    "required_paired_sample_size",
    "strict_json_loads",
    "tensor_set_identity",
    "validate_base_checkpoint_record",
    "validate_compute_budget",
    "validate_custody_ingest_record",
    "validate_envelope",
    "validate_evaluator_output",
    "validate_grader",
    "validate_journal_entry",
    "validate_licence_record",
    "validate_lineage_edge",
    "validate_look_request",
    "validate_look_result",
    "validate_preregistration",
    "validate_preregistration_approval",
    "validate_promotion_decision",
    "validate_sealed_split_pool",
    "validate_slice_estimate",
]
