# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Re-export facade for the domain-grouped CLI verification commands."""

from __future__ import annotations

from .cli_verify.compliance import _cmd_compliance as _cmd_compliance
from .cli_verify.compliance import _cmd_cost_report as _cmd_cost_report
from .cli_verify.compliance import _cmd_forensics as _cmd_forensics
from .cli_verify.compliance import _cmd_kpis as _cmd_kpis
from .cli_verify.compliance import (
    _forensics_records_from_payload as _forensics_records_from_payload,
)
from .cli_verify.compliance import _kpi_items_from_records as _kpi_items_from_records
from .cli_verify.compliance import _load_article15_context as _load_article15_context
from .cli_verify.diagnostics import _check_optional_module as _check_optional_module
from .cli_verify.diagnostics import _cmd_doctor as _cmd_doctor
from .cli_verify.diagnostics import _cmd_license as _cmd_license
from .cli_verify.diagnostics import _stack_status as _stack_status
from .cli_verify.diagnostics import _stack_warnings as _stack_warnings
from .cli_verify.tools import _cmd_kb_health as _cmd_kb_health
from .cli_verify.tools import _cmd_safety_dashboard as _cmd_safety_dashboard
from .cli_verify.tools import _cmd_wizard as _cmd_wizard
from .cli_verify.verification import _cmd_adversarial_test as _cmd_adversarial_test
from .cli_verify.verification import _cmd_check_step as _cmd_check_step
from .cli_verify.verification import _cmd_consensus as _cmd_consensus
from .cli_verify.verification import _cmd_temporal_freshness as _cmd_temporal_freshness
from .cli_verify.verification import _cmd_verify_numeric as _cmd_verify_numeric
from .cli_verify.verification import _cmd_verify_reasoning as _cmd_verify_reasoning
