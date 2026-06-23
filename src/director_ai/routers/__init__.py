# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FastAPI route group routers

"""FastAPI ``APIRouter`` groups split out of the ``create_app`` factory.

Each module here builds one cohesive group of routes. Handlers read their state
from ``request.app.state`` (config, start time, scorer, audit, …) rather than
closing over ``create_app`` locals, so the routers carry no construction-time
dependencies and ``create_app`` only has to mount them.
"""
