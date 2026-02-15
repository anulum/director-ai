# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Package Initialisation
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from .director_module import DirectorModule
from .actor_module import MockActor
from .backfire_kernel import BackfireKernel
from .knowledge_base import KnowledgeBase
from .strange_loop_agent import StrangeLoopAgent

__all__ = [
    "DirectorModule",
    "MockActor",
    "BackfireKernel",
    "KnowledgeBase",
    "StrangeLoopAgent",
]

__version__ = "0.2.0"
