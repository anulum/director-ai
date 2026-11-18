# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — invocation-local grounding for native scoring callbacks
"""Bind tenant context to the native scorer's synchronous retrieval callback."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class NativeGrounding:
    """Adapt a real grounding retriever to the native one-query callback.

    Parameters
    ----------
    retrieve : callable or None
        Bound store retrieval method. With no store, every query returns None
        so the Python no-KB contract does not acquire native demo facts.
    """

    def __init__(self, retrieve: Callable[..., object] | None) -> None:
        """Keep the configured retriever and an independent execution context."""
        self._retrieve = retrieve
        self._tenant = ContextVar[str](
            "director_ai_native_grounding_tenant", default=""
        )

    def __call__(self, query: str) -> str | None:
        """Retrieve context for the active native scoring invocation.

        Parameters
        ----------
        query : str
            Actual query supplied by the native scorer.

        Returns
        -------
        str or None
            Retrieved text, or missing context. Other return types retain the
            native callback's existing missing-context interpretation.
        """
        retrieve = self._retrieve
        if retrieve is None:
            return None
        tenant = self._tenant.get()
        value = retrieve(query, tenant_id=tenant) if tenant else retrieve(query)
        return value if isinstance(value, str) else None

    @contextmanager
    def scoped(self, tenant_id: str) -> Iterator[None]:
        """Apply tenant identity until this synchronous native call exits.

        Parameters
        ----------
        tenant_id : str
            Caller-supplied tenant, with an empty identity preserving unscoped
            retrieval. Nested calls, threads and tasks retain separate values.

        Yields
        ------
        None
            The caller executes the real native operation inside this scope.
            The previous value is restored after success or an exception.
        """
        token = self._tenant.set(tenant_id)
        try:
            yield
        finally:
            self._tenant.reset(token)
