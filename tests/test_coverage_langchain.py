# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for langchain.py â€” DirectorAIGuard."""

from __future__ import annotations

import asyncio

import pytest

from director_ai.core.exceptions import HallucinationError
from director_ai.integrations.langchain import DirectorAIGuard


class TestDirectorAIGuard:
    def test_check_approved(self):
        guard = DirectorAIGuard(
            facts={"sky": "The sky is blue."},
            threshold=0.3,
            use_nli=False,
        )
        result = guard.check("sky", "The sky is blue.")
        assert result["approved"]
        assert "coherence_score" in result

    def test_check_rejected(self):
        guard = DirectorAIGuard(threshold=0.99, use_nli=False)
        result = guard.check("xyz", "totally unrelated abc")
        assert not result["approved"]

    def test_raise_on_fail(self):
        guard = DirectorAIGuard(
            threshold=0.99,
            use_nli=False,
            raise_on_fail=True,
        )
        with pytest.raises(HallucinationError):
            guard.check("xyz", "totally unrelated abc")


class TestInvoke:
    def test_invoke_dict(self):
        guard = DirectorAIGuard(threshold=0.3, use_nli=False)
        result = guard.invoke(
            {
                "query": "sky",
                "response": "The sky is blue.",
            },
        )
        assert "approved" in result

    def test_invoke_str(self):
        guard = DirectorAIGuard(threshold=0.3, use_nli=False)
        result = guard.invoke("The sky is blue.", query="sky")
        assert "approved" in result

    def test_invoke_input_output_keys(self):
        guard = DirectorAIGuard(threshold=0.3, use_nli=False)
        result = guard.invoke(
            {
                "input": "sky",
                "output": "The sky is blue.",
            },
        )
        assert "approved" in result


class TestAsync:
    def test_acheck(self):
        guard = DirectorAIGuard(threshold=0.3, use_nli=False)

        async def run():
            return await guard.acheck("sky", "The sky is blue.")

        result = asyncio.get_event_loop().run_until_complete(run())
        assert "approved" in result

    def test_ainvoke(self):
        guard = DirectorAIGuard(threshold=0.3, use_nli=False)

        async def run():
            return await guard.ainvoke({"query": "sky", "response": "blue."})

        result = asyncio.get_event_loop().run_until_complete(run())
        assert "approved" in result

    def test_ainvoke_str(self):
        guard = DirectorAIGuard(threshold=0.3, use_nli=False)

        async def run():
            return await guard.ainvoke("blue.", query="sky")

        result = asyncio.get_event_loop().run_until_complete(run())
        assert "approved" in result

    def test_acheck_raise(self):
        guard = DirectorAIGuard(
            threshold=0.99,
            use_nli=False,
            raise_on_fail=True,
        )

        async def run():
            return await guard.acheck("xyz", "totally different abc")

        with pytest.raises(HallucinationError):
            asyncio.get_event_loop().run_until_complete(run())
