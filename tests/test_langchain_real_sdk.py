# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LangChain Integration Tests with Real SDK
"""Tests that import real langchain_core and verify integration types.

These tests run only when langchain-core is installed (CI extras matrix).
They verify that DirectorAIGuard and CoherenceCallbackHandler work
correctly with real LangChain types.
"""

from __future__ import annotations


class TestDirectorAIGuardWithRealLC:
    def test_invoke_returns_dict(self):
        pytest = __import__("pytest")
        pytest.importorskip("langchain_core")
        from director_ai.integrations.langchain import DirectorAIGuard

        guard = DirectorAIGuard(
            facts={"capital": "Paris is the capital of France."},
            use_nli=False,
            raise_on_fail=False,
        )
        result = guard.invoke("Paris is the capital of France.")
        assert isinstance(result, dict)
        assert "approved" in result
        assert "score" in result

    def test_check_method(self):
        pytest = __import__("pytest")
        pytest.importorskip("langchain_core")
        from director_ai.integrations.langchain import DirectorAIGuard

        guard = DirectorAIGuard(
            facts={"policy": "Refunds within 30 days."},
            use_nli=False,
        )
        result = guard.check(
            query="What is the refund policy?",
            response="Refunds within 30 days.",
        )
        assert isinstance(result, dict)
        assert "approved" in result

    async def test_ainvoke_returns_dict(self):
        pytest = __import__("pytest")
        pytest.importorskip("langchain_core")
        from director_ai.integrations.langchain import DirectorAIGuard

        guard = DirectorAIGuard(
            facts={"capital": "Paris is the capital of France."},
            use_nli=False,
            raise_on_fail=False,
        )
        result = await guard.ainvoke("Paris is the capital of France.")
        assert isinstance(result, dict)
        assert "approved" in result


class TestCallbackHandlerWithRealLC:
    def test_inherits_real_base_callback_handler(self):
        pytest = __import__("pytest")
        pytest.importorskip("langchain_core")
        from langchain_core.callbacks import BaseCallbackHandler

        from director_ai.integrations.langchain_callback import (
            CoherenceCallbackHandler,
        )

        handler = CoherenceCallbackHandler(threshold=0.5, use_nli=False)
        assert isinstance(handler, BaseCallbackHandler)

    def test_on_llm_end_scores_response(self):
        pytest = __import__("pytest")
        pytest.importorskip("langchain_core")
        from langchain_core.outputs import Generation, LLMResult

        from director_ai.integrations.langchain_callback import (
            CoherenceCallbackHandler,
        )

        handler = CoherenceCallbackHandler(threshold=0.3, use_nli=False)
        handler.on_llm_start(serialized={}, prompts=["What is the sky color?"])

        result = LLMResult(
            generations=[[Generation(text="The sky is blue.")]],
        )
        handler.on_llm_end(result)

        assert handler.last_score is not None
        assert len(handler.scores) == 1

    def test_scores_accumulate_across_calls(self):
        pytest = __import__("pytest")
        pytest.importorskip("langchain_core")
        from langchain_core.outputs import Generation, LLMResult

        from director_ai.integrations.langchain_callback import (
            CoherenceCallbackHandler,
        )

        handler = CoherenceCallbackHandler(threshold=0.3, use_nli=False)

        for prompt, text in [
            ("sky?", "The sky is blue."),
            ("grass?", "Grass is green."),
        ]:
            handler.on_llm_start(serialized={}, prompts=[prompt])
            handler.on_llm_end(
                LLMResult(generations=[[Generation(text=text)]]),
            )

        assert len(handler.scores) == 2
