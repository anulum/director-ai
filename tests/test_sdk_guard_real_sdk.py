# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SDK Guard Tests with Real SDK Objects
"""Multi-angle tests for SDK guard with real SDK objects.

These tests run only when the SDKs are installed (CI extras matrix).
They verify that guard() correctly detects SDK shapes and installs
proxy objects on real client instances.
"""

from __future__ import annotations


class TestGuardWithRealOpenAI:
    def test_guard_detects_openai_shape(self):
        openai = __import__("pytest").importorskip("openai")
        from director_ai.integrations.sdk_guard import guard

        client = openai.OpenAI(api_key="sk-test-fake")
        original_completions_type = type(client.chat.completions)

        guarded = guard(client, facts={"test": "value"}, use_nli=False)

        assert guarded is client
        assert type(guarded.chat.completions) is not original_completions_type
        assert hasattr(guarded.chat.completions, "create")
        assert hasattr(guarded.chat.completions, "_original")

    def test_guard_preserves_client_attributes(self):
        openai = __import__("pytest").importorskip("openai")
        from director_ai.integrations.sdk_guard import guard

        client = openai.OpenAI(api_key="sk-test-fake")
        guarded = guard(client, use_nli=False)

        assert hasattr(guarded, "models")
        assert hasattr(guarded, "embeddings")
        assert hasattr(guarded, "chat")

    def test_guard_on_fail_modes(self):
        openai = __import__("pytest").importorskip("openai")
        from director_ai.integrations.sdk_guard import guard

        for mode in ("raise", "log", "metadata"):
            client = openai.OpenAI(api_key="sk-test-fake")
            guarded = guard(client, use_nli=False, on_fail=mode)
            assert guarded is client

    def test_guard_with_facts(self):
        openai = __import__("pytest").importorskip("openai")
        from director_ai.integrations.sdk_guard import guard

        client = openai.OpenAI(api_key="sk-test-fake")
        facts = {"earth": "Earth orbits the Sun", "sky": "The sky is blue"}
        guarded = guard(client, facts=facts, use_nli=False)
        assert guarded is client

    def test_guard_idempotent(self):
        openai = __import__("pytest").importorskip("openai")
        from director_ai.integrations.sdk_guard import guard

        client = openai.OpenAI(api_key="sk-test-fake")
        g1 = guard(client, use_nli=False)
        g2 = guard(g1, use_nli=False)
        assert g2 is client


class TestGuardWithRealAnthropic:
    def test_guard_detects_anthropic_shape(self):
        anthropic = __import__("pytest").importorskip("anthropic")
        from director_ai.integrations.sdk_guard import guard

        client = anthropic.Anthropic(api_key="sk-ant-test-fake")
        original_messages_type = type(client.messages)

        guarded = guard(client, facts={"test": "value"}, use_nli=False)

        assert guarded is client
        assert type(guarded.messages) is not original_messages_type
        assert hasattr(guarded.messages, "create")
        assert hasattr(guarded.messages, "_original")

    def test_guard_preserves_anthropic_attributes(self):
        anthropic = __import__("pytest").importorskip("anthropic")
        from director_ai.integrations.sdk_guard import guard

        client = anthropic.Anthropic(api_key="sk-ant-test-fake")
        guarded = guard(client, use_nli=False)

        assert guarded is client
        assert hasattr(guarded, "messages")

    def test_guard_anthropic_on_fail_modes(self):
        anthropic = __import__("pytest").importorskip("anthropic")
        from director_ai.integrations.sdk_guard import guard

        for mode in ("raise", "log", "metadata"):
            client = anthropic.Anthropic(api_key="sk-ant-test-fake")
            guarded = guard(client, use_nli=False, on_fail=mode)
            assert guarded is client


class TestGuardPipelinePerformance:
    """Document SDK guard pipeline characteristics."""

    def test_guard_module_importable(self):
        from director_ai.integrations.sdk_guard import guard

        assert callable(guard)

    def test_guard_returns_same_client(self):
        openai = __import__("pytest").importorskip("openai")
        from director_ai.integrations.sdk_guard import guard

        client = openai.OpenAI(api_key="sk-test-fake")
        guarded = guard(client, use_nli=False)
        assert guarded is client  # guard wraps in-place, returns same object
