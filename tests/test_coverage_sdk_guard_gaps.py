# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from director_ai.core.exceptions import HallucinationError
from director_ai.core.scorer import CoherenceScorer
from director_ai.core.types import CoherenceScore
from director_ai.integrations.sdk_guard import (
    STREAM_CHECK_INTERVAL,
    _ascore_and_gate,
    _BedrockProxy,
    _CohereProxy,
    _extract_bedrock_prompt,
    _extract_bedrock_stream_delta,
    _extract_gemini_prompt,
    _GeminiProxy,
    _GuardedAnthropicStream,
    _GuardedBedrockStream,
    _GuardedCohereStream,
    _GuardedGeminiStream,
    _has_bedrock_shape,
    _has_cohere_shape,
    _has_gemini_shape,
    _score_and_gate,
    get_score,
    guard,
    score,
)


def _passing_scorer():
    s = MagicMock(spec=CoherenceScorer)
    cs = CoherenceScore(
        score=0.9, approved=True, h_logical=0.1, h_factual=0.1, warning=False
    )
    s.review.return_value = (True, cs)
    return s, cs


def _failing_scorer():
    s = MagicMock(spec=CoherenceScorer)
    cs = CoherenceScore(
        score=0.2, approved=False, h_logical=0.8, h_factual=0.8, warning=True
    )
    s.review.return_value = (False, cs)
    return s, cs


# ── score() with profile + facts (lines 60-63) ───────────────────────────────


class TestScoreProfileWithFacts:
    def test_profile_with_facts_populates_store(self):
        cs = score(
            "What is the refund policy?",
            "Refunds within 30 days.",
            facts={"refund": "Refunds within 30 days only."},
            profile="fast",
        )
        assert isinstance(cs, CoherenceScore)

    def test_profile_no_facts_uses_empty_store(self):
        cs = score(
            "Hello",
            "Hi there!",
            profile="fast",
        )
        assert isinstance(cs, CoherenceScore)


# ── _score_and_gate async coroutine path (lines 189-199) ──────────────────────


class TestScoreAndGateAsyncCoroutine:
    def test_coroutine_result_no_running_loop(self):
        async def _async_review(prompt, response, **_kw):
            cs = CoherenceScore(
                score=0.9, approved=True, h_logical=0.1, h_factual=0.1, warning=False
            )
            return True, cs

        scorer = MagicMock()
        scorer.review.side_effect = lambda p, r, **kw: _async_review(p, r)

        cs = _score_and_gate(scorer, "log", "q", "good response")
        assert cs.score == pytest.approx(0.9)

    def test_coroutine_result_with_running_loop(self):
        async def _async_review(prompt, response, **_kw):
            cs = CoherenceScore(
                score=0.85, approved=True, h_logical=0.1, h_factual=0.1, warning=False
            )
            return True, cs

        scorer = MagicMock()
        scorer.review.side_effect = lambda p, r, **kw: _async_review(p, r)

        result_holder = []

        async def _run():
            cs = _score_and_gate(scorer, "log", "q", "good response")
            result_holder.append(cs)

        asyncio.run(_run())
        assert result_holder[0].score == pytest.approx(0.85)

    def test_coroutine_metadata_on_fail(self):
        async def _async_review(prompt, response, **_kw):
            cs = CoherenceScore(
                score=0.9, approved=True, h_logical=0.1, h_factual=0.1, warning=False
            )
            return True, cs

        scorer = MagicMock()
        scorer.review.side_effect = lambda p, r, **kw: _async_review(p, r)

        _score_and_gate(scorer, "metadata", "q", "good response")
        assert get_score() is not None


# ── _ascore_and_gate (lines 212, 216) ────────────────────────────────────────


class TestAscoreAndGate:
    def test_coroutine_result(self):
        async def _inner():
            async def _async_review(p, r, **kw):
                cs = CoherenceScore(
                    score=0.9,
                    approved=True,
                    h_logical=0.1,
                    h_factual=0.1,
                    warning=False,
                )
                return True, cs

            scorer = MagicMock()
            scorer.review.side_effect = lambda p, r, **kw: _async_review(p, r)
            cs = await _ascore_and_gate(scorer, "log", "q", "response")
            assert cs.score == pytest.approx(0.9)

        asyncio.run(_inner())

    def test_sync_result_with_metadata(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            result = await _ascore_and_gate(scorer, "metadata", "q", "response")
            assert get_score() is not None
            assert result.score == pytest.approx(0.9)

        asyncio.run(_inner())

    def test_raises_on_fail(self):
        async def _inner():
            scorer, cs = _failing_scorer()
            with pytest.raises(HallucinationError):
                await _ascore_and_gate(scorer, "raise", "q", "bad response")

        asyncio.run(_inner())


# ── _GuardedOpenAIStream async (lines 319→exit, 325→exit) ─────────────────────


class TestGuardedOpenAIStreamAsync:
    def test_aiter_with_text_and_final_check(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            chunks = [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=f"tok{i}"))]
                )
                for i in range(3)
            ]

            async def _async_stream():
                for c in chunks:
                    yield c

            from director_ai.integrations.sdk_guard import _GuardedOpenAIStream

            stream = _GuardedOpenAIStream(_async_stream(), scorer, "log", "prompt")
            collected = []
            async for chunk in stream:
                collected.append(chunk)
            assert len(collected) == 3

        asyncio.run(_inner())

    def test_aiter_periodic_check_fires(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            chunks = [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=f"t{i}"))]
                )
                for i in range(STREAM_CHECK_INTERVAL)
            ]

            async def _async_stream():
                for c in chunks:
                    yield c

            from director_ai.integrations.sdk_guard import _GuardedOpenAIStream

            stream = _GuardedOpenAIStream(_async_stream(), scorer, "log", "prompt")
            async for _ in stream:
                pass
            assert scorer.review.call_count >= 1

        asyncio.run(_inner())

    def test_aiter_empty_stream(self):
        async def _inner():
            scorer, cs = _passing_scorer()

            async def _async_stream():
                return
                yield  # pragma: no cover

            from director_ai.integrations.sdk_guard import _GuardedOpenAIStream

            stream = _GuardedOpenAIStream(_async_stream(), scorer, "log", "prompt")
            result = [c async for c in stream]
            assert result == []

        asyncio.run(_inner())


# ── _GuardedAnthropicStream async (lines 445→exit, 451→exit) ──────────────────


class TestGuardedAnthropicStreamAsync:
    def test_aiter_with_text_events(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            events = [SimpleNamespace(text=f"word{i}", delta=None) for i in range(3)]

            async def _async_stream():
                for e in events:
                    yield e

            stream = _GuardedAnthropicStream(_async_stream(), scorer, "log", "prompt")
            collected = [e async for e in stream]
            assert len(collected) == 3

        asyncio.run(_inner())

    def test_aiter_periodic_fires(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            events = [
                SimpleNamespace(text=f"w{i}", delta=None)
                for i in range(STREAM_CHECK_INTERVAL)
            ]

            async def _async_stream():
                for e in events:
                    yield e

            stream = _GuardedAnthropicStream(_async_stream(), scorer, "log", "prompt")
            async for _ in stream:
                pass
            assert scorer.review.call_count >= 1

        asyncio.run(_inner())

    def test_aiter_empty_stream_no_final_check(self):
        async def _inner():
            scorer, cs = _passing_scorer()

            async def _async_stream():
                return
                yield  # pragma: no cover

            stream = _GuardedAnthropicStream(_async_stream(), scorer, "log", "prompt")
            result = [e async for e in stream]
            assert result == []
            scorer.review.assert_not_called()

        asyncio.run(_inner())


# ── Bedrock helpers (lines 474, 479→478, 483→482, 485-487, 494) ──────────────


class TestBedrockHelpers:
    def test_bedrock_response_text_happy(self):
        from director_ai.integrations.sdk_guard import _bedrock_response_text

        resp = {"output": {"message": {"content": [{"text": "Hello Bedrock"}]}}}
        assert _bedrock_response_text(resp) == "Hello Bedrock"

    def test_bedrock_response_text_missing_key(self):
        from director_ai.integrations.sdk_guard import _bedrock_response_text

        assert _bedrock_response_text({}) == ""

    def test_extract_bedrock_prompt_user_list(self):
        msgs = [{"role": "user", "content": [{"text": "hello bedrock"}]}]
        assert _extract_bedrock_prompt(msgs) == "hello bedrock"

    def test_extract_bedrock_prompt_user_string(self):
        msgs = [{"role": "user", "content": "plain string"}]
        assert _extract_bedrock_prompt(msgs) == "plain string"

    def test_extract_bedrock_prompt_no_user(self):
        msgs = [{"role": "assistant", "content": [{"text": "answer"}]}]
        assert _extract_bedrock_prompt(msgs) == ""

    def test_extract_bedrock_prompt_list_no_text_block(self):
        msgs = [{"role": "user", "content": [{"image": "data"}]}]
        assert _extract_bedrock_prompt(msgs) == ""

    def test_extract_bedrock_stream_delta_valid(self):
        event = {"contentBlockDelta": {"delta": {"text": "chunk"}}}
        assert _extract_bedrock_stream_delta(event) == "chunk"

    def test_extract_bedrock_stream_delta_missing(self):
        assert _extract_bedrock_stream_delta({}) is None

    def test_has_bedrock_shape_true(self):
        client = MagicMock()
        client.converse = MagicMock()
        client.invoke_model = MagicMock()
        assert _has_bedrock_shape(client)

    def test_has_bedrock_shape_false(self):
        client = MagicMock(spec=[])
        assert not _has_bedrock_shape(client)


# ── _BedrockProxy.__getattr__ (line 518) ──────────────────────────────────────


class TestBedrockProxy:
    def test_getattr_passthrough(self):
        scorer, cs = _passing_scorer()
        client = MagicMock()
        client.converse.return_value = {
            "output": {"message": {"content": [{"text": "answer"}]}}
        }
        proxy = _BedrockProxy(client, scorer, "log")
        proxy.some_custom_method
        client.some_custom_method  # confirm delegation

    def test_converse_calls_score(self):
        scorer, cs = _passing_scorer()
        client = MagicMock()
        client.converse.return_value = {
            "output": {"message": {"content": [{"text": "The sky is blue."}]}}
        }
        proxy = _BedrockProxy(client, scorer, "log")
        result = proxy.converse(
            messages=[{"role": "user", "content": [{"text": "sky?"}]}]
        )
        scorer.review.assert_called_once()
        assert result is client.converse.return_value

    def test_converse_stream_returns_guarded(self):
        scorer, cs = _passing_scorer()
        client = MagicMock()
        client.converse_stream.return_value = {"stream": iter([])}
        proxy = _BedrockProxy(client, scorer, "log")
        result = proxy.converse_stream(
            messages=[{"role": "user", "content": [{"text": "q"}]}]
        )
        assert isinstance(result, _GuardedBedrockStream)


# ── _GuardedBedrockStream sync + async (lines 536→541, 545, 548-564, 569→exit, 574→exit) ──


class TestGuardedBedrockStream:
    def test_iter_with_stream_key(self):
        scorer, cs = _passing_scorer()
        events = [
            {"contentBlockDelta": {"delta": {"text": f"tok{i}"}}} for i in range(3)
        ]
        response = {"stream": iter(events)}
        stream = _GuardedBedrockStream(response, scorer, "log", "prompt")
        result = list(stream)
        assert len(result) == 3
        scorer.review.assert_called()

    def test_iter_without_stream_key(self):
        scorer, cs = _passing_scorer()
        events = [{"other": "data"}, {"other": "data2"}]

        class _ResponseAsStream:
            def __init__(self, items):
                self._items = items

            def get(self, key, default=None):
                return default

            def __iter__(self):
                return iter(self._items)

        response = _ResponseAsStream(events)
        stream = _GuardedBedrockStream(response, scorer, "log", "prompt")
        result = list(stream)
        assert len(result) == 2

    def test_iter_periodic_check_fires(self):
        scorer, cs = _passing_scorer()
        events = [
            {"contentBlockDelta": {"delta": {"text": f"t{i}"}}}
            for i in range(STREAM_CHECK_INTERVAL)
        ]
        stream = _GuardedBedrockStream(
            {"stream": iter(events)}, scorer, "log", "prompt"
        )
        list(stream)
        assert scorer.review.call_count >= 1

    def test_iter_empty_no_final_check(self):
        scorer, cs = _passing_scorer()
        stream = _GuardedBedrockStream({"stream": iter([])}, scorer, "log", "prompt")
        list(stream)
        scorer.review.assert_not_called()

    def test_aiter_with_text(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            events = [
                {"contentBlockDelta": {"delta": {"text": f"t{i}"}}} for i in range(3)
            ]

            async def _async_stream():
                for e in events:
                    yield e

            class _AsyncResponse:
                def get(self, key, default=None):
                    return _async_stream() if key == "stream" else default

            stream = _GuardedBedrockStream(_AsyncResponse(), scorer, "log", "prompt")
            result = [e async for e in stream]
            assert len(result) == 3

        asyncio.run(_inner())

    def test_aiter_periodic_fires(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            events = [
                {"contentBlockDelta": {"delta": {"text": f"t{i}"}}}
                for i in range(STREAM_CHECK_INTERVAL)
            ]

            async def _async_stream():
                for e in events:
                    yield e

            class _AsyncResponse:
                def get(self, key, default=None):
                    return _async_stream() if key == "stream" else default

            stream = _GuardedBedrockStream(_AsyncResponse(), scorer, "log", "prompt")
            async for _ in stream:
                pass
            assert scorer.review.call_count >= 1

        asyncio.run(_inner())

    def test_aiter_final_check_with_text(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            events = [{"contentBlockDelta": {"delta": {"text": "hello"}}}]

            async def _async_stream():
                for e in events:
                    yield e

            class _AsyncResponse:
                def get(self, key, default=None):
                    return _async_stream() if key == "stream" else default

            stream = _GuardedBedrockStream(_AsyncResponse(), scorer, "log", "prompt")
            async for _ in stream:
                pass
            scorer.review.assert_called()

        asyncio.run(_inner())


# ── _extract_gemini_prompt (lines 590-601) ────────────────────────────────────


class TestExtractGeminiPrompt:
    def test_string_arg(self):
        assert _extract_gemini_prompt(("tell me something",), {}) == "tell me something"

    def test_string_in_kwargs(self):
        assert _extract_gemini_prompt((), {"contents": "from kwargs"}) == "from kwargs"

    def test_list_of_strings(self):
        assert _extract_gemini_prompt((["first", "second"],), {}) == "second"

    def test_list_of_dicts_with_parts_string(self):
        contents = [{"parts": ["part text"]}]
        assert _extract_gemini_prompt((contents,), {}) == "part text"

    def test_list_of_dicts_with_parts_dict(self):
        contents = [{"parts": [{"text": "dict part"}]}]
        assert _extract_gemini_prompt((contents,), {}) == "dict part"

    def test_list_of_dicts_no_text(self):
        contents = [{"parts": [{"image": "data"}]}]
        result = _extract_gemini_prompt((contents,), {})
        assert isinstance(result, str)

    def test_fallback_str_conversion(self):
        result = _extract_gemini_prompt((42,), {})
        assert result == "42"


# ── _GeminiProxy (lines 623) + _GuardedGeminiStream (lines 640→645, 649, 652-667) ──


class TestGeminiProxy:
    def test_generate_content_non_streaming(self):
        scorer, cs = _passing_scorer()
        client = MagicMock()
        response = MagicMock()
        response.text = "Gemini answer."
        client.generate_content.return_value = response
        proxy = _GeminiProxy(client, scorer, "log")
        result = proxy.generate_content("What is 2+2?")
        scorer.review.assert_called_once()
        assert result is response

    def test_generate_content_streaming(self):
        scorer, cs = _passing_scorer()
        client = MagicMock()
        chunks = [SimpleNamespace(text=f"tok{i}") for i in range(3)]
        client.generate_content.return_value = iter(chunks)
        proxy = _GeminiProxy(client, scorer, "log")
        result = proxy.generate_content("prompt", stream=True)
        assert isinstance(result, _GuardedGeminiStream)

    def test_getattr_passthrough(self):
        scorer, cs = _passing_scorer()
        client = MagicMock()
        client.embed_content = "embed_fn"
        proxy = _GeminiProxy(client, scorer, "log")
        assert proxy.embed_content == "embed_fn"

    def test_has_gemini_shape_true(self):
        client = MagicMock()
        client.generate_content = MagicMock()
        assert _has_gemini_shape(client)

    def test_has_gemini_shape_false(self):
        client = MagicMock(spec=[])
        assert not _has_gemini_shape(client)


class TestGuardedGeminiStream:
    def test_iter_with_text_chunks(self):
        scorer, cs = _passing_scorer()
        chunks = [SimpleNamespace(text=f"tok{i}") for i in range(4)]
        stream = _GuardedGeminiStream(iter(chunks), scorer, "log", "prompt")
        result = list(stream)
        assert len(result) == 4
        scorer.review.assert_called()

    def test_iter_periodic_fires(self):
        scorer, cs = _passing_scorer()
        chunks = [SimpleNamespace(text=f"t{i}") for i in range(STREAM_CHECK_INTERVAL)]
        stream = _GuardedGeminiStream(iter(chunks), scorer, "log", "prompt")
        list(stream)
        assert scorer.review.call_count >= 1

    def test_iter_no_text_skips(self):
        scorer, cs = _passing_scorer()
        chunks = [SimpleNamespace(text=None) for _ in range(3)]
        stream = _GuardedGeminiStream(iter(chunks), scorer, "log", "prompt")
        result = list(stream)
        assert len(result) == 3
        scorer.review.assert_not_called()

    def test_aiter_with_text(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            chunks = [SimpleNamespace(text=f"tok{i}") for i in range(3)]

            async def _async_stream():
                for c in chunks:
                    yield c

            stream = _GuardedGeminiStream(_async_stream(), scorer, "log", "prompt")
            result = [c async for c in stream]
            assert len(result) == 3

        asyncio.run(_inner())

    def test_aiter_periodic_fires(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            chunks = [
                SimpleNamespace(text=f"t{i}") for i in range(STREAM_CHECK_INTERVAL)
            ]

            async def _async_stream():
                for c in chunks:
                    yield c

            stream = _GuardedGeminiStream(_async_stream(), scorer, "log", "prompt")
            async for _ in stream:
                pass
            assert scorer.review.call_count >= 1

        asyncio.run(_inner())

    def test_aiter_final_check(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            chunks = [SimpleNamespace(text="hello")]

            async def _async_stream():
                for c in chunks:
                    yield c

            stream = _GuardedGeminiStream(_async_stream(), scorer, "log", "prompt")
            async for _ in stream:
                pass
            scorer.review.assert_called()

        asyncio.run(_inner())

    def test_aiter_empty_no_final_check(self):
        async def _inner():
            scorer, cs = _passing_scorer()

            async def _async_stream():
                return
                yield  # pragma: no cover

            stream = _GuardedGeminiStream(_async_stream(), scorer, "log", "prompt")
            result = [c async for c in stream]
            assert result == []
            scorer.review.assert_not_called()

        asyncio.run(_inner())


# ── guard() Gemini + Bedrock + Cohere shapes ─────────────────────────────────


class TestGuardShapeDetection:
    def test_guard_gemini_shape(self):
        client = MagicMock(spec=["generate_content"])
        client.generate_content = MagicMock()
        guarded = guard(client, on_fail="log")
        assert isinstance(guarded, _GeminiProxy)

    def test_guard_bedrock_shape(self):
        client = MagicMock(spec=["converse", "invoke_model"])
        client.converse = MagicMock()
        client.invoke_model = MagicMock()
        guarded = guard(client, on_fail="log")
        assert isinstance(guarded, _BedrockProxy)

    def test_guard_cohere_shape(self):
        client = MagicMock(spec=["chat"])
        client.chat = MagicMock()
        # no completions attribute to avoid openai-shape detection
        type(client.chat).completions = property(lambda self: None)
        guarded = guard(client, on_fail="log")
        assert isinstance(guarded, _CohereProxy)

    def test_guard_invalid_on_fail(self):
        client = MagicMock()
        with pytest.raises(ValueError, match="on_fail"):
            guard(client, on_fail="invalid")


# ── _has_cohere_shape ─────────────────────────────────────────────────────────


class TestHasCohereShape:
    def test_true_for_chat_without_completions(self):
        client = MagicMock(spec=["chat"])
        client.chat = MagicMock(spec=[])
        assert _has_cohere_shape(client)

    def test_false_for_openai_shape(self):
        client = MagicMock()
        client.chat.completions.create = MagicMock()
        assert not _has_cohere_shape(client)


# ── _CohereProxy + _GuardedCohereStream (lines 714, 731→736, 740, 743-758) ────


class TestCohereProxy:
    def test_chat_scores(self):
        scorer, cs = _passing_scorer()
        client = MagicMock()
        response = MagicMock()
        response.text = "Cohere answer."
        client.chat.return_value = response
        proxy = _CohereProxy(client, scorer, "log")
        result = proxy.chat(message="hello?")
        scorer.review.assert_called_once()
        assert result is response

    def test_chat_stream_returns_guarded(self):
        scorer, cs = _passing_scorer()
        client = MagicMock()
        client.chat_stream.return_value = iter([])
        proxy = _CohereProxy(client, scorer, "log")
        result = proxy.chat_stream(message="hello?")
        assert isinstance(result, _GuardedCohereStream)

    def test_getattr_passthrough(self):
        scorer, cs = _passing_scorer()
        client = MagicMock()
        client.embed = "embed_fn"
        proxy = _CohereProxy(client, scorer, "log")
        assert proxy.embed == "embed_fn"


class TestGuardedCohereStream:
    def test_iter_with_text_events(self):
        scorer, cs = _passing_scorer()
        events = [SimpleNamespace(text=f"tok{i}") for i in range(4)]
        stream = _GuardedCohereStream(iter(events), scorer, "log", "prompt")
        result = list(stream)
        assert len(result) == 4
        scorer.review.assert_called()

    def test_iter_periodic_fires(self):
        scorer, cs = _passing_scorer()
        events = [SimpleNamespace(text=f"t{i}") for i in range(STREAM_CHECK_INTERVAL)]
        stream = _GuardedCohereStream(iter(events), scorer, "log", "prompt")
        list(stream)
        assert scorer.review.call_count >= 1

    def test_iter_no_text_skips(self):
        scorer, cs = _passing_scorer()
        events = [SimpleNamespace(text=None) for _ in range(3)]
        stream = _GuardedCohereStream(iter(events), scorer, "log", "prompt")
        result = list(stream)
        assert len(result) == 3
        scorer.review.assert_not_called()

    def test_aiter_with_text(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            events = [SimpleNamespace(text=f"tok{i}") for i in range(3)]

            async def _async_stream():
                for e in events:
                    yield e

            stream = _GuardedCohereStream(_async_stream(), scorer, "log", "prompt")
            result = [e async for e in stream]
            assert len(result) == 3

        asyncio.run(_inner())

    def test_aiter_periodic_fires(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            events = [
                SimpleNamespace(text=f"t{i}") for i in range(STREAM_CHECK_INTERVAL)
            ]

            async def _async_stream():
                for e in events:
                    yield e

            stream = _GuardedCohereStream(_async_stream(), scorer, "log", "prompt")
            async for _ in stream:
                pass
            assert scorer.review.call_count >= 1

        asyncio.run(_inner())

    def test_aiter_final_check_fires(self):
        async def _inner():
            scorer, cs = _passing_scorer()
            events = [SimpleNamespace(text="hello")]

            async def _async_stream():
                for e in events:
                    yield e

            stream = _GuardedCohereStream(_async_stream(), scorer, "log", "prompt")
            async for _ in stream:
                pass
            scorer.review.assert_called()

        asyncio.run(_inner())

    def test_aiter_empty_no_final_check(self):
        async def _inner():
            scorer, cs = _passing_scorer()

            async def _async_stream():
                return
                yield  # pragma: no cover

            stream = _GuardedCohereStream(_async_stream(), scorer, "log", "prompt")
            result = [e async for e in stream]
            assert result == []
            scorer.review.assert_not_called()

        asyncio.run(_inner())

    def test_periodic_check_raises_on_fail(self):
        scorer, cs = _failing_scorer()
        events = [SimpleNamespace(text=f"t{i}") for i in range(STREAM_CHECK_INTERVAL)]
        stream = _GuardedCohereStream(iter(events), scorer, "raise", "prompt")
        with pytest.raises(HallucinationError):
            list(stream)
