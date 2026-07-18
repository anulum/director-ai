# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming Disclosure Modes (KIMI2-A)
"""Behavioural contracts for the proxy ``stream_disclosure`` modes.

KIMI proved end-to-end that in the immediate mode a mid-stream halt leaks the
tokens emitted before the halt ("800 mg of ibuprofen is safe per" reached the
client). The ``buffered`` mode inverts that proof: chunks are withheld until
the accumulated content passes a review, a halt discards the unreleased
window, and the terminal ``[DONE]`` review gates the final release — so a
rejected stream never discloses unreviewed content. These tests pin both
modes, the fail-closed upstream-drop path, and the config validation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from director_ai.proxy import (
    STREAM_CHECK_INTERVAL,
    STREAM_MAX_PENDING_CHARS,
    STREAM_MAX_PENDING_LINES,
    _stream_chat_content,
    _stream_tool_call_content,
    create_proxy_app,
)

pytestmark = pytest.mark.asyncio


class _FixedScorer:
    def __init__(self, approved: bool = True, score: float = 0.9) -> None:
        self.approved = approved
        self.score = score
        self.calls: list[tuple[str, str]] = []

    def review(self, prompt: str, content: str) -> tuple[bool, SimpleNamespace]:
        self.calls.append((prompt, content))
        return self.approved, SimpleNamespace(
            score=self.score,
            verdict_confidence=0.75,
        )


def _streaming_transport(lines: list[str]) -> httpx.MockTransport:
    async def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="\n".join(lines) + "\n",
        )

    return httpx.MockTransport(_handler)


def _app(
    scorer: _FixedScorer,
    lines: list[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    disclosure: str,
) -> FastAPI:
    import director_ai.proxy as proxy

    monkeypatch.setattr(proxy, "CoherenceScorer", lambda **_kw: scorer)
    return create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        on_fail="reject",
        stream_disclosure=disclosure,
        _transport=_streaming_transport(lines),
    )


async def _post_stream(app: FastAPI, prompt: str = "ask") -> str:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stream-model",
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    assert resp.status_code == 200
    return resp.text


def _content_lines(n: int) -> list[str]:
    return [
        f'data: {{"choices":[{{"delta":{{"content":"t{idx}"}}}}]}}' for idx in range(n)
    ]


async def test_buffered_halted_stream_discloses_no_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The inverse of KIMI's proof: the periodic review rejects, and the client
    # receives NONE of the withheld tokens — only the halt marker.
    scorer = _FixedScorer(approved=False, score=0.1)
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(9) + ["data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    for idx in range(9):
        assert f"t{idx}" not in body
    assert '"finish_reason": "content_filter"' in body
    assert body.rstrip().endswith("data: [DONE]")


async def test_buffered_clean_stream_releases_everything_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _FixedScorer(approved=True, score=0.95)
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(10) + ["data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    positions = [body.index(f'"content":"t{idx}"') for idx in range(10)]
    assert positions == sorted(positions)
    assert body.rstrip().endswith("data: [DONE]")


async def test_buffered_final_review_rejection_discards_the_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fewer chunks than the periodic interval: the only review is at [DONE],
    # and its rejection must withhold the entire stream.
    scorer = _FixedScorer(approved=False, score=0.2)
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(3) + ["data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    for idx in range(3):
        assert f"t{idx}" not in body
    assert '"finish_reason": "content_filter"' in body
    assert scorer.calls == [("ask", "t0t1t2")]


async def test_buffered_final_review_pass_releases_the_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _FixedScorer(approved=True, score=0.9)
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(3) + ["data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    for idx in range(3):
        assert f'"content":"t{idx}"' in body
    assert body.rstrip().endswith("data: [DONE]")


async def test_buffered_upstream_drop_without_done_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No [DONE]: the unreviewed pending window is discarded rather than
    # released unreviewed.
    scorer = _FixedScorer(approved=True, score=0.9)
    body = await _post_stream(
        _app(scorer, _content_lines(3), monkeypatch, disclosure="buffered"),
    )

    for idx in range(3):
        assert f"t{idx}" not in body
    assert scorer.calls == []


async def test_buffered_drops_malformed_data_and_keeps_framing_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Framing lines stay ordered; malformed data lines never reach the client.

    KIMI3-H3: an unparseable data line can never pass review, so buffered
    mode drops it at parse time instead of releasing it with the reviewed
    window (the pre-fix behaviour released it unreviewed).
    """
    scorer = _FixedScorer(approved=True, score=0.9)
    body = await _post_stream(
        _app(
            scorer,
            [": keepalive", "data: not-json", *_content_lines(2), "data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    assert "data: not-json" not in body
    # SSE framing is still withheld and released in upstream order.
    assert body.index(": keepalive") < body.index('"content":"t0"')
    assert body.rstrip().endswith("data: [DONE]")


async def test_buffered_garbage_only_stream_releases_no_unreviewed_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KIMI3-H3 regression: a stream of only malformed data lines leaks nothing.

    Pre-fix, the withheld window (including the malformed lines) was
    released unreviewed at ``[DONE]`` because the empty content buffer
    skipped the final review entirely.
    """
    scorer = _FixedScorer(approved=True, score=0.9)
    body = await _post_stream(
        _app(
            scorer,
            ["data: {broken", "data: also-broken", "data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    assert "broken" not in body
    assert body.rstrip().endswith("data: [DONE]")
    # Nothing parseable ever accumulated, so the scorer was never consulted.
    assert scorer.calls == []


async def test_immediate_default_still_discloses_pre_halt_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pin today's default: the tokens streamed BEFORE the rejecting review
    # reach the client (the documented partial-disclosure behaviour), while
    # the line that triggered the review and everything after it do not —
    # proving the buffered fix did not change the default mode.
    import director_ai.proxy as proxy

    scorer = _FixedScorer(approved=False, score=0.1)
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(proxy.STREAM_CHECK_INTERVAL + 1) + ["data: [DONE]"],
            monkeypatch,
            disclosure="immediate",
        ),
    )

    for idx in range(proxy.STREAM_CHECK_INTERVAL - 1):
        assert f'"content":"t{idx}"' in body
    assert f"t{proxy.STREAM_CHECK_INTERVAL - 1}" not in body
    assert f"t{proxy.STREAM_CHECK_INTERVAL}" not in body
    assert '"finish_reason": "content_filter"' in body


async def test_create_proxy_app_rejects_unknown_disclosure_mode() -> None:
    with pytest.raises(ValueError, match="stream_disclosure"):
        create_proxy_app(stream_disclosure="bogus")


# --- KIMI3-H1: tool-call streams must not bypass the review ----------------


def _tool_call_lines(*fragments: str, name: str | None = "transfer") -> list[str]:
    """Tool-call-only delta chunks (no ``delta.content``), one per fragment.

    The function name rides on the first chunk (OpenAI streaming shape) and
    the arguments stream across the remaining chunks.
    """
    lines: list[str] = []
    for index, fragment in enumerate(fragments):
        function: dict[str, object] = {"arguments": fragment}
        if index == 0 and name is not None:
            function["name"] = name
        chunk = {
            "choices": [{"delta": {"tool_calls": [{"index": 0, "function": function}]}}]
        }
        lines.append("data: " + json.dumps(chunk))
    return lines


async def test_stream_tool_call_content_extracts_name_and_arguments() -> None:
    chunk = json.loads(_tool_call_lines('{"amount": 1000000}')[0][6:])
    assert _stream_tool_call_content(chunk) == 'transfer{"amount": 1000000}'
    # Content-only and malformed shapes contribute nothing.
    assert _stream_tool_call_content({"choices": [{"delta": {"content": "hi"}}]}) == ""
    assert _stream_tool_call_content({"choices": []}) == ""
    assert _stream_tool_call_content("not-a-dict") == ""


async def test_stream_chat_content_merges_content_and_tool_calls() -> None:
    # A chunk carrying both content and a tool call yields the concatenation,
    # so the reviewed text never silently drops the tool-call payload.
    chunk = {
        "choices": [
            {
                "delta": {
                    "content": "calling ",
                    "tool_calls": [{"function": {"name": "pay", "arguments": "{}"}}],
                }
            }
        ]
    }
    assert _stream_chat_content(chunk) == "calling pay{}"


async def test_buffered_tool_call_only_stream_is_reviewed_and_can_halt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The H1 inverse proof: a stream that ONLY calls a tool (no delta.content)
    # is still reviewed, and a rejecting review discloses none of the tool-call
    # arguments — before the fix reviews=0 and the arguments reached the client.
    scorer = _FixedScorer(approved=False, score=0.1)
    body = await _post_stream(
        _app(
            scorer,
            _tool_call_lines('{"amount": ', "1000000}") + ["data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    assert "1000000" not in body
    assert "transfer" not in body
    assert '"finish_reason": "content_filter"' in body
    # The review actually ran on the reconstructed tool-call payload.
    assert scorer.calls == [("ask", 'transfer{"amount": 1000000}')]


async def test_buffered_tool_call_clean_stream_releases_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = _FixedScorer(approved=True, score=0.95)
    body = await _post_stream(
        _app(
            scorer,
            _tool_call_lines("city=", "Paris") + ["data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    assert body.index("city=") < body.index("Paris")
    assert body.rstrip().endswith("data: [DONE]")
    assert scorer.calls == [("ask", "transfercity=Paris")]


async def test_immediate_tool_call_stream_halts_future_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # In immediate mode a tool-call stream now also triggers the mid-stream
    # review (previously it never did); the rejecting review stops the chunk
    # that crossed the interval and everything after it.
    import director_ai.proxy as proxy

    scorer = _FixedScorer(approved=False, score=0.1)
    fragments = [f"a{idx}" for idx in range(proxy.STREAM_CHECK_INTERVAL)]
    body = await _post_stream(
        _app(
            scorer,
            _tool_call_lines(*fragments) + ["data: [DONE]"],
            monkeypatch,
            disclosure="immediate",
        ),
    )

    assert '"finish_reason": "content_filter"' in body
    assert scorer.calls, "the tool-call stream must reach the scorer in immediate mode"


def _multi_choice_tool_call_line(arg_fragment: str) -> str:
    """A valid n>1 chunk: empty first choice, sensitive tool call in a later one."""
    chunk = {
        "choices": [
            {"index": 0, "delta": {}},
            {
                "index": 1,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "name": "transfer",
                                "arguments": arg_fragment,
                            },
                        }
                    ]
                },
            },
        ]
    }
    return "data: " + json.dumps(chunk)


async def test_stream_chat_content_reads_all_choices_in_wire_order() -> None:
    # Content on the first choice, tool call on the second — both reviewed.
    chunk = {
        "choices": [
            {"delta": {"content": "hi "}},
            {
                "delta": {
                    "tool_calls": [{"function": {"name": "pay", "arguments": "{}"}}]
                }
            },
        ]
    }
    assert _stream_chat_content(chunk) == "hi pay{}"
    # The second-eye probe: empty first choice, sensitive tool call later.
    probe = json.loads(_multi_choice_tool_call_line("amt=5")[6:])
    assert _stream_chat_content(probe) == "transferamt=5"


async def test_buffered_multi_choice_later_tool_call_is_reviewed_and_can_halt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Second-eye blocking finding: a valid n>1 chunk with an empty first choice
    # and a sensitive tool call in a later choice must still be reviewed —
    # reading only choices[0] recreated H1. A rejecting review discloses none
    # of the later-choice tool-call arguments.
    scorer = _FixedScorer(approved=False, score=0.1)
    body = await _post_stream(
        _app(
            scorer,
            [_multi_choice_tool_call_line('{"amount": 999999}'), "data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    assert "999999" not in body
    assert "transfer" not in body
    assert '"finish_reason": "content_filter"' in body
    assert scorer.calls == [("ask", 'transfer{"amount": 999999}')]


# --- KIMI3-H2: the withheld pending window must be bounded (memory DoS) ---


@pytest.mark.parametrize(
    "flood_line",
    [
        ": keepalive",  # non-data line (withheld for ordering)
        'data: {"choices":[{"delta":{}}]}',  # valid but content-less chunk
    ],
    ids=["non-data", "content-less"],
    # NB: malformed data lines ("data: not-json") no longer flood the window —
    # KIMI3-H3 drops them at parse time, covered by
    # test_buffered_drops_malformed_data_and_keeps_framing_order and
    # test_buffered_garbage_only_stream_releases_no_unreviewed_bytes.
)
async def test_buffered_pending_line_flood_fails_closed(
    flood_line: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flood of withheld-but-unreviewable lines halts instead of buffering."""
    scorer = _FixedScorer(approved=True, score=0.9)
    lines = (
        [flood_line] * (STREAM_MAX_PENDING_LINES + 5)
        + _content_lines(1)
        + ["data: [DONE]"]
    )
    body = await _post_stream(_app(scorer, lines, monkeypatch, disclosure="buffered"))

    # failed closed with a halt marker, and the withheld flood was dropped
    assert '"finish_reason": "content_filter"' in body
    assert "data: [DONE]" in body
    assert flood_line not in body
    assert "t0" not in body  # content queued after the flood never reached the client
    # overflow halts before an approving review of released content could run
    assert scorer.calls == []


async def test_buffered_pending_char_flood_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A few oversized withheld lines trip the byte cap before the line cap."""
    scorer = _FixedScorer(approved=True, score=0.9)
    big = ": " + "x" * (STREAM_MAX_PENDING_CHARS // 4)
    lines = [big] * 5 + _content_lines(1) + ["data: [DONE]"]
    body = await _post_stream(_app(scorer, lines, monkeypatch, disclosure="buffered"))

    assert '"finish_reason": "content_filter"' in body
    assert "xxxx" not in body  # the oversized withheld lines were dropped


async def test_buffered_pending_under_cap_streams_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Withholding below the caps is untouched: released in order after review."""
    scorer = _FixedScorer(approved=True, score=0.9)
    lines = (
        [": keepalive"] * 10 + _content_lines(STREAM_CHECK_INTERVAL) + ["data: [DONE]"]
    )
    body = await _post_stream(_app(scorer, lines, monkeypatch, disclosure="buffered"))

    assert '"finish_reason": "content_filter"' not in body
    assert ": keepalive" in body  # released once the review passed
    assert "t0" in body
    assert scorer.calls  # a real review ran and approved the release


# --- KIMI3-H5: a scorer exception mid-stream must fail closed, not abort ---


class _RaisingScorer:
    """A scorer whose review always raises, to prove the halt/audit path."""

    def __init__(self) -> None:
        self.calls = 0

    def review(self, prompt: str, content: str) -> tuple[bool, object]:
        self.calls += 1
        raise RuntimeError("scorer boom")


@pytest.mark.parametrize("disclosure", ["immediate", "buffered"])
async def test_scorer_exception_at_done_fails_closed(
    disclosure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fewer than the interval: the only review is at [DONE]; its raise halts."""
    scorer = _RaisingScorer()
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(2) + ["data: [DONE]"],
            monkeypatch,
            disclosure=disclosure,
        ),
    )

    assert scorer.calls == 1  # the terminal review was attempted and raised
    # failed closed with a halt + terminal marker, not a silent truncated stream
    assert '"finish_reason": "content_filter"' in body
    assert body.rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize("disclosure", ["immediate", "buffered"])
async def test_scorer_exception_at_interval_fails_closed(
    disclosure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The periodic review raises mid-stream and the generator halts + audits."""
    scorer = _RaisingScorer()
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(STREAM_CHECK_INTERVAL + 1) + ["data: [DONE]"],
            monkeypatch,
            disclosure=disclosure,
        ),
    )

    assert scorer.calls == 1  # raised on the interval review, not swallowed
    assert '"finish_reason": "content_filter"' in body
    assert body.rstrip().endswith("data: [DONE]")
