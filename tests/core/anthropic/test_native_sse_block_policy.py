"""Unit tests for shared native Anthropic SSE thinking policy / block remapping."""

from __future__ import annotations

import json

from core.anthropic.native_sse_block_policy import (
    NativeSseBlockPolicyState,
    format_native_sse_event,
    transform_native_sse_block_event,
)


def test_thinking_start_dropped_when_disabled() -> None:
    st = NativeSseBlockPolicyState()
    payload = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": ""},
    }
    ev = format_native_sse_event(
        "content_block_start",
        json.dumps(payload),
    )
    assert transform_native_sse_block_event(ev, st, thinking_enabled=False) is None


def test_thinking_delta_dropped_when_disabled() -> None:
    st = NativeSseBlockPolicyState()
    # No prior start in stream (OpenRouter-style: returns None when thinking off)
    payload = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "secret"},
    }
    ev = format_native_sse_event("content_block_delta", json.dumps(payload))
    assert transform_native_sse_block_event(ev, st, thinking_enabled=False) is None


def test_text_block_passthrough_when_thinking_disabled() -> None:
    st = NativeSseBlockPolicyState()
    payload = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    ev = format_native_sse_event("content_block_start", json.dumps(payload))
    out = transform_native_sse_block_event(ev, st, thinking_enabled=False)
    assert out is not None
    assert '"index": 0' in (out or "")
