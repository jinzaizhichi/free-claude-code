import asyncio
from typing import cast
from unittest.mock import MagicMock

import pytest
from starlette.responses import StreamingResponse

from api.models.anthropic import Message, MessagesRequest, Tool
from api.services import ClaudeProxyService
from api.web_server_tools import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    enforce_web_fetch_egress,
    is_web_server_tool_request,
    stream_web_server_tool_response,
)
from config.settings import Settings
from core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
)

_STRICT_EGRESS = WebFetchEgressPolicy(
    allow_private_network_targets=False,
    allowed_schemes=frozenset({"http", "https"}),
)


def test_web_server_tool_not_detected_when_tool_only_listed():
    """Listing web_search without forcing it must not skip the upstream provider."""
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="search")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
    )

    assert not is_web_server_tool_request(request)


def test_web_server_tool_detected_when_tool_choice_forces_it():
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="search")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    assert is_web_server_tool_request(request)


def test_web_server_tool_not_detected_when_forced_name_missing_from_tools():
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="hi")],
        tools=[Tool(name="other", type="function")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    assert not is_web_server_tool_request(request)


def test_service_routes_forced_web_tool_to_provider_when_disabled():
    """Local web tools are opt-in; disabled means normal provider streaming."""
    settings = Settings()
    assert settings.enable_web_server_tools is False
    calls: list[int] = []

    async def fake_stream(*_args, **_kwargs):
        calls.append(1)
        yield 'event: message_start\ndata: {"type":"message_start"}\n\n'

    mock_provider = MagicMock()
    mock_provider.stream_response = fake_stream
    service = ClaudeProxyService(settings, provider_getter=lambda _: mock_provider)
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content="Perform a web search for the query: DeepSeek V4 model release 2026",
            )
        ],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )
    response = cast(StreamingResponse, service.create_message(request))
    body = response.body_iterator

    async def drain():
        async for _chunk in body:
            pass

    asyncio.run(drain())
    assert calls == [1]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://192.168.1.1/",
        "http://10.0.0.1/",
        "http://[::1]/",
        "http://localhost/foo",
        "http://mybox.local/",
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",
    ],
)
def test_enforce_web_fetch_egress_blocks_internal_or_disallowed(url: str):
    with pytest.raises(WebFetchEgressViolation):
        enforce_web_fetch_egress(url, _STRICT_EGRESS)


def test_enforce_web_fetch_egress_allows_global_literal_ip():
    enforce_web_fetch_egress("http://8.8.8.8/", _STRICT_EGRESS)


def test_enforce_web_fetch_egress_skips_private_checks_when_opted_in():
    enforce_web_fetch_egress(
        "http://127.0.0.1/",
        WebFetchEgressPolicy(
            allow_private_network_targets=True,
            allowed_schemes=frozenset({"http", "https"}),
        ),
    )


@pytest.mark.asyncio
async def test_streams_web_search_server_tool_result(monkeypatch):
    async def fake_search(query: str) -> list[dict[str, str]]:
        assert query == "DeepSeek V4 model release 2026"
        return [{"title": "DeepSeek V4 Released", "url": "https://example.com/v4"}]

    monkeypatch.setattr("api.web_server_tools._run_web_search", fake_search)
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content=(
                    "Perform a web search for the query: DeepSeek V4 model release 2026"
                ),
            )
        ],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    raw = "".join(
        [
            event
            async for event in stream_web_server_tool_response(
                request, input_tokens=42, web_fetch_egress=_STRICT_EGRESS
            )
        ]
    )
    events = parse_sse_text(raw)
    assert_anthropic_stream_contract(events)
    starts = [e for e in events if e.event == "content_block_start"]
    assert starts[0].data["content_block"]["type"] == "server_tool_use"
    assert starts[0].data["content_block"]["name"] == "web_search"
    assert starts[1].data["content_block"]["type"] == "web_search_tool_result"
    assert starts[1].data["content_block"]["content"][0]["url"] == (
        "https://example.com/v4"
    )
    deltas = [e for e in events if e.event == "message_delta"]
    assert deltas[-1].data["usage"]["server_tool_use"] == {"web_search_requests": 1}


@pytest.mark.asyncio
async def test_streams_web_fetch_server_tool_result(monkeypatch):
    async def fake_fetch(url: str, _egress: WebFetchEgressPolicy) -> dict[str, str]:
        assert url == "https://example.com/article"
        return {
            "url": url,
            "title": "Example Article",
            "media_type": "text/plain",
            "data": "Article body",
        }

    monkeypatch.setattr("api.web_server_tools._run_web_fetch", fake_fetch)
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(role="user", content="Fetch https://example.com/article please")
        ],
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    raw = "".join(
        [
            event
            async for event in stream_web_server_tool_response(
                request, input_tokens=42, web_fetch_egress=_STRICT_EGRESS
            )
        ]
    )
    events = parse_sse_text(raw)
    assert_anthropic_stream_contract(events)
    starts = [e for e in events if e.event == "content_block_start"]
    assert starts[0].data["content_block"]["type"] == "server_tool_use"
    assert starts[1].data["content_block"]["type"] == "web_fetch_tool_result"
    assert starts[1].data["content_block"]["content"]["content"]["title"] == (
        "Example Article"
    )
    deltas = [e for e in events if e.event == "message_delta"]
    assert deltas[-1].data["usage"]["server_tool_use"] == {"web_fetch_requests": 1}
