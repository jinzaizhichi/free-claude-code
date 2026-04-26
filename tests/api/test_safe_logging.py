"""Tests that API and SSE logging avoid raw sensitive payloads by default."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from api import services as services_mod
from api.models.anthropic import Message, MessagesRequest
from api.services import ClaudeProxyService
from config.settings import Settings
from core.anthropic.sse import SSEBuilder


def test_create_message_skips_full_payload_debug_log_by_default():
    settings = Settings()
    assert settings.log_raw_api_payloads is False
    mock_provider = MagicMock()

    async def fake_stream(*_a, **_kw):
        yield "event: ping\ndata: {}\n\n"

    mock_provider.stream_response = fake_stream
    service = ClaudeProxyService(settings, provider_getter=lambda _: mock_provider)

    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[Message(role="user", content="secret-user-text")],
    )

    with patch.object(services_mod.logger, "debug") as mock_debug:
        service.create_message(request)

    full_payload_calls = [
        c
        for c in mock_debug.call_args_list
        if c.args and str(c.args[0]) == "FULL_PAYLOAD [{}]: {}"
    ]
    assert not full_payload_calls


def test_create_message_logs_full_payload_when_opt_in():
    settings = Settings()
    settings.log_raw_api_payloads = True
    mock_provider = MagicMock()

    async def fake_stream(*_a, **_kw):
        yield "event: ping\ndata: {}\n\n"

    mock_provider.stream_response = fake_stream
    service = ClaudeProxyService(settings, provider_getter=lambda _: mock_provider)
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[Message(role="user", content="visible")],
    )

    with patch.object(services_mod.logger, "debug") as mock_debug:
        service.create_message(request)

    keys = [c.args[0] for c in mock_debug.call_args_list if c.args]
    assert any(k == "FULL_PAYLOAD [{}]: {}" for k in keys)


def test_sse_builder_default_debug_has_no_serialized_json_content():
    with patch("core.anthropic.sse.logger.debug") as mock_debug:
        sse = SSEBuilder("msg_x", "m", 1, log_raw_events=False)
        sse.message_start()

    assert mock_debug.call_count == 1
    message = str(mock_debug.call_args)
    assert "serialized_bytes=" in message
    assert "role" not in message
    assert "assistant" not in message


def test_sse_builder_raw_logging_includes_event_body_when_enabled():
    with patch("core.anthropic.sse.logger.debug") as mock_debug:
        sse = SSEBuilder("msg_x", "m", 1, log_raw_events=True)
        sse.message_start()

    assert mock_debug.call_count == 1
    message = str(mock_debug.call_args)
    assert "message_start" in message
    assert "role" in message
