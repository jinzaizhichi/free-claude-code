"""Pydantic passthrough of Anthropic protocol fields (e.g. ``cache_control``)."""

from __future__ import annotations

from api.models.anthropic import (
    ContentBlockText,
    Message,
    MessagesRequest,
    SystemContent,
    Tool,
)
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import (
    build_base_native_anthropic_request_body,
)


def test_cache_control_preserved_on_parsed_user_text_system_and_tool() -> None:
    raw = {
        "model": "m",
        "max_tokens": 20,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "system": [
            {
                "type": "text",
                "text": "be brief",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": [
            {
                "name": "alpha",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    req = MessagesRequest.model_validate(raw)
    block = req.messages[0].content[0]
    assert isinstance(block, ContentBlockText)
    assert block.model_dump()["cache_control"] == {"type": "ephemeral"}
    s0 = req.system[0] if isinstance(req.system, list) else None
    assert isinstance(s0, SystemContent)
    assert s0.model_dump()["cache_control"] == {"type": "ephemeral"}
    t0 = req.tools[0] if req.tools else None
    assert isinstance(t0, Tool)
    assert t0.model_dump()["cache_control"] == {"type": "ephemeral"}


def test_build_base_native_body_includes_cache_control() -> None:
    req = MessagesRequest(
        model="m",
        max_tokens=20,
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText.model_validate(
                        {
                            "type": "text",
                            "text": "x",
                            "cache_control": {"type": "ephemeral"},
                        }
                    )
                ],
            )
        ],
        system=[
            SystemContent.model_validate(
                {
                    "type": "text",
                    "text": "s",
                    "cache_control": {"type": "ephemeral"},
                }
            )
        ],
        tools=[
            Tool.model_validate(
                {
                    "name": "n",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                }
            )
        ],
    )
    body = build_base_native_anthropic_request_body(
        req,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=False,
    )
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][0]["cache_control"] == {"type": "ephemeral"}


def test_pydantic_discriminator_still_distinguishes_blocks() -> None:
    m = Message.model_validate(
        {
            "role": "user",
            "content": [{"type": "text", "text": "a", "z": 1}],
        }
    )
    b = m.content[0]
    assert isinstance(b, ContentBlockText)
    assert b.model_dump()["z"] == 1
