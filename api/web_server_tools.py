"""Local handlers for Anthropic web server tools (facade for :mod:`api.web_tools`).

OpenAI-compatible upstreams can emit regular function calls, but Anthropic's
web tools are server-side: the API response itself must include the tool result.
"""

from __future__ import annotations

import httpx

from api.web_tools import constants as _web_tool_constants
from api.web_tools import outbound
from api.web_tools.egress import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    enforce_web_fetch_egress,
)
from api.web_tools.request import is_web_server_tool_request
from api.web_tools.streaming import stream_web_server_tool_response

# Re-exports for tests and backwards-compatible monkeypatching.
_MAX_WEB_FETCH_REDIRECTS = _web_tool_constants._MAX_WEB_FETCH_REDIRECTS
_MAX_WEB_FETCH_RESPONSE_BYTES = _web_tool_constants._MAX_WEB_FETCH_RESPONSE_BYTES
_drain_response_body_capped = outbound._drain_response_body_capped
_read_response_body_capped = outbound._read_response_body_capped
_run_web_fetch = outbound._run_web_fetch
_run_web_search = outbound._run_web_search

__all__ = [
    "_MAX_WEB_FETCH_REDIRECTS",
    "_MAX_WEB_FETCH_RESPONSE_BYTES",
    "WebFetchEgressPolicy",
    "WebFetchEgressViolation",
    "_drain_response_body_capped",
    "_read_response_body_capped",
    "_run_web_fetch",
    "_run_web_search",
    "enforce_web_fetch_egress",
    "httpx",
    "is_web_server_tool_request",
    "stream_web_server_tool_response",
]
