"""Local handlers for Anthropic web server tools.

OpenAI-compatible upstreams can emit regular function calls, but Anthropic's
web tools are server-side: the API response itself must include the tool result.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from loguru import logger

from core.anthropic.server_tool_sse import (
    SERVER_TOOL_USE,
    WEB_FETCH_TOOL_ERROR,
    WEB_FETCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT_ERROR,
)
from core.anthropic.sse import format_sse_event

from .models.anthropic import MessagesRequest

_REQUEST_TIMEOUT_S = 20.0
_MAX_SEARCH_RESULTS = 10
_MAX_FETCH_CHARS = 24_000
# Hard cap on raw bytes read from HTTP responses before decode / HTML parse (memory bound).
_MAX_WEB_FETCH_RESPONSE_BYTES = 2 * 1024 * 1024
# Drain at most this many bytes from redirect responses before following Location.
_REDIRECT_RESPONSE_BODY_CAP_BYTES = 65_536
_MAX_WEB_FETCH_REDIRECTS = 10
_WEB_FETCH_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# Shared HTTP defaults for outbound web tool requests (avoid duplicated headers).
_WEB_TOOL_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 compatible; free-claude-code/2.0",
}


class WebFetchEgressViolation(ValueError):
    """Raised when a web_fetch URL is rejected by egress policy (SSRF guard)."""


@dataclass(frozen=True, slots=True)
class WebFetchEgressPolicy:
    """Egress rules for user-influenced web_fetch URLs."""

    allow_private_network_targets: bool
    allowed_schemes: frozenset[str]


def enforce_web_fetch_egress(url: str, policy: WebFetchEgressPolicy) -> None:
    """Validate ``url`` before performing web_fetch; raise on policy violations.

    .. note::
        Hostnames are resolved to IP addresses at validation time. A malicious or
        flaky DNS server could in theory change answers before the HTTP client
        connects (time-of-check/time-of-use). Fully closing that gap requires
        transport-level pinning or connect-time re-validation; this tool accepts
        that residual risk (see tests).
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in policy.allowed_schemes:
        raise WebFetchEgressViolation(
            f"URL scheme {scheme!r} is not allowed for web_fetch"
        )

    host = parsed.hostname
    if host is None or host == "":
        raise WebFetchEgressViolation("web_fetch URL must include a host")

    if policy.allow_private_network_targets:
        return

    host_lower = host.lower()
    if host_lower == "localhost" or host_lower.endswith(".localhost"):
        raise WebFetchEgressViolation("localhost targets are not allowed for web_fetch")
    if host_lower.endswith(".local"):
        raise WebFetchEgressViolation(".local hostnames are not allowed for web_fetch")

    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        parsed_ip = None

    if parsed_ip is not None:
        if not parsed_ip.is_global:
            raise WebFetchEgressViolation(
                f"Non-public IP host {host!r} is not allowed for web_fetch"
            )
        return

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise WebFetchEgressViolation(
            f"Could not resolve host {host!r}: {exc}"
        ) from exc

    for *_, sockaddr in infos:
        addr = sockaddr[0]
        try:
            resolved = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not resolved.is_global:
            raise WebFetchEgressViolation(
                f"Host {host!r} resolves to a non-public address ({resolved})"
            )


class _SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._href: str | None = None
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href or "uddg=" not in href:
            return
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        uddg = query.get("uddg", [""])[0]
        if not uddg:
            return
        self._href = unquote(uddg)
        self._title_parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        title = " ".join("".join(self._title_parts).split())
        if title and not any(result["url"] == self._href for result in self.results):
            self.results.append({"title": html.unescape(title), "url": self._href})
        self._href = None
        self._title_parts = []


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.text_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        elif not self._skip_depth:
            self.text_parts.append(text)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _request_text(request: MessagesRequest) -> str:
    return "\n".join(_content_text(message.content) for message in request.messages)


def _forced_web_server_tool_name(request: MessagesRequest) -> str | None:
    """Return web_search or web_fetch only when tool_choice forces that server tool."""
    tc = request.tool_choice
    if not isinstance(tc, dict):
        return None
    if tc.get("type") != "tool":
        return None
    name = tc.get("name")
    if name in {"web_search", "web_fetch"}:
        return str(name)
    return None


def _has_tool_named(request: MessagesRequest, name: str) -> bool:
    return any(tool.name == name for tool in request.tools or [])


def is_web_server_tool_request(request: MessagesRequest) -> bool:
    """True when the client forces a web server tool via tool_choice (not merely listed)."""
    forced = _forced_web_server_tool_name(request)
    if forced is None:
        return False
    return _has_tool_named(request, forced)


def _extract_query(text: str) -> str:
    match = re.search(r"query:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().strip("\"'")
    return text.strip()


def _extract_url(text: str) -> str:
    match = re.search(r"https?://\S+", text)
    return match.group(0).rstrip(").,]") if match else text.strip()


def _safe_public_host_for_logs(url: str) -> str:
    """Return hostname only (no path/query) for structured logs."""
    host = urlparse(url).hostname or ""
    return host[:253]


async def _drain_response_body_capped(response: httpx.Response, max_bytes: int) -> None:
    received = 0
    async for chunk in response.aiter_bytes():
        received += len(chunk)
        if received >= max_bytes:
            break


async def _read_response_body_capped(response: httpx.Response, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_bytes:
            break
    body = b"".join(chunks)
    return body[:max_bytes]


def _log_web_tool_failure(
    tool_name: str,
    error: BaseException,
    *,
    fetch_url: str | None = None,
) -> None:
    exc_type = type(error).__name__
    if isinstance(error, WebFetchEgressViolation):
        host = _safe_public_host_for_logs(fetch_url) if fetch_url else ""
        logger.warning(
            "web_tool_egress_rejected tool={} exc_type={} host={!r}",
            tool_name,
            exc_type,
            host,
        )
        return
    if tool_name == "web_fetch" and fetch_url:
        logger.warning(
            "web_tool_failure tool={} exc_type={} host={!r}",
            tool_name,
            exc_type,
            _safe_public_host_for_logs(fetch_url),
        )
    else:
        logger.warning("web_tool_failure tool={} exc_type={}", tool_name, exc_type)


def _web_tool_client_error_summary(
    tool_name: str,
    error: BaseException,
    *,
    verbose: bool,
) -> str:
    if verbose:
        return f"{tool_name} failed: {type(error).__name__}"
    return "Web tool request failed."


async def _run_web_search(query: str) -> list[dict[str, str]]:
    async with (
        httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT_S,
            follow_redirects=True,
            headers=_WEB_TOOL_HTTP_HEADERS,
        ) as client,
        client.stream(
            "GET",
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
        ) as response,
    ):
        response.raise_for_status()
        body_bytes = await _read_response_body_capped(
            response, _MAX_WEB_FETCH_RESPONSE_BYTES
        )
    text = body_bytes.decode("utf-8", errors="replace")
    parser = _SearchResultParser()
    parser.feed(text)
    return parser.results[:_MAX_SEARCH_RESULTS]


async def _run_web_fetch(url: str, egress: WebFetchEgressPolicy) -> dict[str, str]:
    """Fetch URL with manual redirects so each hop is validated by egress policy."""
    current_url = url
    redirect_hops = 0
    async with httpx.AsyncClient(
        timeout=_REQUEST_TIMEOUT_S,
        follow_redirects=False,
        headers=_WEB_TOOL_HTTP_HEADERS,
    ) as client:
        while True:
            await asyncio.to_thread(enforce_web_fetch_egress, current_url, egress)
            async with client.stream("GET", current_url) as response:
                if response.status_code in _WEB_FETCH_REDIRECT_STATUSES:
                    await _drain_response_body_capped(
                        response, _REDIRECT_RESPONSE_BODY_CAP_BYTES
                    )
                    if redirect_hops >= _MAX_WEB_FETCH_REDIRECTS:
                        raise WebFetchEgressViolation(
                            f"web_fetch exceeded maximum redirects ({_MAX_WEB_FETCH_REDIRECTS})"
                        )
                    location = response.headers.get("location")
                    if not location or not location.strip():
                        raise WebFetchEgressViolation(
                            "web_fetch redirect response missing Location header"
                        )
                    current_url = urljoin(str(response.url), location.strip())
                    redirect_hops += 1
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "text/plain")
                final_url = str(response.url)
                encoding = response.encoding or "utf-8"
                body_bytes = await _read_response_body_capped(
                    response, _MAX_WEB_FETCH_RESPONSE_BYTES
                )

            text = body_bytes.decode(encoding, errors="replace")
            title = final_url
            data = text
            if "html" in content_type.lower():
                parser = _HTMLTextParser()
                parser.feed(text)
                title = parser.title or final_url
                data = "\n".join(parser.text_parts)
            return {
                "url": final_url,
                "title": title,
                "media_type": "text/plain",
                "data": data[:_MAX_FETCH_CHARS],
            }


def _search_summary(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return f"No web search results found for: {query}"
    lines = [f"Search results for: {query}"]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}\n{result['url']}")
    return "\n\n".join(lines)


async def stream_web_server_tool_response(
    request: MessagesRequest,
    input_tokens: int,
    *,
    web_fetch_egress: WebFetchEgressPolicy,
    verbose_client_errors: bool = False,
) -> AsyncIterator[str]:
    tool_name = _forced_web_server_tool_name(request)
    if tool_name is None or not _has_tool_named(request, tool_name):
        return

    text = _request_text(request)
    message_id = f"msg_{uuid.uuid4()}"
    tool_id = f"srvtoolu_{uuid.uuid4().hex}"
    usage_key = (
        "web_search_requests" if tool_name == "web_search" else "web_fetch_requests"
    )
    tool_input = (
        {"query": _extract_query(text)}
        if tool_name == "web_search"
        else {"url": _extract_url(text)}
    )
    _result_block_for_tool = {
        "web_search": WEB_SEARCH_TOOL_RESULT,
        "web_fetch": WEB_FETCH_TOOL_RESULT,
    }
    _error_payload_type_for_tool = {
        "web_search": WEB_SEARCH_TOOL_RESULT_ERROR,
        "web_fetch": WEB_FETCH_TOOL_ERROR,
    }

    yield format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": request.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        },
    )
    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": SERVER_TOOL_USE,
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            },
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": 0}
    )

    try:
        if tool_name == "web_search":
            query = str(tool_input["query"])
            results = await _run_web_search(query)
            result_content: Any = [
                {
                    "type": "web_search_result",
                    "title": result["title"],
                    "url": result["url"],
                }
                for result in results
            ]
            summary = _search_summary(query, results)
            result_block_type = WEB_SEARCH_TOOL_RESULT
        else:
            fetched = await _run_web_fetch(str(tool_input["url"]), web_fetch_egress)
            result_content = {
                "type": "web_fetch_result",
                "url": fetched["url"],
                "content": {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": fetched["media_type"],
                        "data": fetched["data"],
                    },
                    "title": fetched["title"],
                    "citations": {"enabled": True},
                },
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
            summary = fetched["data"][:_MAX_FETCH_CHARS]
            result_block_type = WEB_FETCH_TOOL_RESULT
    except Exception as error:
        fetch_url = str(tool_input["url"]) if tool_name == "web_fetch" else None
        _log_web_tool_failure(tool_name, error, fetch_url=fetch_url)
        result_block_type = _result_block_for_tool[tool_name]
        result_content = {
            "type": _error_payload_type_for_tool[tool_name],
            "error_code": "unavailable",
        }
        summary = _web_tool_client_error_summary(
            tool_name, error, verbose=verbose_client_errors
        )

    output_tokens = max(1, len(summary) // 4)

    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": result_block_type,
                "tool_use_id": tool_id,
                "content": result_content,
            },
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": 1}
    )
    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": summary},
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": 2}
    )
    yield format_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "server_tool_use": {usage_key: 1},
            },
        },
    )
    yield format_sse_event("message_stop", {"type": "message_stop"})
