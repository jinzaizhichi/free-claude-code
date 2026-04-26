"""Outbound HTTP for web_search / web_fetch (client, body caps, logging)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger

from . import constants
from .constants import (
    _MAX_FETCH_CHARS,
    _MAX_SEARCH_RESULTS,
    _REDIRECT_RESPONSE_BODY_CAP_BYTES,
    _REQUEST_TIMEOUT_S,
    _WEB_FETCH_REDIRECT_STATUSES,
    _WEB_TOOL_HTTP_HEADERS,
)
from .egress import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    enforce_web_fetch_egress,
)
from .parsers import HTMLTextParser, SearchResultParser


def _safe_public_host_for_logs(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host[:253]


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


async def _iter_response_body_under_cap(
    response: httpx.Response, max_bytes: int
) -> AsyncIterator[bytes]:
    if max_bytes <= 0:
        return
    received = 0
    async for chunk in response.aiter_bytes(chunk_size=65_536):
        if received >= max_bytes:
            break
        remaining = max_bytes - received
        if len(chunk) <= remaining:
            received += len(chunk)
            yield chunk
            if received >= max_bytes:
                break
        else:
            yield chunk[:remaining]
            break


async def _drain_response_body_capped(response: httpx.Response, max_bytes: int) -> None:
    async for _ in _iter_response_body_under_cap(response, max_bytes):
        pass


async def _read_response_body_capped(response: httpx.Response, max_bytes: int) -> bytes:
    return b"".join(
        [piece async for piece in _iter_response_body_under_cap(response, max_bytes)]
    )


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
            response, constants._MAX_WEB_FETCH_RESPONSE_BYTES
        )
    text = body_bytes.decode("utf-8", errors="replace")
    parser = SearchResultParser()
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
                    if redirect_hops >= constants._MAX_WEB_FETCH_REDIRECTS:
                        raise WebFetchEgressViolation(
                            "web_fetch exceeded maximum redirects "
                            f"({constants._MAX_WEB_FETCH_REDIRECTS})"
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
                    response, constants._MAX_WEB_FETCH_RESPONSE_BYTES
                )

            text = body_bytes.decode(encoding, errors="replace")
            title = final_url
            data = text
            if "html" in content_type.lower():
                parser = HTMLTextParser()
                parser.feed(text)
                title = parser.title or final_url
                data = "\n".join(parser.text_parts)
            return {
                "url": final_url,
                "title": title,
                "media_type": "text/plain",
                "data": data[:_MAX_FETCH_CHARS],
            }
