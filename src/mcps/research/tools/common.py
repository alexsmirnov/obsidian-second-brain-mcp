"""HTTP helpers, error strings, and output formatting shared by fetchers."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

import httpx

from mcps.research.tools.result import FetchResult, FetchStatus

__all__ = [
    "CHROME_HEADERS",
    "ERROR_EMPTY_RESPONSE",
    "ERROR_FETCHER_UNAVAILABLE",
    "ERROR_FILTERING",
    "ERROR_REQUEST_TIMEOUT",
    "ERROR_UNSUPPORTED_CONTENT",
    "MIME_HTML",
    "MIME_MARKDOWN",
    "Fetch",
    "Retrieve",
    "error_result",
    "extract_hostname",
    "failure",
    "format_source_output",
    "http_status_error",
    "is_escalatable",
    "request_get",
    "safe_fetch",
    "to_error_message",
]

logger = logging.getLogger(__name__)

MIME_HTML = "text/html"
MIME_MARKDOWN = "text/markdown"

Fetch = Callable[[str, str | None], Awaitable[str]]
# A single-URL source callable (browser render or commercial provider).
Retrieve = Callable[[str], Awaitable[str]]

# Collected from actual Chrome request headers
CHROME_HEADERS = {
    "accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;"
        "q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "accept-encoding": "gzip, deflate",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "dnt": "1",
    "pragma": "no-cache",
    "priority": "u=0, i",
    "sec-ch-ua": (
        '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"'
    ),
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0.0.0 Safari/537.36"
    ),
}

ERROR_REQUEST_TIMEOUT = "ERROR: request timeout"
ERROR_UNSUPPORTED_CONTENT = "ERROR: unsupported content"
ERROR_EMPTY_RESPONSE = "ERROR: empty response"
# A fallback fetcher could not run (browser down, provider out of credits).
# Says nothing about the target page, so the chain keeps the previous error.
ERROR_FETCHER_UNAVAILABLE = "ERROR: fetcher unavailable"
# The post-retrieval content filter itself failed (not the target page).
ERROR_FILTERING = "ERROR: content filtering failed"

# Blocked, auth-walled, rate-limited, or JS-rendered (empty) pages: a
# browser or unblocking provider may succeed where plain httpx did not.
_ESCALATABLE_STATUS_CODES = (401, 403, 429)


def http_status_error(status_code: int) -> str:
    """Return the error string for an HTTP error status."""
    return f"ERROR: http code {status_code}"


_ESCALATABLE_ERRORS = frozenset(
    [
        *(http_status_error(code) for code in _ESCALATABLE_STATUS_CODES),
        ERROR_EMPTY_RESPONSE,
        ERROR_REQUEST_TIMEOUT,
        ERROR_FETCHER_UNAVAILABLE,
    ]
)


def is_escalatable(result: str) -> bool:
    """Return True when a fetch result warrants trying the next fetcher."""
    return result in _ESCALATABLE_ERRORS


def to_error_message(error: Exception) -> str:
    """Map an httpx exception to a fetch error string."""
    if isinstance(error, httpx.TimeoutException):
        return ERROR_REQUEST_TIMEOUT
    if isinstance(error, httpx.HTTPStatusError):
        return http_status_error(error.response.status_code)
    return ERROR_UNSUPPORTED_CONTENT


def failure(
    url: str, status: FetchStatus, http_status: int | None = None
) -> FetchResult:
    """Build an unsuccessful result; error results carry no mime."""
    return FetchResult(url=url, status=status, mime="", http_status=http_status)


def error_result(url: str, error: Exception) -> FetchResult:
    """Map an httpx exception to a failed result."""
    if isinstance(error, httpx.TimeoutException):
        return failure(url, FetchStatus.TIMEOUT)
    if isinstance(error, httpx.HTTPStatusError):
        return failure(url, FetchStatus.HTTP_ERROR, error.response.status_code)
    return failure(url, FetchStatus.UNSUPPORTED)


async def safe_fetch(url: str, attempt: Awaitable[FetchResult]) -> FetchResult:
    """Await ``attempt``; any exception becomes a failed result, never raised.

    ``deep_research`` gathers fetches without ``return_exceptions``, so one
    bad URL must not fail the whole search step.
    """
    try:
        return await attempt
    except httpx.HTTPError as error:
        result = error_result(url, error)
    except Exception:
        result = failure(url, FetchStatus.UNSUPPORTED)
    logger.warning("Web fetch failed for %s: %s", url, result.status)
    return result


def format_source_output(url: str, content: str, max_chars: int) -> str:
    """Log fetched size and truncate content to ``max_chars``."""
    logger.info("Fetch %d chars from url %s", len(content), url)
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n[Content truncated]"


def extract_hostname(url: str) -> str:
    """Return the lowercase network location of ``url``."""
    return urlparse(url).netloc.lower()


async def request_get(
    url: str,
    *,
    http_client: httpx.AsyncClient | None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """GET ``url`` with the shared client (or an owned one) and raise on 4xx/5xx."""
    if http_client is not None:
        response = await http_client.get(url, headers=headers)
        response.raise_for_status()
        return response
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as owned:
        response = await owned.get(url, headers=headers)
        response.raise_for_status()
        return response
