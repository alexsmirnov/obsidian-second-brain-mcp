"""GitHub fetchers: raw file contents for blobs, README for repositories."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx

from mcps.research.tools.common import (
    CHROME_HEADERS,
    MIME_MARKDOWN,
    failure,
    request_get,
    safe_fetch,
)
from mcps.research.tools.result import FetchResult, FetchStatus

__all__ = [
    "GitHubBlobFetch",
    "GitHubRepoFetch",
    "github_link_base",
    "is_github_blob_url",
    "is_github_repo_url",
]

logger = logging.getLogger(__name__)

_RAW_BASE = "https://raw.githubusercontent.com"
_README_BRANCHES = ("main", "master")
_README_FILES = ("README.md", "README.rst", "README.txt", "README")


def _github_path_parts(url: str) -> list[str] | None:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "github.com":
        return None
    return [part for part in parsed.path.split("/") if part]


def is_github_blob_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.lower() == "github.com" and "/blob/" in parsed.path


def is_github_repo_url(url: str) -> bool:
    parts = _github_path_parts(url)
    return parts is not None and len(parts) == 2


def github_link_base(url: str) -> str:
    """Base URL for resolving relative links in a fetched GitHub source."""
    return f"{url.rstrip('/')}/blob/HEAD/" if is_github_repo_url(url) else url


def _github_blob_to_raw_url(url: str) -> str | None:
    parts = _github_path_parts(url) or []
    if len(parts) < 5 or parts[2] != "blob":
        return None
    owner, repo, _, branch, *file_parts = parts
    return f"{_RAW_BASE}/{owner}/{repo}/{branch}/{'/'.join(file_parts)}"


def _github_repo_readme_urls(url: str) -> list[str]:
    parts = _github_path_parts(url) or []
    if len(parts) < 2:
        return []
    owner, repo = parts[0], parts[1]
    return [
        f"{_RAW_BASE}/{owner}/{repo}/{branch}/{file_name}"
        for branch in _README_BRANCHES
        for file_name in _README_FILES
    ]


def _markdown_result(url: str, content: str) -> FetchResult:
    return FetchResult(
        url=url,
        status=FetchStatus.OK,
        mime=MIME_MARKDOWN,
        content=content,
        base_url=github_link_base(url),
    )


class GitHubBlobFetch:
    """Raw file contents for ``github.com/.../blob/...`` URLs."""

    def __init__(self, http_client: httpx.AsyncClient | None) -> None:
        self._http_client = http_client

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        return await safe_fetch(url, self._fetch(url))

    async def _fetch(self, url: str) -> FetchResult:
        raw_url = _github_blob_to_raw_url(url)
        if raw_url is None:
            logger.warning("GitHub blob fetch failed for %s: unsupported", url)
            return failure(url, FetchStatus.UNSUPPORTED)
        response = await request_get(
            raw_url, http_client=self._http_client, headers=CHROME_HEADERS
        )
        content = response.text.strip()
        if not content:
            logger.warning("GitHub blob fetch failed for %s: empty", url)
            return failure(url, FetchStatus.EMPTY)
        return _markdown_result(url, content)


class GitHubRepoFetch:
    """README for ``github.com/<owner>/<repo>`` URLs."""

    def __init__(self, http_client: httpx.AsyncClient | None) -> None:
        self._http_client = http_client

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        return await safe_fetch(url, self._fetch(url))

    async def _fetch(self, url: str) -> FetchResult:
        for readme_url in _github_repo_readme_urls(url):
            try:
                response = await request_get(
                    readme_url, http_client=self._http_client, headers=CHROME_HEADERS
                )
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 404:
                    continue
                raise
            content = response.text.strip()
            if content:
                return _markdown_result(url, content)
        logger.warning("GitHub repo fetch failed for %s: empty", url)
        return failure(url, FetchStatus.EMPTY)
