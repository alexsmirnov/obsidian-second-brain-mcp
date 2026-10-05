"""GitHub fetchers: raw file contents for blobs, README for repositories."""

from __future__ import annotations

import logging
from dataclasses import replace
from urllib.parse import urlparse

from mcps.research.tools.common import MIME_MARKDOWN, MIME_PLAIN, failure
from mcps.research.tools.models import Fetch, FetchResult, FetchStatus

__all__ = [
    "GitHubBlobFetch",
    "GitHubRepoFetch",
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


def _as_requested_url(url: str, result: FetchResult) -> FetchResult:
    """Retarget any delegated result at the requested GitHub URL."""
    return replace(result, url=url)


def _is_known_markdown_path(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".md") or path.endswith(".markdown")


def _as_github_content(
    url: str, result: FetchResult, content: str, source_url: str
) -> FetchResult:
    """Retarget at the requested URL, promoting known raw Markdown files."""
    mime = result.mime
    if mime == MIME_PLAIN and _is_known_markdown_path(source_url):
        mime = MIME_MARKDOWN
    return replace(result, url=url, content=content, mime=mime)


class GitHubBlobFetch:
    """Raw file contents for ``github.com/.../blob/...`` URLs."""

    def __init__(self, http: Fetch) -> None:
        self._http = http

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        raw_url = _github_blob_to_raw_url(url)
        if raw_url is None:
            logger.warning("GitHub blob fetch failed for %s: unsupported", url)
            return failure(url, FetchStatus.UNSUPPORTED)
        result = await self._http(raw_url, query)
        if not result.ok:
            return _as_requested_url(url, result)
        content = result.content.strip()
        if not content:
            logger.warning("GitHub blob fetch failed for %s: empty", url)
            return failure(url, FetchStatus.EMPTY)
        return _as_github_content(url, result, content, raw_url)


class GitHubRepoFetch:
    """README for ``github.com/<owner>/<repo>`` URLs."""

    def __init__(self, http: Fetch) -> None:
        self._http = http

    async def __call__(self, url: str, query: str | None = None, /) -> FetchResult:
        for readme_url in _github_repo_readme_urls(url):
            result = await self._http(readme_url, query)
            if result.ok:
                content = result.content.strip()
                if content:
                    return _as_github_content(url, result, content, readme_url)
                continue
            if result.status is FetchStatus.EMPTY or (
                result.status is FetchStatus.HTTP_ERROR
                and result.http_status == 404
            ):
                continue
            # A rate limit, outage, or block is terminal: retrying the
            # remaining candidates cannot succeed.
            return _as_requested_url(url, result)
        logger.warning("GitHub repo fetch failed for %s: empty", url)
        return failure(url, FetchStatus.EMPTY)
